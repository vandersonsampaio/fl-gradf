"""
W1 (results/w1_protocolo_independente_oficial/PLANO.md, frozen 2026-10-23): Independent
Evaluation Protocol in the official AdaAggRL. Nothing in external/AdaAggRL is edited.

Protocol (PLANO §1), installed as a wrapper around `exp_environments.construct_dataloaders`
(after `bloodmnist_shim.install`, so it wraps the BloodMNIST loader too):
  - validation = a held-out part of the official NON-test pool, of size |official test|,
    indices from np.random.RandomState(20261023) (independent of the run seed and of
    Python's `random`, whose state after random.seed(150) drives the official partition),
    with the non-augmented (official test) transform:
        MNIST       10,000 of the 60,000 training images  -> 50,000 left for the clients
        BloodMNIST  3,421 of official train + val (13,671) -> 10,250 left for the clients
  - the environment receives (train_rest, validation) in place of (trainset, testset), so
    the reward, the resets, the EB attacker's slices and `history_acc` all use validation;
  - the official test set is used only by `HeldOutAdapter`, which evaluates the global
    model after every step with a dedicated torch.Generator and with the torch (CPU/CUDA),
    numpy and random states saved and restored around the evaluation (logging only).

Conditions: td3 (official main.py, via run_b21's actor checkpoints and states),
fixed ([0.475]*5 = b*, PLANO §2) and fedavg (uniform aggregation, attack none; reference).

Usage (official venv):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_w1.py run --dataset MNIST --attack EB --condition td3 --seed 160
  CUDA_VISIBLE_DEVICES="" external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_w1.py verificar
"""

import argparse
import contextlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_b21 as B21  # noqa: E402
import run_b3 as B3  # noqa: E402
import bloodmnist_shim as BS  # noqa: E402

R = B21.R
import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402

SPLIT_SEED = 20261023
OUT = os.path.join(R.REPO, "results", "w1_protocolo_independente_oficial")
SEEDS = {"MNIST": list(range(160, 170)), "BloodMNIST": list(range(170, 180))}
FEDAVG_SEEDS = {"MNIST": [160, 161, 162], "BloodMNIST": [170, 171, 172]}
HELD = {}  # filled by the construct_dataloaders wrapper: official test set and split info


def split_indices(n_pool, n_val):
    perm = np.random.RandomState(SPLIT_SEED).permutation(n_pool)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def _mnist_split(orig_cd, data_path):
    tr_aug, test = orig_cd("MNIST", augmentations=True, data_path=data_path)
    tr_plain, _ = orig_cd("MNIST", augmentations=False, data_path=data_path)
    rest, val = split_indices(len(tr_aug), len(test))
    return (torch.utils.data.Subset(tr_aug, rest), torch.utils.data.Subset(tr_plain, val), test,
            {"pool": len(tr_aug), "train_rest": len(rest), "val": len(val), "test": len(test)})


def _blood_split():
    d = np.load(BS.NPZ)
    tr_ds, test = BS.build_bloodmnist(True, True)   # transforms (stats from the official train split)
    plain_tf = test.transform
    imgs = np.concatenate([d["train_images"], d["val_images"]])
    labs = np.concatenate([d["train_labels"], d["val_labels"]])
    rest, val = split_indices(len(labs), len(test))
    return (BS.BloodMNISTDataset(imgs[rest], labs[rest], tr_ds.transform),
            BS.BloodMNISTDataset(imgs[val], labs[val], plain_tf), test,
            {"pool": len(labs), "train_rest": len(rest), "val": len(val), "test": len(test)})


def install_split(E, dataset):
    # keep the first (official or BloodMNIST-shim) loader, so a second run in the same process
    # (verificar) does not wrap the wrapper
    if not hasattr(E, "_w1_orig_cd"):
        E._w1_orig_cd = E.construct_dataloaders
    orig = E._w1_orig_cd

    def construct_dataloaders(ds, *a, **k):
        if ds == "BloodMNIST":
            train_rest, val, test, info = _blood_split()
        elif ds == "MNIST":
            train_rest, val, test, info = _mnist_split(orig, k.get("data_path", "./data"))
        else:
            raise ValueError(ds)
        HELD.update({"test": test, "info": {**info, "split_seed": SPLIT_SEED}})
        return train_rest, val
    E.construct_dataloaders = construct_dataloaders
    return E


@contextlib.contextmanager
def preserved_rng():
    states = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
              np.random.get_state(), random.getstate())
    try:
        yield
    finally:
        torch.set_rng_state(states[0])
        if states[1] is not None:
            torch.cuda.set_rng_state_all(states[1])
        np.random.set_state(states[2])
        random.setstate(states[3])


class HeldOutAdapter(B21.ObsRecordingAdapter):
    """ObsRecordingAdapter plus the held-out test evaluation after every step."""

    def __init__(self, inner, log_stream, E, evaluate=True):
        super().__init__(inner, log_stream)
        self.E = E
        self.evaluate = evaluate
        self.test_hist = []
        self.loader = torch.utils.data.DataLoader(HELD["test"], batch_size=inner.args.batch_size, shuffle=False,
                                                  drop_last=True, generator=torch.Generator().manual_seed(0))

    def step(self, action):
        out = super().step(action)
        if self.evaluate:
            with preserved_rng():
                loss, acc = self.E.test(self.inner.net, self.loader)
            self.test_hist.append({"t": self._t, "test_loss": float(loss), "test_acc": float(acc)})
        return out


def run(dataset, attack, condition, seed, rounds, out_dir, q=0.5, evaluate=True):
    out_dir = os.path.abspath(out_dir)
    for sub in ("", "obs", "actors"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    tag = f"{dataset}_{attack}_q{q}_{condition}_seed{seed}_R{rounds}"
    out_path = os.path.join(out_dir, f"{tag}.json")
    if os.path.exists(out_path):
        print(f"already exists: {out_path}")
        return out_path
    partial_path = os.path.join(out_dir, f"{tag}.partial.json")
    log_path = os.path.join(out_dir, f"{tag}.stdout.log")

    cwd = os.getcwd()
    os.chdir(R.OFICIAL)
    try:
        import exp_environments as E
        E.SummaryWriter = R._NoOpWriter
        BS.install(E, dataset)
        install_split(E, dataset)
        if condition == "fedavg":
            E.aggeregate = lambda new_weights, fractions: E.average(new_weights)
            R._preview_weights = B3._uniform_preview
        args = BS.adapt_args(R._official_args(dataset, B3.ENV_ATTACK[attack], q))

        t_start = time.perf_counter()
        with open(log_path, "w", buffering=1) as log_stream:
            with contextlib.redirect_stdout(log_stream):
                inner = E.FL_mnist(args)  # random.seed(150)
            inner.args = args
            if attack == "none":
                inner.att_ids = []
            set_random_seed(seed, using_cuda=torch.cuda.is_available())
            env = HeldOutAdapter(inner, log_stream, E, evaluate=evaluate)
            env.action_space.seed(seed)

            def _rec():
                rec = R._record(inner, env, E, attack, condition, seed, rounds, q, dataset,
                                time.perf_counter() - t_start)
                rec.update({"protocol": "W1 independent evaluation", "split": HELD["info"],
                            "n_val_batches": len(inner.testloader), "n_test_batches": len(env.loader),
                            "held_out": env.test_hist, "num_clients": args.num_clients,
                            "subsample_rate": args.subsample_rate, "num_class": args.num_class})
                return rec

            def _checkpoint(t):
                if t % R.CHECKPOINT_EVERY == 0:
                    R._dump(_rec(), partial_path)
            env.on_step = _checkpoint

            if condition == "td3":
                n_actions = env.action_space.shape[-1]
                noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                model = R.TD3("MlpPolicy", env, buffer_size=1000, policy_kwargs={"net_arch": [256, 128]},
                              verbose=0, gamma=0.99, action_noise=noise, learning_rate=1e-5,
                              train_freq=(3, "step"), batch_size=64, seed=seed)
                model.learn(total_timesteps=rounds, log_interval=None,
                            callback=B21.ActorCheckpoint(os.path.join(out_dir, "actors", tag)))
            else:
                env.reset()
                for _ in range(rounds):
                    _o, _r, term, _t, _ = env.step(np.asarray(R.A_FIXED, dtype=np.float32))
                    if term:
                        env.reset()
        rec = _rec()
    finally:
        os.chdir(cwd)

    np.save(os.path.join(out_dir, "obs", f"{tag}.npy"), np.stack(env.obs_in).astype(np.float32))
    R._dump(rec, out_path)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    return out_path


def verificar(rounds=5):
    """PLANO §6 checks 1-3 (CPU; run with CUDA_VISIBLE_DEVICES=""). Check 4 (10-round GPU
    smoke per dataset) is `run ... --rounds 10 --out_dir <verificacao>` for td3 and fixed."""
    lines = []
    cwd = os.getcwd()
    os.chdir(R.OFICIAL)
    try:
        import exp_environments as E
        orig = E.construct_dataloaders
        for name, fn in (("MNIST", lambda: _mnist_split(orig, "./data")), ("BloodMNIST", _blood_split)):
            if name == "BloodMNIST" and not os.path.exists(BS.NPZ):
                lines.append(f"[1] {name}: npz not found ({BS.NPZ}) - SKIPPED")
                continue
            tr, val, test, info = fn()
            n_pool = info["pool"]
            rest, vidx = split_indices(n_pool, info["test"])
            disjoint = len(np.intersect1d(rest, vidx)) == 0 and len(rest) + len(vidx) == n_pool
            ok = disjoint and len(val) == len(test) and len(tr) == n_pool - len(test)
            lines.append(f"[1] {name}: {info}; val ∩ train_rest = ∅ and val ⊂ non-test pool: "
                         f"{'OK' if ok else 'FAIL'}")
            nb = lambda ds: len(torch.utils.data.DataLoader(ds, batch_size=64, drop_last=True))  # noqa: E731
            lines.append(f"[3] {name}: validation batches {nb(val)} vs test batches {nb(test)} "
                         + ("OK" if nb(val) == nb(test) else "FAIL"))
    finally:
        os.chdir(cwd)

    vdir = os.path.join(OUT, "verificacao")
    paths = [run("MNIST", "EB", "fixed", 160, rounds, os.path.join(vdir, sub), evaluate=ev)
             for sub, ev in (("with_eval", True), ("without_eval", False))]
    a, b = (json.load(open(p)) for p in paths)
    same = a["history_acc"] == b["history_acc"] and a["history_loss"] == b["history_loss"]
    lines.append(f"[2] MNIST EB fixed seed 160, {rounds} rounds on {a['device']}: histories with vs without "
                 f"held-out evaluation identical: {'OK' if same else 'FAIL'} "
                 f"(held-out acc at the last step {a['held_out'][-1]['test_acc']:.4f})")
    text = "\n".join(lines) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "verificacao.txt"), "w").write(text)
    print(text)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True, choices=["MNIST", "BloodMNIST"])
    r.add_argument("--attack", required=True, choices=list(B3.ENV_ATTACK))
    r.add_argument("--condition", required=True, choices=["fedavg", "fixed", "td3"])
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--rounds", type=int, default=500)
    r.add_argument("--out_dir", default=os.path.join(OUT, "raw"))
    v = sub.add_parser("verificar")
    v.add_argument("--rounds", type=int, default=5)
    a = p.parse_args()
    if a.cmd == "verificar":
        verificar(a.rounds)
        return
    allowed = FEDAVG_SEEDS if a.condition == "fedavg" else SEEDS
    if a.out_dir == os.path.join(OUT, "raw") and a.seed not in allowed[a.dataset]:
        raise SystemExit(f"W1 seeds for {a.dataset}/{a.condition}: {allowed[a.dataset]} (PLANO §3)")
    if a.condition == "fedavg" and a.attack != "none":
        raise SystemExit("fedavg is the no-attack reference (PLANO §3)")
    print(run(a.dataset, a.attack, a.condition, a.seed, a.rounds, a.out_dir))


if __name__ == "__main__":
    main()
