"""
B3.0 / B3.1 (results/b31_medmnist_oficial/PREREGISTRO.md): AdaAggRL oficial em
MNIST (inalterado) ou BloodMNIST (acréscimo via bloodmnist_shim; para MNIST o shim
não faz nada).

Condições:
  fedavg   agregação uniforme (troca de `aggeregate` por `average` neste processo,
           como no B2.5) — só B3.0
  fixed    ação [0,475]*5 (centro, como no B2.1)
  td3      TD3 como no main.py oficial (via run_b21: estados e checkpoints do ator)
Ataques: none (atacantes sorteados como no oficial e esvaziados, para preservar o
RNG; como no B2.5), LMP e EB (oficiais, sem alteração).

Uso (venv oficial):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_b3.py --dataset BloodMNIST --attack none --condition fedavg --seed 130 --out_dir results/b30_medmnist_sanity/raw
"""

import argparse
import contextlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_b21 as B21  # noqa: E402
import bloodmnist_shim as BS  # noqa: E402

R = B21.R
import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402

ENV_ATTACK = {"none": "LMP", "LMP": "LMP", "EB": "EB"}


def _uniform_preview(env, action):
    n = len(env.cids)
    return np.ones(n) / n


def run(dataset, attack, condition, seed, rounds, out_dir, q=0.5):
    out_dir = os.path.abspath(out_dir)
    for sub in ("", "obs", "actors"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    tag = f"{dataset}_{attack}_q{q}_{condition}_seed{seed}_R{rounds}"
    out_path = os.path.join(out_dir, f"{tag}.json")
    if os.path.exists(out_path):
        print(f"já existe: {out_path}")
        return out_path
    partial_path = os.path.join(out_dir, f"{tag}.partial.json")
    log_path = os.path.join(out_dir, f"{tag}.stdout.log")

    cwd = os.getcwd()
    os.chdir(R.OFICIAL)
    try:
        import exp_environments as E
        E.SummaryWriter = R._NoOpWriter
        BS.install(E, dataset)  # no-op para MNIST
        if condition == "fedavg":
            E.aggeregate = lambda new_weights, fractions: E.average(new_weights)
            R._preview_weights = _uniform_preview
        args = BS.adapt_args(R._official_args(dataset, ENV_ATTACK[attack], q))  # no-op para MNIST

        t_start = time.perf_counter()
        with open(log_path, "w", buffering=1) as log_stream:
            with contextlib.redirect_stdout(log_stream):
                inner = E.FL_mnist(args)  # random.seed(150)
            if attack == "none":
                inner.att_ids = []
            set_random_seed(seed, using_cuda=torch.cuda.is_available())
            env = B21.ObsRecordingAdapter(inner, log_stream)
            env.action_space.seed(seed)

            def _checkpoint(t):
                if t % R.CHECKPOINT_EVERY == 0:
                    R._dump(R._record(inner, env, E, attack, condition, seed, rounds, q, dataset,
                                      time.perf_counter() - t_start), partial_path)
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
        elapsed = time.perf_counter() - t_start
    finally:
        os.chdir(cwd)

    np.save(os.path.join(out_dir, "obs", f"{tag}.npy"), np.stack(env.obs_in).astype(np.float32))
    rec = R._record(inner, env, E, attack, condition, seed, rounds, q, dataset, elapsed)
    rec.update({"num_clients": args.num_clients, "subsample_rate": args.subsample_rate,
                "num_class": args.num_class,
                "extractor": BS.EXTRACTOR_PATH if dataset == "BloodMNIST" else "extract_feature.pt (oficial)"})
    R._dump(rec, out_path)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["MNIST", "BloodMNIST"])
    p.add_argument("--attack", required=True, choices=list(ENV_ATTACK))
    p.add_argument("--condition", required=True, choices=["fedavg", "fixed", "td3"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    print(run(a.dataset, a.attack, a.condition, a.seed, a.rounds, a.out_dir))


if __name__ == "__main__":
    main()
