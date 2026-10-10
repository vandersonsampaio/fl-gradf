"""
B2.5: IPM attack in the official AdaAggRL.
Plan in results/b25_ipm_oficial/PLANO.md.

The official code's IPM is a NULL update: IPM_attack calls
craft(old_weights, get_parameters(net), 5, -1) with the network already set to
old_weights, so weight_diff = 0 and the attacker sends the global model itself.
Here:
  IPM_oficial   the repository's attack, unchanged (to document that it is null)
  IPM_real      Xie et al.'s IPM: crafted delta = −ε × mean of the round's honest deltas
                (omniscient attacker); sent weights = old + crafted delta
  none          no attack (same partition: attackers are sampled as in the
                official code and then emptied, to preserve the RNG)

Implementation, without editing the official code: the official `IPM_attack` does not receive the
honest updates, but `LMP_attack` does (the list of honest weights already
computed in the round). For IPM_real, the environment runs on the LMP path and, in this
process, `exp_environments.LMP_attack` is replaced by the real-IPM function. The JSON
records the effective attack (`attack_label`) and ε.

Conditions:
  td3      TD3 as in the official main.py (via run_b21: states and actor checkpoints)
  fixed    action [0.475]*5 (center of the Box), as in B2.1
  fedavg   uniform aggregation (without the official filter; for the sanity check only)

Usage (official venv):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_b25.py --attack IPM_real --eps 10 --condition fixed --seed 100 --rounds 100
"""

import argparse
import contextlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_b21 as B21  # noqa: E402

R = B21.R
import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402

DEFAULT_OUT = os.path.join(R.REPO, "results", "b25_ipm_oficial", "raw")
ENV_ATTACK = {"IPM_real": "LMP", "IPM_oficial": "IPM", "none": "LMP"}


def ipm_real_attack(eps: float):
    """Returns a function with the official LMP_attack signature that implements the real IPM."""
    def _attack(net, old_weights, cids, att_ids, trainloaders, weights_lis):
        print(f"----------Real IPM (B2.5, eps={eps:g}) Attack--------------")
        n_layers = len(old_weights)
        mean_delta = [np.mean([np.asarray(w[i], dtype=np.float64) - old_weights[i] for w in weights_lis], axis=0)
                      for i in range(n_layers)]
        crafted = [(old_weights[i] - eps * mean_delta[i]).astype(old_weights[i].dtype) for i in range(n_layers)]
        return {cid: [c.copy() for c in crafted] for cid in att_ids}
    return _attack


def _uniform_preview(env, action):
    n = len(env.cids)
    return np.ones(n) / n


def run(attack_label: str, eps: float, condition: str, seed: int, rounds: int, out_dir: str,
        q: float = 0.5, dataset: str = "MNIST") -> str:
    out_dir = os.path.abspath(out_dir)
    for sub in ("", "obs", "actors"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    alabel = f"IPMr{eps:g}" if attack_label == "IPM_real" else attack_label
    tag = f"{dataset}_{alabel}_q{q}_{condition}_seed{seed}_R{rounds}"
    out_path = os.path.join(out_dir, f"{tag}.json")
    partial_path = os.path.join(out_dir, f"{tag}.partial.json")
    log_path = os.path.join(out_dir, f"{tag}.stdout.log")

    cwd = os.getcwd()
    os.chdir(R.OFICIAL)
    try:
        import exp_environments as E
        E.SummaryWriter = R._NoOpWriter
        if attack_label == "IPM_real":
            E.LMP_attack = ipm_real_attack(eps)
        if condition == "fedavg":
            E.aggeregate = lambda new_weights, fractions: E.average(new_weights)
            R._preview_weights = _uniform_preview

        t_start = time.perf_counter()
        with open(log_path, "w", buffering=1) as log_stream:
            with contextlib.redirect_stdout(log_stream):
                inner = E.FL_mnist(R._official_args(dataset, ENV_ATTACK[attack_label], q))  # random.seed(150)
            if attack_label == "none":
                inner.att_ids = []  # sampling done (RNG preserved), attackers emptied
            set_random_seed(seed, using_cuda=torch.cuda.is_available())
            env = B21.ObsRecordingAdapter(inner, log_stream)
            env.action_space.seed(seed)

            def _checkpoint(t):
                if t % R.CHECKPOINT_EVERY == 0:
                    R._dump(R._record(inner, env, E, alabel, condition, seed, rounds, q, dataset,
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
    rec = R._record(inner, env, E, alabel, condition, seed, rounds, q, dataset, elapsed)
    rec.update({"attack_label": attack_label, "eps": eps if attack_label == "IPM_real" else None,
                "env_attack_path": ENV_ATTACK[attack_label]})
    R._dump(rec, out_path)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", required=True, choices=list(ENV_ATTACK))
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--condition", required=True, choices=["td3", "fixed", "fedavg"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    a = p.parse_args()
    if a.attack == "IPM_real" and a.eps is None:
        p.error("--eps is required for IPM_real")
    print(run(a.attack, a.eps, a.condition, a.seed, a.rounds, a.out_dir))


if __name__ == "__main__":
    main()
