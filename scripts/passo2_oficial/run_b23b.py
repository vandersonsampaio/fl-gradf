"""
B2.3b (results/b23b_steelman_normalizado/PLANO.md): LAST TD3 steelman in the
official AdaAggRL. Question: with a well-conditioned reward, does TD3 learn
a state-dependent policy and beat the fixed action?

Motivation: in B2.3 (lr 1e-3, learning_starts 10) the policy saturated in a constant
corner of the Box. The conditioning hypothesis is the scale of the official
reward (SUM of the loss over ~156 test batches, tens to hundreds per
round, without normalization).

Changes relative to the official TD3 (main.py):
  reward          normalized by SB3's VecNormalize (norm_reward=True,
                  gamma=0.99, clip 10); observations NOT normalized, so the
                  actor can still be evaluated with the B2.2 tools
  learning_rate   1e-5 -> 1e-4   (10x; actor and critic)
  learning_starts 100  -> 10     (same as B2.3)
Everything else is identical to the official code and to B2.3. The reward recorded in the JSON
is still the raw one (the adapter records it before the wrapper).

Reuses `run_b21.run` without changing it (state log and actor checkpoints),
replacing only the TD3 constructor in this process.

Usage (always with the isolated venv):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_b23b.py --attack EB --seed 100
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_b21 as B21  # noqa: E402

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

STEELMAN = {"learning_rate": 1e-4, "learning_starts": 10}
VECNORM = {"norm_obs": False, "norm_reward": True, "gamma": 0.99, "clip_reward": 10.0}
DEFAULT_OUT = os.path.join(B21.R.REPO, "results", "b23b_steelman_normalizado", "raw")

_official_td3 = B21.R.TD3
_used = {}


def _steelman_td3(policy, env, **kwargs):
    """Replaces lr and learning_starts and wraps the environment in a reward-only VecNormalize."""
    kwargs.update(STEELMAN)
    venv = VecNormalize(DummyVecEnv([lambda: env]), **VECNORM)
    _used.clear()
    _used.update({k: v for k, v in kwargs.items() if k != "action_noise"})
    _used["vecnormalize"] = dict(VECNORM)
    return _official_td3(policy, venv, **kwargs)


def run(attack: str, seed: int, rounds: int, q: float, dataset: str, out_dir: str) -> str:
    B21.R.TD3 = _steelman_td3  # applies to this process only
    path = B21.run(attack, "td3", seed, rounds, q, dataset, out_dir)
    d = json.load(open(path))
    d["variant"] = "td3_steelman_normalizado"
    d["td3_kwargs"] = {k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v))
                       for k, v in _used.items()}
    B21.R._dump(d, path)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", required=True, choices=["LMP", "EB"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--q", type=float, default=0.5)
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    a = p.parse_args()
    print(run(a.attack, a.seed, a.rounds, a.q, a.dataset, a.out_dir))


if __name__ == "__main__":
    main()
