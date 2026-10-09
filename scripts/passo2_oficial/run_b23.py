"""
B2.3 (results/b23_steelman_oficial/PLANO.md): TD3 steelman in the official
AdaAggRL. Question: given much more favorable learning conditions,
does TD3 come to beat the fixed action?

The only change relative to the official TD3 (main.py):
  learning_rate   1e-5 -> 1e-3   (100x; actor and critic, SB3's Adam)
  learning_starts 100  -> 10     (10x shorter random-action warm-up)
Everything else is identical: MlpPolicy [256,128], buffer 1000, batch 64,
train_freq 3, noise N(0, 0.1), gamma 0.99, same environment and same seeds.

Exploratory, on already-used seeds (100-104), paired with the `fixed` and `td3` runs
of Step 2 (results/frente1_passo2_oficial/raw/). Reuses `run_b21.run` without
changing it (including the state log and the actor checkpoints), replacing only the
TD3 constructor in this process.

Usage (always with the isolated venv):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_b23.py --attack EB --seed 100
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_b21 as B21  # noqa: E402

STEELMAN = {"learning_rate": 1e-3, "learning_starts": 10}
DEFAULT_OUT = os.path.join(B21.R.REPO, "results", "b23_steelman_oficial", "raw")

_official_td3 = B21.R.TD3
_used_kwargs = {}


def _steelman_td3(*args, **kwargs):
    """Replaces only lr and learning_starts; records the effective kwargs."""
    kwargs.update(STEELMAN)
    _used_kwargs.clear()
    _used_kwargs.update({k: v for k, v in kwargs.items() if k not in ("action_noise",)})
    return _official_td3(*args, **kwargs)


def run(attack: str, seed: int, rounds: int, q: float, dataset: str, out_dir: str) -> str:
    B21.R.TD3 = _steelman_td3  # applies to this process only
    path = B21.run(attack, "td3", seed, rounds, q, dataset, out_dir)
    d = json.load(open(path))
    d["variant"] = "td3_steelman"
    d["td3_kwargs"] = {k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v))
                       for k, v in _used_kwargs.items()}
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
