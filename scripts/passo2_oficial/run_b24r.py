"""
B2.4r (results/b24r_limiar_oficial/PLANO.md): reduced sweep of the threshold a₅
in the official AdaAggRL, under EB. Fixed action [0.475]*4 + [a₅].

Reuses `run_oficial.run` (the Step 2 runner, where the center a₅ = 0.475 comes from)
without changing it: it only replaces `run_oficial.A_FIXED` in this process. One directory per a₅.

Usage (official venv):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_b24r.py --a5 0.1 --seed 100
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_oficial as R  # noqa: E402

A5_VALUES = [0.0, 0.1, 0.25, 0.95]
OUT = os.path.join(R.REPO, "results", "b24r_limiar_oficial", "raw")


def run(a5: float, seed: int, rounds: int = 500, out_root: str = OUT) -> str:
    R.A_FIXED = [0.475] * 4 + [float(a5)]  # applies to this process only
    return R.run("EB", "fixed", seed, rounds, 0.5, "MNIST", os.path.join(out_root, f"a5_{a5:g}"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a5", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--out_root", default=OUT)
    a = p.parse_args()
    print(run(a.a5, a.seed, a.rounds, a.out_root))


if __name__ == "__main__":
    main()
