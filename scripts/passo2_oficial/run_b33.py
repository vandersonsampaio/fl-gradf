"""
B3.3 (results/b33_limiar_bloodmnist/PLANO.md): short sweep of the threshold a₅ in the
official AdaAggRL, BloodMNIST, EB. Fixed action [0.475]*4 + [a₅], a₅ ∈ {0.475; 0.75; 0.95}.

Reuses `run_b3.run` (the B3.1 runner, with the BloodMNIST shim) without changing it: it only replaces
`R.A_FIXED` in this process, as `run_b24r.py` does on MNIST. One directory per a₅
(the file name does not include a₅).

Usage (official venv):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_b33.py --a5 0.75 --seed 145
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_b3 as B3  # noqa: E402

R = B3.R
A5_VALUES = [0.475, 0.75, 0.95]
OUT = os.path.join(R.REPO, "results", "b33_limiar_bloodmnist", "raw")


def run(a5: float, seed: int, rounds: int = 500, out_root: str = OUT) -> str:
    R.A_FIXED = [0.475] * 4 + [float(a5)]  # applies to this process only
    return B3.run("BloodMNIST", "EB", "fixed", seed, rounds, os.path.join(out_root, f"a5_{a5:g}"))


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
