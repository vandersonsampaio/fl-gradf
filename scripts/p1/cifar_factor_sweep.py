"""
One job of the FLTrust factor sweep on CIFAR-10 (results/p1_parity/cifar_factor_sweep/ADDENDUM.md):
one run of block C with the aggregate multiplied by 1/den, through run() of
cifar_step_pilot.py. Seeds outside the confirmatory block 42-51.

Usage: venv/bin/python scripts/p1/cifar_factor_sweep.py --attack sign_flipping --root_size 100 --seed 0 --den 5
       venv/bin/python scripts/p1/cifar_factor_sweep.py --check OUT_TXT
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))

from cifar_step_pilot import run  # noqa: E402

OUT = "results/p1_parity/cifar_factor_sweep/raw"
DENS = (5, 10)


def check(out):
    lines, ok = [], True
    one = run("trim_attack", 100, 0, rounds=1, factor=1.0)["rounds"][0]["step_along_g0"]
    for den in DENS:
        step = run("trim_attack", 100, 0, rounds=1, factor=1 / den)["rounds"][0]["step_along_g0"]
        passed = abs(step / one - 1 / den) < 1e-6
        ok = ok and passed
        lines.append(f"{'PASS' if passed else 'FAIL'} first round's step is 1/{den} of factor 1's: ratio {step / one:.6f}")
    lines.append("CHECKS_PASS" if ok else "CHECKS_FAIL")
    text = "\n".join(lines) + "\n"
    with open(out, "w") as f:
        f.write(text)
    print(text)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack")
    ap.add_argument("--root_size", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--den", type=int)
    ap.add_argument("--check")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check(args.check) else 1)
    path = f"{OUT}/{args.attack}_r{args.root_size}_s{args.seed}_f{args.den}.json"
    if os.path.exists(path):
        print("exists", path)
        return
    res = run(args.attack, args.root_size, args.seed, factor=1 / args.den)
    os.makedirs(OUT, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
