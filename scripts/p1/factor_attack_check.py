"""
One job of the verification under attack on CIFAR-10 (results/p1_parity/factor_attack_check/ADDENDUM.md):
the trimmed mean or FedAvg under block C's attacks, with the aggregate multiplied by a constant factor.
The run itself is cifar_factor_rules.run.

Usage: venv/bin/python scripts/p1/factor_attack_check.py --rule trimmed_mean --attack sign_flipping --seed 0 --den 3
       venv/bin/python scripts/p1/factor_attack_check.py --check OUT_TXT
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cifar_factor_rules import ALPHA, E0B, run  # noqa: E402

OUT = "results/p1_parity/factor_attack_check/raw"


def check(path):
    lines, ok = [], True

    def report(name, passed, detail=""):
        nonlocal ok
        ok = ok and bool(passed)
        lines.append(f"{'PASS' if passed else 'FAIL'} {name} {detail}")

    for attack in ("sign_flipping", "trim_attack"):
        with open(f"{E0B}/a{ALPHA}_{attack}_fedavg_e2_s0_r100.json") as f:
            ref = json.load(f)["accuracy"][:2]
        a = run("fedavg", attack, 0, 1, rounds=2)
        report(f"check1 FedAvg at factor 1 reproduces E0b, {attack}", a["accuracy"] == ref, f"{a['accuracy']} vs {ref}")
        one = run("trimmed_mean", attack, 0, 1, rounds=1)["delta_norm"][0]
        third = run("trimmed_mean", attack, 0, 3, rounds=1)["delta_norm"][0]
        report(f"check2 the trimmed mean's first aggregate at factor 1/3 has 1/3 of the norm, {attack}", abs(third / one - 1 / 3) < 1e-6, f"ratio {third / one:.6f}")
    try:
        run("fedavg", "trim_attack", 42, 1, rounds=1)
        report("check3 confirmatory seeds are refused", False)
    except SystemExit:
        report("check3 confirmatory seeds are refused", True)
    lines.append("CHECKS_PASS" if ok else "CHECKS_FAIL")
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    print(text)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule")
    ap.add_argument("--attack")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--den", type=int)
    ap.add_argument("--check")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check(args.check) else 1)
    path = f"{OUT}/{args.rule}_{args.attack}_s{args.seed}_f{args.den}.json"
    if os.path.exists(path):
        print("exists", path)
        return
    res = run(args.rule, args.attack, args.seed, args.den)
    os.makedirs(OUT, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
