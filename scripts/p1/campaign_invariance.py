"""
First output of the campaign (PREREGISTRATION.md, Sec. 8): outside fltrust_aligned the trimmed
mean must not depend on the root, seed 42, 18 cells. Compares curves and final parameters for
equality and prints no accuracy. On a mismatch it touches the dispatcher's stop file.

Usage: campaign_invariance.py --log results/p1_campaign/dispatch.log
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from campaign_job import ALL_CELLS, OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    cells = [c for c in ALL_CELLS if c[1] != "fltrust_aligned"]
    pairs = [(f"{OUT}/e1_main/raw/trimmed_mean_r100_a{a}_{t}_s42.json",
              f"{OUT}/e1_invariant/raw/trimmed_mean_r2500_a{a}_{t}_s42.json") for a, t in cells]
    while not all(os.path.exists(p) for pair in pairs for p in pair):
        time.sleep(60)
    mismatches = []
    for small, large in pairs:
        with open(small) as f:
            s = json.load(f)
        with open(large) as f:
            b = json.load(f)
        if s["accuracy"] != b["accuracy"] or s["learner"]["params_sha256"] != b["learner"]["params_sha256"]:
            mismatches.append(os.path.basename(large))
    with open(f"{OUT}/invariance.txt", "w") as f:
        f.write(f"{time.strftime('%F %T')} {len(pairs)} cells, {len(mismatches)} mismatches\n")
        f.writelines(m + "\n" for m in mismatches)
        f.write("INVARIANCE_FAIL\n" if mismatches else "INVARIANCE_PASS\n")
    if mismatches:
        open(args.log + ".stop", "w").close()
        print("INVARIANCE_FAIL", len(mismatches))
    else:
        print("INVARIANCE_PASS", len(pairs))


if __name__ == "__main__":
    main()
