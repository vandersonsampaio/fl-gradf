"""
Analysis of the global learning rate tuning (results/p1_parity/lr_tuning/ADDENDUM.md), with the
choice rule frozen there. Writes analysis.txt and runs.csv under --out.

Usage: venv/bin/python scripts/p1/analyze_lr_tuning.py [--raw DIR] [--out DIR]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

DATASETS = {"mnist": 15, "cifar10": 110}
RULES = ("fedavg", "trimmed_mean", "fltrust")
DENS = (1, 3, 5, 10)
SEEDS = (0, 1, 2)
MARGIN = 1.0


def summary(train, test, r):
    err = 100 * (1 - np.array(train[:r]))
    acc = 100 * np.array(test[:r])
    final = err[r - 5:r].mean()
    within = np.nonzero(err <= final + MARGIN)[0]
    return {"train_error": final, "test_accuracy": acc[r - 5:r].mean(),
            "first_round_near_final": int(within[0]) + 1, "swing": np.abs(np.diff(err[r // 2 - 1:])).mean()}


def choose(table):
    """table: index den, columns rule, values training error. Returns (den, gaps, admissible)."""
    gaps = table - table.min()
    worst = gaps.max(axis=1)
    admissible = [d for d in DENS if worst[d] <= MARGIN]
    if admissible:
        return min(admissible), worst, True
    return min(DENS, key=lambda d: (worst[d], d)), worst, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/p1_parity/lr_tuning/raw")
    ap.add_argument("--out", default="results/p1_parity/lr_tuning/analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows, out = [], []
    for b, r in DATASETS.items():
        for u in RULES:
            for d in DENS:
                for s in SEEDS:
                    with open(f"{args.raw}/{b}_{u}_s{s}_f{d}.json") as f:
                        j = json.load(f)
                    rows.append({"dataset": b, "rule": u, "den": d, "seed": s, **summary(j["train_accuracy"], j["test_accuracy"], r), "s_per_round": j["seconds"] / r})
    t = pd.DataFrame(rows)
    t.to_csv(f"{args.out}/runs.csv", index=False)
    for b, r in DATASETS.items():
        x = t[t.dataset == b]
        table = x.pivot_table(index="den", columns="rule", values="train_error")[list(RULES)]
        den, worst, met = choose(table)
        out += [f"=== {b} (R = {r})", "Training error, trailing 5 rounds, seed means:", table.round(2).to_string(), "",
                "Largest gap to each rule's own best, per factor:", worst.round(2).to_string(), ""]
        out.append(f"CHOSEN: factor 1/{den} ({'largest factor within the margin in all rules' if met else 'NO FACTOR MET THE MARGIN; smallest largest gap'})")
        out.append("")
        for name, col in (("Test accuracy, trailing 5 rounds", "test_accuracy"), ("First round within the margin of the final error", "first_round_near_final"),
                          ("Swing of the training error, second half", "swing")):
            out += [f"{name} (descriptive):", x.pivot_table(index="den", columns="rule", values=col)[list(RULES)].round(2).to_string(), ""]
        out += ["Training error per seed:", x.pivot_table(index=["rule", "seed"], columns="den", values="train_error").round(2).to_string(), "",
                f"Seconds per round: {x.s_per_round.mean():.1f}", ""]
    text = "\n".join(out)
    with open(f"{args.out}/analysis.txt", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
