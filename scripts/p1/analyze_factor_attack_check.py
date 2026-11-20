"""
Analysis of the verification under attack on CIFAR-10 (results/p1_parity/factor_attack_check/ADDENDUM.md).
It reports and does not choose. Writes analysis.txt and runs.csv under --out.

Usage: venv/bin/python scripts/p1/analyze_factor_attack_check.py [--raw DIR] [--out DIR]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

BLOCK_C = "results/p1_parity/cifar_blockC/raw"
PILOT = "results/p1_parity/cifar_step_pilot/raw"
E0B = "results/p1_cifar10/e0b/raw"
R = 110
ATTACKS = ("sign_flipping", "trim_attack")
SEEDS = (0, 1, 2)
ROOTS = (100, 2500)
PAYS = 5.0


def summary(acc):
    c = 100 * np.array(acc[:R])
    return {"accuracy": c[R - 5:R].mean(), "stability": np.abs(np.diff(c[50:])).mean()}


def load(path):
    with open(path) as f:
        return summary(json.load(f)["accuracy"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/p1_parity/factor_attack_check/raw")
    ap.add_argument("--out", default="results/p1_parity/factor_attack_check/analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for a in ATTACKS:
        for s in SEEDS:
            for den in (1, 3):
                rows.append({"rule": "trimmed_mean", "factor": den, "attack": a, "seed": s, **load(f"{args.raw}/trimmed_mean_{a}_s{s}_f{den}.json")})
            rows.append({"rule": "fedavg", "factor": 1, "attack": a, "seed": s, **load(f"{E0B}/a0.05_{a}_fedavg_e2_s{s}_r100.json")})
            rows.append({"rule": "fedavg", "factor": 3, "attack": a, "seed": s, **load(f"{args.raw}/fedavg_{a}_s{s}_f3.json")})
            for r in ROOTS:
                rows.append({"rule": f"fltrust_r{r}", "factor": 1, "attack": a, "seed": s, **load(f"{BLOCK_C}/a0.05_{a}_fltrust_e2_s{s}_r{r}_stmedian.json")})
                rows.append({"rule": f"fltrust_r{r}", "factor": 3, "attack": a, "seed": s, **load(f"{PILOT}/{a}_r{r}_s{s}.json")})
    t = pd.DataFrame(rows)
    t.to_csv(f"{args.out}/runs.csv", index=False)
    m = t.groupby(["attack", "rule", "factor"])[["accuracy", "stability"]].mean().unstack("factor")
    out = ["Seed means (factor 3 means 1/3):", m.round(2).to_string(), ""]

    acc = t.groupby(["rule", "factor"]).accuracy.mean().unstack("factor")
    acc["change"] = acc[3] - acc[1]
    acc["pays"] = acc["change"] < -PAYS
    out += ["Accuracy over the 2 attacks, and whether a rule pays at factor 1/3 (more than 5.0 p.p. below factor 1):", acc.round(2).to_string(), ""]

    p = t.pivot_table(index=["attack", "seed", "factor"], columns="rule", values="accuracy")
    for r in ROOTS:
        p[f"V_r{r}"] = p[f"fltrust_r{r}"] - p["trimmed_mean"]
    p["D"] = p["V_r2500"] - p["V_r100"]
    out += ["FLTrust minus trimmed mean per root, and D = V(2500) - V(100), seed means:",
            p.groupby(["attack", "factor"])[["V_r100", "V_r2500", "D"]].mean().round(2).to_string(), "",
            "D per seed:", p["D"].unstack("factor").round(2).to_string(), ""]

    best = t.groupby(["factor", "attack", "rule"]).accuracy.mean().reset_index()
    best = best.loc[best.groupby(["factor", "attack"]).accuracy.idxmax()]
    out += ["Best rule per attack:", best.round(2).to_string(index=False), ""]
    text = "\n".join(out)
    with open(f"{args.out}/analysis.txt", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
