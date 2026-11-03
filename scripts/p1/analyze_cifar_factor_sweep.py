"""
Analysis of the FLTrust factor sweep on CIFAR-10 (results/p1_parity/cifar_factor_sweep/ADDENDUM.md),
with the reading rule frozen there. Writes results/p1_parity/cifar_factor_sweep/analysis/.

Usage: venv/bin/python scripts/p1/analyze_cifar_factor_sweep.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd

RAW = "results/p1_parity/cifar_factor_sweep/raw"
PILOT = "results/p1_parity/cifar_step_pilot/raw"
BLOCK_C = "results/p1_parity/cifar_blockC/raw"
E0B = "results/p1_cifar10/e0b/raw"
OUT = "results/p1_parity/cifar_factor_sweep/analysis"
R = 110
ATTACKS = ("sign_flipping", "trim_attack")
ROOTS = (100, 2500)
SEEDS = (0, 1, 2)
FACTORS = ("1", "1/3", "1/5", "1/10")


def summary(acc):
    c = 100 * np.array(acc[:R])
    return {"trailing": c[R - 5:R].mean(), "final": c[R - 1], "swing": np.abs(np.diff(c[50:])).mean()}


def path(factor, a, r, s):
    if factor == "1":
        return f"{BLOCK_C}/a0.05_{a}_fltrust_e2_s{s}_r{r}_stmedian.json"
    if factor == "1/3":
        return f"{PILOT}/{a}_r{r}_s{s}.json"
    return f"{RAW}/{a}_r{r}_s{s}_f{factor[2:]}.json"


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, curves = [], []
    for factor in FACTORS:
        for a in ATTACKS:
            for r in ROOTS:
                for s in SEEDS:
                    with open(path(factor, a, r, s)) as f:
                        d = json.load(f)
                    rows.append({"factor": factor, "attack": a, "root": r, "seed": s, **summary(d["accuracy"])})
                    curves += [{"factor": factor, "attack": a, "root": r, "seed": s, "round": i + 1, "accuracy": 100 * x} for i, x in enumerate(d["accuracy"][:R])]
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/runs.csv", index=False)
    out = [f"JSONs: {len(glob.glob(f'{RAW}/*.json'))}/24", ""]

    cell = t.groupby(["attack", "root", "factor"])[["trailing", "final", "swing"]].mean().unstack("factor")
    cell = cell.reindex(columns=[(m, f) for m in ("trailing", "final", "swing") for f in FACTORS])
    out += ["Seed means per cell:", cell.round(2).to_string(), ""]

    stable = []
    for factor in FACTORS[1:]:
        ratio = cell["swing"][factor] / cell["swing"]["1"]
        n = int((ratio <= 0.5).sum())
        reading = "STABILIZES" if n == 4 else "DOES NOT STABILIZE" if n == 0 else "PARTLY"
        out += [f"Factor {factor}: swing ratio over factor 1 {ratio.round(3).tolist()}; READING: {reading} ({n} of 4 cells at most half)"]
        if n == 4:
            stable.append(factor)
    out.append("")

    mean_acc = cell["trailing"].mean()
    out += ["Accuracy averaged over the 4 cells:", mean_acc.round(2).to_string(), ""]
    if stable:
        top = max(mean_acc[f] for f in stable)
        best = next(f for f in FACTORS if f in stable and top - mean_acc[f] <= 1.0)
        out.append(f"BEST FOR FLTRUST: factor {best} (stabilizing: {stable})")
    else:
        out.append("BEST FOR FLTRUST: none named, no factor stabilizes")
    out.append("")

    p = t.pivot_table(index=["attack", "root", "seed"], columns="factor", values="trailing")[list(FACTORS)]
    out += ["Trailing mean per seed:", p.round(2).to_string(), ""]
    for factor in FACTORS:
        v = t[t.factor == factor].pivot_table(index=["attack", "seed"], columns="root", values="trailing")
        d = v[2500] - v[100]
        out.append(f"Root contrast (2500 - 100), factor {factor}: per attack {d.groupby(level='attack').mean().round(2).to_dict()}; per seed {d.round(2).tolist()}")
    c = pd.DataFrame(curves)
    c.to_csv(f"{OUT}/curves.csv", index=False)
    w = c[c["round"].between(71, 110)].assign(window=lambda x: pd.cut(x["round"], [70, 90, 110], labels=["71-90", "91-110"]))
    w = w.groupby(["factor", "attack", "root", "window"], observed=True).accuracy.mean().unstack("window")
    w["change"] = w["91-110"] - w["71-90"]
    out += ["", "Accuracy by window, seed means (descriptive, not in the reading rule):", w.reindex(FACTORS, level="factor").round(2).to_string()]
    fa = []
    for a in ATTACKS:
        for s in SEEDS:
            with open(f"{E0B}/a0.05_{a}_fedavg_e2_s{s}_r100.json") as f:
                fa.append({"attack": a, "seed": s, **summary(json.load(f)["accuracy"])})
    out += ["", "FedAvg at factor 1, same attacks and seeds:", pd.DataFrame(fa).groupby("attack")[["trailing", "final", "swing"]].mean().round(2).to_string(), ""]
    text = "\n".join(out) + "\n"
    with open(f"{OUT}/analysis.txt", "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
