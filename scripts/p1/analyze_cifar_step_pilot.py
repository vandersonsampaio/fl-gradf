"""
Analysis of the CIFAR-10 step pilot (results/p1_parity/cifar_step_pilot/ADDENDUM.md),
with the reading rule frozen there. Writes results/p1_parity/cifar_step_pilot/analysis/.

Usage: venv/bin/python scripts/p1/analyze_cifar_step_pilot.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd

RAW = "results/p1_parity/cifar_step_pilot/raw"
BLOCK_C = "results/p1_parity/cifar_blockC/raw"
E0B = "results/p1_cifar10/e0b/raw"
OUT = "results/p1_parity/cifar_step_pilot/analysis"
R = 110
ATTACKS = ("sign_flipping", "trim_attack")
ROOTS = (100, 2500)
SEEDS = (0, 1, 2)


def summary(acc):
    c = 100 * np.array(acc[:R])
    return {"trailing": c[R - 5:R].mean(), "final": c[R - 1], "swing": np.abs(np.diff(c[50:])).mean()}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, curves, norms = [], [], []
    for a in ATTACKS:
        for r in ROOTS:
            for s in SEEDS:
                with open(f"{RAW}/{a}_r{r}_s{s}.json") as f:
                    d = json.load(f)
                with open(f"{BLOCK_C}/a0.05_{a}_fltrust_e2_s{s}_r{r}_stmedian.json") as f:
                    b = json.load(f)
                rows.append({"attack": a, "root": r, "seed": s, "arm": "factor_1/3", **summary(d["accuracy"]), "s_per_round": d["seconds"] / R})
                rows.append({"attack": a, "root": r, "seed": s, "arm": "factor_1", **summary(b["accuracy"])})
                curves += [{"attack": a, "root": r, "seed": s, "arm": "factor_1/3", "round": i + 1, "accuracy": 100 * x} for i, x in enumerate(d["accuracy"])]
                curves += [{"attack": a, "root": r, "seed": s, "arm": "factor_1", "round": i + 1, "accuracy": 100 * x} for i, x in enumerate(b["accuracy"][:R])]
                norms += [{"attack": a, "root": r, "seed": s, "round": i + 1, **x} for i, x in enumerate(d["rounds"])]
        for s in SEEDS:
            with open(f"{E0B}/a0.05_{a}_fedavg_e2_s{s}_r100.json") as f:
                rows.append({"attack": a, "root": 0, "seed": s, "arm": "fedavg", **summary(json.load(f)["accuracy"])})
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/runs.csv", index=False)
    pd.DataFrame(curves).to_csv(f"{OUT}/curves.csv", index=False)
    out = [f"JSONs: {len(glob.glob(f'{RAW}/*.json'))}/12", ""]

    f = t[t.arm != "fedavg"]
    cell = f.groupby(["attack", "root", "arm"])[["trailing", "final", "swing"]].mean().unstack("arm")
    out += ["Seed means per cell:", cell.round(2).to_string(), ""]
    ratio = cell["swing"]["factor_1/3"] / cell["swing"]["factor_1"]
    out += ["Swing ratio, factor 1/3 over factor 1:", ratio.round(3).to_string(), ""]
    n = int((ratio <= 0.5).sum())
    reading = "STABILIZES" if n == 4 else "DOES NOT STABILIZE" if n == 0 else "PARTLY"
    out += [f"READING: {reading} ({n} of 4 cells at most half)", ""]

    p = f.pivot_table(index=["attack", "root", "seed"], columns="arm", values="trailing")
    p["diff"] = p["factor_1/3"] - p["factor_1"]
    out += ["Trailing mean, factor 1/3 minus factor 1, per seed:", p.round(2).to_string(), ""]
    out.append(f"Mean of the differences, 12 runs: {p['diff'].mean():+.2f} p.p.")
    for arm in ("factor_1", "factor_1/3"):
        v = f[f.arm == arm].pivot_table(index=["attack", "seed"], columns="root", values="trailing")
        d = v[2500] - v[100]
        out.append(f"Root contrast (2500 - 100), {arm}: per attack {d.groupby(level='attack').mean().round(2).to_dict()}; per seed {d.round(2).tolist()}")
    fa = t[t.arm == "fedavg"].groupby("attack")[["trailing", "final", "swing"]].mean()
    out += ["", "FedAvg, same attacks and seeds:", fa.round(2).to_string(), ""]
    c = pd.DataFrame(curves).groupby(["arm", "attack", "root", "round"]).accuracy.mean().unstack(["arm"]).reset_index()
    pick = c[c["round"].isin([10, 30, 50, 70, 90, 100, 110])]
    out += ["Seed-mean curves:", pick.round(1).to_string(index=False), ""]
    n = pd.DataFrame(norms)
    n["period"] = pd.cut(n["round"], [0, 10, 50, 110], labels=["1-10", "11-50", "51-110"])
    out += ["Logged norms, factor 1/3:", n.groupby(["root", "period"], observed=True)[["g0_norm", "delta_norm", "step_along_g0"]].mean().round(4).to_string(), ""]
    out.append(f"Seconds per round: {t.s_per_round.mean():.1f}")
    text = "\n".join(out) + "\n"
    with open(f"{OUT}/analysis.txt", "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
