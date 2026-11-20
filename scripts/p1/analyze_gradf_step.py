"""
Analysis of the noise-free step arm (results/p1_parity/gradf_layer2/step/ADDENDUM.md),
with the reading rule frozen there. Writes results/p1_parity/gradf_layer2/step/analysis/.

Usage: venv/bin/python scripts/p1/analyze_gradf_step.py
"""

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import inference as inf  # noqa: E402
from analyze_gradf_layer2 import KEY, LOGGED, RAW, load  # noqa: E402
from analyze_gradf_repeat import fmt  # noqa: E402
from analyze_parity import FA  # noqa: E402

STEP = "results/p1_parity/gradf_layer2/step/raw"
OUT = "results/p1_parity/gradf_layer2/step/analysis"
CONTRASTS = [("S010", "F000"), ("N010", "S010")]


def reading(first, second):
    if first != "superiority":
        return "NOT SUPPORTED BY THIS ARM"
    return {"equivalence": "EXPLAINED BY THE STEP", "inferiority": "STEP ALONE GOES FURTHER",
            "superiority": "PARTLY EXPLAINED"}.get(second, "INCONCLUSIVE")


def main():
    os.makedirs(OUT, exist_ok=True)
    acc_old, rounds_old = load(RAW)
    acc_new, rounds_new = load(STEP)
    acc = pd.concat([acc_old, acc_new], ignore_index=True)
    rounds = pd.concat([rounds_old, rounds_new], ignore_index=True)
    out = [f"JSONs: {len(glob.glob(f'{STEP}/*.json'))}/10; S010 accuracy rows {len(acc_new)} (expected 210)", ""]
    P = acc.pivot_table(index=KEY, columns="arm", values="accuracy")
    C = P[P.index.get_level_values("attack_type") != FA]
    rows, cat = [], []
    for a, b in CONTRASTS:
        r18 = inf.describe((C[a] - C[b]).groupby(level="seed").mean().values)
        r21 = inf.describe((P[a] - P[b]).groupby(level="seed").mean().values)
        cat.append(r18["category"])
        rows.append({"contrast": f"{a} - {b}", **r18})
        out.append(f"{a} - {b}: 18 cells {fmt(r18)}, TOST p {r18['p_tost']:.4f}, category {r18['category']}, "
                   f"per seed {(100 * (C[a] - C[b]).groupby(level='seed').mean()).round(2).tolist()}; 21 cells {fmt(r21)}")
    pd.DataFrame(rows).to_csv(f"{OUT}/contrasts.csv", index=False)
    out += ["", f"READING: {reading(*cat)}", "",
            f"Arm means, 18 cells: {(100 * C[['F000', 'S010', 'N010', 'F010']].mean()).round(2).to_dict()}"]
    for col in inf.CELL:
        t = pd.DataFrame({f"{a}-{b}": (P[a] - P[b]).groupby(level=col).mean() for a, b in CONTRASTS})
        t.to_csv(f"{OUT}/contrasts_by_{col}.csv")
        out += ["", f"Contrasts by {col} (p.p., 21 cells):", (100 * t).round(2).to_string()]
    cell = (100 * P[["F000", "S010", "N010"]].groupby(level=inf.CELL).mean()).round(2)
    cell["S-F"] = (cell.S010 - cell.F000).round(2)
    cell["N-S"] = (cell.N010 - cell.S010).round(2)
    cell.to_csv(f"{OUT}/per_cell.csv")
    out += ["", "Per cell (p.p.):", cell.to_string()]
    r = rounds[rounds.arm.isin(["F000", "S010", "N010"])]
    out += ["", "Logged means, all rounds:", r.groupby("arm")[LOGGED].mean().round(4).to_string()]
    t = r.groupby(["arm", "round"]).step_along_g0.mean().unstack("arm")
    out += ["", "Step along g0 by round:", t.round(4).to_string()]
    g = r.groupby(["arm", "round"]).global_accuracy.mean().unstack("arm")
    g.to_csv(f"{OUT}/accuracy_by_round.csv")
    out += ["", "Global accuracy by round (p.p., 21 cells):", (100 * g).round(2).to_string()]
    text = "\n".join(out) + "\n"
    with open(f"{OUT}/analysis.txt", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
