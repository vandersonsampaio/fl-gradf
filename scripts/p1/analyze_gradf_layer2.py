"""
Analysis of the Layer 2 split (results/p1_parity/gradf_layer2/PLAN.md), with the
reading rule frozen there.

  Sec. 1  completeness, and F000 and F010 against the ablation
  Sec. 2  Q1: noise, clipping or both
  Sec. 3  Q2 (descriptive): the logged norms per arm, by alpha and by attack

Writes results/p1_parity/gradf_layer2/analysis/.

Usage: venv/bin/python scripts/p1/analyze_gradf_layer2.py
"""

import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import inference as inf  # noqa: E402
from analyze_gradf_repeat import fmt  # noqa: E402
from analyze_parity import FA  # noqa: E402

RAW = "results/p1_parity/gradf_layer2/raw"
ABLATION = "results/p1_parity/gradf_ablation/raw"
OUT = "results/p1_parity/gradf_layer2/analysis"
SEEDS = list(range(42, 52))
KEY = inf.CELL + ["seed"]
CONTRASTS = [("N010", "F000"), ("C010", "F000"), ("F010", "N010"), ("F010", "C010")]
LOGGED = ["g0_norm", "client_norm_honest", "delta_norm", "step_along_g0", "trust_honest", "trust_byzantine", "clipped_fraction"]


def load(folder):
    acc, rounds = [], []
    for s in SEEDS:
        with open(f"{folder}/seed{s}.json") as f:
            d = json.load(f)
        acc += [{**a, "seed": s} for a in d["accuracy"]]
        rounds += [{**r, "seed": s} for r in d["rounds"]]
    return pd.DataFrame(acc), pd.DataFrame(rounds)


def per_seed(P, a, b):
    return (P[a] - P[b]).groupby(level="seed").mean().values


def regression(acc, out):
    out += ["## Sec. 1 Completeness and regression", ""]
    out.append(f"JSONs: {len(glob.glob(f'{RAW}/*.json'))}/10; accuracy rows {len(acc)} (expected 840)")
    old, _ = load(ABLATION)
    ok = True
    for arm in ("F000", "F010"):
        a = acc[acc.arm == arm].set_index(KEY).accuracy
        b = old[old.arm == arm].set_index(KEY).accuracy
        d = (a - b).dropna().abs()
        ok = ok and len(d) == 210 and (d == 0).all()
        out.append(f"{arm} vs ablation: {len(d)} pairs, {(d > 0).sum()} differ, max |d| {d.max():.2e}")
    return ok


def q1(acc, out):
    out += ["", "## Sec. 2 Q1: contrasts (p.p.)", ""]
    P = acc.pivot_table(index=KEY, columns="arm", values="accuracy")
    C = P[P.index.get_level_values("attack_type") != FA]
    rows, cat = [], {}
    for a, b in CONTRASTS:
        r18, r21 = inf.describe(per_seed(C, a, b)), inf.describe(per_seed(P, a, b))
        cat[(a, b)] = r18["category"]
        rows.append({"contrast": f"{a} - {b}", **r18})
        out.append(f"{a} - {b}: 18 cells {fmt(r18)}, TOST p {r18['p_tost']:.4f}, category {r18['category']}; 21 cells {fmt(r21)}")
    pd.DataFrame(rows).to_csv(f"{OUT}/q1_contrasts.csv", index=False)
    noise = cat[("N010", "F000")] == "superiority"
    clip = cat[("C010", "F000")] == "superiority"
    if noise and clip:
        verdict = "BOTH"
    elif noise and cat[("F010", "N010")] == "equivalence":
        verdict = "NOISE"
    elif clip and cat[("F010", "C010")] == "equivalence":
        verdict = "CLIPPING"
    else:
        verdict = "INCONCLUSIVE"
    out += ["", f"Q1 READING: {verdict}", ""]
    out.append(f"Arm means, 18 cells: {(100 * C.mean()).round(2).to_dict()}")
    out.append(f"Pairs equal to the bit: C010 = F000 {(P.C010 == P.F000).sum()} of 210; N010 = F010 {(P.N010 == P.F010).sum()} of 210")
    for col in inf.CELL:
        t = pd.DataFrame({f"{a}-{b}": (P[a] - P[b]).groupby(level=col).mean() for a, b in CONTRASTS})
        t.to_csv(f"{OUT}/q1_by_{col}.csv")
        out += ["", f"Contrasts by {col} (p.p., 21 cells):", (100 * t).round(2).to_string()]


def q2(rounds, out):
    out += ["", "## Sec. 3 Q2: logged quantities (descriptive)", ""]
    r = rounds
    last = r[r["round"] >= r["round"].max() - 4]
    for name, d in (("all rounds", r), ("last 5 rounds", last)):
        t = d.groupby("arm")[LOGGED].mean()
        t.to_csv(f"{OUT}/q2_{name.replace(' ', '_')}.csv")
        out += [f"Per arm, {name}:", t.round(4).to_string(), ""]
    for col in inf.CELL:
        t = r.pivot_table(index=col, columns="arm", values="step_along_g0")
        t.to_csv(f"{OUT}/q2_step_by_{col}.csv")
        out += [f"Step along g0 by {col}:", t.round(4).to_string(), ""]
    t = r.groupby(["arm", "round"]).step_along_g0.mean().unstack("arm")
    t.to_csv(f"{OUT}/q2_step_by_round.csv")
    out += ["Step along g0 by round:", t.round(4).to_string(), ""]
    t = r.pivot_table(index="attack_type", columns="arm", values="clipped_fraction")
    out += ["Fraction of coordinates above the clipping bound, by attack:", t.iloc[:, :1].round(5).to_string()]


def main():
    os.makedirs(OUT, exist_ok=True)
    acc, rounds = load(RAW)
    out = []
    if not regression(acc, out):
        out.append("REGRESSION MISMATCH: no attribution until understood")
    q1(acc, out)
    q2(rounds, out)
    text = "\n".join(out) + "\n"
    with open(f"{OUT}/analysis.txt", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
