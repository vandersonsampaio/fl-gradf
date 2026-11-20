"""
Analysis of the GRADF ablation (results/p1_parity/gradf_ablation/PLAN.md), with
the reading rule frozen there.

  Sec. 1  completeness, and Sec. 5 of the plan: F000 against fixed FLTrust
  Sec. 2  Q1: GRADF - F111 per execution
  Sec. 3  Q2: GRADF and Random against the cross-fitted hardened best fixed rule
  Sec. 4  Q3: main effects of Layers 1, 2 and 4 in the 2^3 on FLTrust
  Sec. 5  Q4 and Q5: Random and the unhardened systems against R000
  Sec. 6  mechanism (descriptive): rejections per arm and attack, and GRADF
          against F111 where GRADF nearly always votes FLTrust

Writes results/p1_parity/gradf_ablation/analysis/.

Usage: venv/bin/python scripts/p1/analyze_gradf_ablation.py
"""

import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import inference as inf  # noqa: E402
from analyze_gradf_repeat import BYZANTINE, NEARLY_ALWAYS, fmt, load_new  # noqa: E402
from analyze_parity import FA, add_published_adaaggrl, load_parity_fixed, load_parity_systems  # noqa: E402

RAW = "results/p1_parity/gradf_ablation/raw"
OUT = "results/p1_parity/gradf_ablation/analysis"
SEEDS = list(range(42, 52))
LAYERS = ("L1", "L2", "L4")
FACTORIAL = ["F" + "".join(b) for b in itertools.product("01", repeat=3)]
HARDENED = {"F111": "fltrust", "M111": "median", "T111": "trimmed_mean", "K111": "krum"}
KEY = inf.CELL + ["seed"]


def load():
    acc, rounds, seconds = [], [], []
    for s in SEEDS:
        with open(f"{RAW}/seed{s}.json") as f:
            d = json.load(f)
        acc += [{**a, "seed": s} for a in d["accuracy"]]
        rounds += [{**r, "seed": s} for r in d["rounds"]]
        seconds.append(d["seconds"])
    return pd.DataFrame(acc), pd.DataFrame(rounds), seconds


def confirmatory(df):
    return df[df.attack_type != FA]


def series(df):
    return df.set_index(KEY).accuracy


def seed_mean(d):
    return d.dropna().groupby(level="seed").mean().values


def reading(cats):
    for c, text in (("superiority", "SUPERIORITY in all 3"), ("equivalence", "EQUIVALENCE in all 3"),
                    ("inferiority", "INFERIORITY in all 3")):
        if all(x == c for x in cats):
            return text
    return "INCONCLUSIVE (categories differ or no evidence)"


def learner_check(acc, rounds, seconds, fx, out):
    out += ["## Sec. 1 Completeness and learner check", ""]
    out.append(f"JSONs: {len(glob.glob(f'{RAW}/*.json'))}/10; accuracy rows {len(acc)} (expected 2520); "
               f"round rows {len(rounds)} (expected 37800); hours per job {np.round(np.array(seconds) / 3600, 2).tolist()}")
    last = rounds[rounds["round"] == rounds["round"].max()].set_index(KEY + ["arm"]).global_accuracy
    d = (acc.set_index(KEY + ["arm"]).accuracy - last).abs()
    out.append(f"Final accuracy equals last-round global accuracy: max |d| {d.max():.6f}")
    voted = rounds[rounds.arm.isin(list(HARDENED) + FACTORIAL)].groupby("arm").strategy.agg(lambda x: sorted(set(x)))
    out.append(f"Voted rules per fixed arm: {voted.to_dict()}")
    out.append(f"Layer-3 errors: {int(rounds.layer3_error.notna().sum())}")
    ref = series(fx[(fx.root == 100) & (fx.strategy == "fltrust")])
    new = series(acc[acc.arm == "F000"])
    d = (new - ref).dropna()
    out += ["", f"F000 - fixed FLTrust, {len(d)} pairs: {(d != 0).sum()} differ, mean {100 * d.mean():+.2f} p.p.,"
            f" max |d| {100 * d.abs().max():.2f}"]
    r21 = inf.describe(seed_mean(d))
    se = d.std(ddof=1) / np.sqrt(len(d))
    out.append(f"  per seed (21 cells): {fmt(r21)}, category {r21['category']}")
    out.append(f"  over the 210 pairs, as if independent: {100 * d.mean():+.2f} "
               f"({100 * (d.mean() - 1.96 * se):+.2f}; {100 * (d.mean() + 1.96 * se):+.2f})")
    inside = -inf.MARGIN < r21["ci95_lo"] and r21["ci95_hi"] < inf.MARGIN
    out.append(f"Learner check (CI95 inside +-1.0 p.p.): {'PASS' if inside else 'FAIL'}")
    per_cell = d.groupby(level=inf.CELL).mean()
    out += ["Largest |F000 - fixed FLTrust| per cell (p.p.):",
            (100 * per_cell.reindex(per_cell.abs().sort_values(ascending=False).index)).head(6).round(2).to_string()]
    return inside


def q1(runs, f111, out):
    out += ["", "## Sec. 2 Q1: GRADF - F111 (p.p.)", ""]
    rows, cats = [], []
    for name, sy in runs.items():
        g = series(sy[sy.system == "GRADF"])
        r18 = inf.describe(seed_mean((g - f111)[lambda x: x.index.get_level_values("attack_type") != FA]))
        r21 = inf.describe(seed_mean(g - f111))
        cats.append(r18["category"])
        rows += [{"execution": name, "cells": 18, **r18}, {"execution": name, "cells": 21, **r21}]
        out.append(f"{name}: 18 cells {fmt(r18)} TOST p {r18['p_tost']:.3f}, category {r18['category']}; 21 cells {fmt(r21)}")
    pd.DataFrame(rows).to_csv(f"{OUT}/q1_gradf_vs_f111.csv", index=False)
    verdict = reading(cats)
    out += ["", f"Q1 READING: {verdict}"]
    return verdict


def q2(runs, acc, fx, out):
    out += ["", "## Sec. 3 Q2: against the hardened best fixed rule B_-s^H (p.p.)", ""]
    h = acc[acc.arm.isin(HARDENED)].assign(strategy=lambda d: d.arm.map(HARDENED))
    rows, cats = [], []
    B18, _ = inf.crossfit_references(confirmatory(h))
    B21, _ = inf.crossfit_references(h)
    mean = confirmatory(h).groupby("strategy").accuracy.mean()
    out.append(f"Hardened fixed rules, mean over 18 cells: {(100 * mean).round(2).to_dict()}")
    for name, sy in runs.items():
        for s in ("GRADF", "Random"):
            v = series(sy[sy.system == s])
            r18 = inf.describe(seed_mean(v - B18))
            r21 = inf.describe(seed_mean(v - B21))
            rows.append({"execution": name, "system": s, **r18})
            if s == "GRADF":
                cats.append(r18["category"])
            out.append(f"{name}, {s} - B_-s^H: 18 cells {fmt(r18)}, category {r18['category']}; 21 cells {fmt(r21)}")
        _, cap = inf.e3_capture(confirmatory(h), confirmatory(sy), ["GRADF", "Random"])
        out.append(f"  {name}: cell-wise capture {cap}")
    pd.DataFrame(rows).to_csv(f"{OUT}/q2_vs_hardened_best.csv", index=False)
    verdict = reading(cats)
    out += ["", f"Q2 READING (GRADF): {verdict}"]
    return B18


def q3(acc, out):
    out += ["", "## Sec. 4 Q3: Layers 1, 2 and 4 on fixed FLTrust (2^3, p.p.)", ""]
    f = acc[acc.arm.isin(FACTORIAL)].copy()
    for i, l in enumerate(LAYERS):
        f[l] = f.arm.str[i + 1].astype(int)

    def effect(df, l):
        return df[df[l] == 1].groupby("seed").accuracy.mean() - df[df[l] == 0].groupby("seed").accuracy.mean()

    conf = confirmatory(f)
    res = pd.DataFrame([{"layer": l, **inf.describe(effect(conf, l).values)} for l in LAYERS])
    res["p_t_holm"] = inf.holm(res.p_t.values)
    res["p_wilcoxon_holm"] = inf.holm(res.p_wilcoxon.values)
    res.to_csv(f"{OUT}/q3_main_effects.csv", index=False)
    for _, r in res.iterrows():
        out.append(f"{r.layer}: {fmt(r)}, Holm p_t {r.p_t_holm:.4f}, Holm Wilcoxon {r.p_wilcoxon_holm:.4f}")
    arm = conf.groupby("arm").accuracy.mean().reindex(FACTORIAL)
    out += ["", "Arm means, 18 cells:", (100 * arm).round(2).to_string(), ""]
    for a, b in itertools.combinations(LAYERS, 2):
        out += [f"Two-way means {a} x {b}:", (100 * conf.pivot_table(index=a, columns=b, values="accuracy")).round(2).to_string(), ""]
    for col in ("attack_type", "alpha"):
        t = pd.DataFrame({l: {k: effect(g, l).mean() for k, g in f.groupby(col)} for l in LAYERS})
        t.to_csv(f"{OUT}/q3_effects_by_{col}.csv")
        out += [f"Main effects by {col} (21 cells; {FA} outside the 18):", (100 * t).round(2).to_string(), ""]


def q4_q5(acc, B_h, fx, first, out):
    out += ["## Sec. 5 Q4 and Q5, 18 cells (p.p.)", ""]
    r000 = series(confirmatory(acc[acc.arm == "R000"]))
    rnd = series(confirmatory(first[first.system == "Random"]))
    B, _ = inf.crossfit_references(confirmatory(fx[fx.root == 100]))
    rows = []
    for name, d in (("Random - R000", rnd - r000), ("Random - B_-s^H", rnd - B_h), ("R000 - B_-s", r000 - B),
                    ("B_-s^H - B_-s", B_h - B)):
        r = inf.describe(seed_mean(d))
        rows.append({"comparison": name, **r})
        out.append(f"{name}: {fmt(r)}, category {r['category']}")
    sy = add_published_adaaggrl(first)
    for s in ("FedStrategist", "AdaAggRL"):
        r = inf.describe(seed_mean(series(confirmatory(sy[sy.system == s])) - r000))
        rows.append({"comparison": f"{s} - R000", **r})
        note = " (published, unseeded runs)" if s == "AdaAggRL" else ""
        out.append(f"{s} - R000{note}: {fmt(r)}, category {r['category']}")
    pd.DataFrame(rows).to_csv(f"{OUT}/q4_q5.csv", index=False)


def mechanism(acc, rounds, runs, new_rounds, out):
    out += ["", "## Sec. 6 Mechanism (descriptive, 21 cells)", ""]
    r = rounds.copy()
    r["rejected_l4"] = ~r.accepted
    r["l1_byz"] = r.layer1_rejected.apply(lambda x: sum(i in BYZANTINE for i in x))
    r["l1_honest"] = r.layer1_rejected.apply(lambda x: sum(i not in BYZANTINE for i in x))
    t = r.groupby("arm").agg(l4_rejection=("rejected_l4", "mean"), l1_byz=("l1_byz", "mean"), l1_honest=("l1_honest", "mean"))
    t.to_csv(f"{OUT}/mechanism_per_arm.csv")
    out += ["Per arm:", t.round(3).to_string(), ""]
    p = r.pivot_table(index="attack_type", columns="arm", values="rejected_l4")
    p.to_csv(f"{OUT}/l4_rejection_by_attack.csv")
    out += ["Layer-4 rejection by attack:", p.round(3).to_string(), ""]
    out += ["Layer-1 Byzantine rejections per round by attack:",
            r.pivot_table(index="attack_type", columns="arm", values="l1_byz").round(3).to_string(), ""]

    g = new_rounds[new_rounds.system == "GRADF"]
    share = g.assign(v=g.strategy == "fltrust").groupby(inf.CELL).v.mean()
    arm = acc.pivot_table(index=inf.CELL, columns="arm", values="accuracy")
    gradf = pd.concat([sy[sy.system == "GRADF"] for sy in runs.values()]).groupby(inf.CELL).accuracy.mean()
    rnd = runs["first run"]
    rnd = rnd[rnd.system == "Random"].groupby(inf.CELL).accuracy.mean()
    c = pd.DataFrame({"gradf_fltrust_share": share, "gradf_3exec": gradf, "F111": arm.F111, "F000": arm.F000,
                      "random": rnd, "R000": arm.R000})
    c["gradf_minus_f111"] = c.gradf_3exec - c.F111
    c.to_csv(f"{OUT}/per_cell.csv")
    out += ["Per cell (p.p.; share in %):", (100 * c).round(1).to_string(), ""]
    near = c[c.gradf_fltrust_share >= NEARLY_ALWAYS]
    out.append(f"Cells with >= {NEARLY_ALWAYS:.0%} FLTrust votes: {len(near)}; GRADF - F111 {100 * near.gradf_minus_f111.mean():+.2f}, "
               f"F111 - F000 {100 * (near.F111 - near.F000).mean():+.2f}, GRADF - F000 {100 * (near.gradf_3exec - near.F000).mean():+.2f}, "
               f"GRADF above F111 in {(near.gradf_minus_f111 > 0).sum()}")


def main():
    os.makedirs(OUT, exist_ok=True)
    acc, rounds, seconds = load()
    fx, first = load_parity_fixed(), load_parity_systems()
    new_acc, new_rounds = load_new()
    runs = {"first run": first}
    for e in (1, 2):
        runs[f"exec {e}"] = new_acc[new_acc.execution == e]
    f111 = series(acc[acc.arm == "F111"])
    out = []
    learner_check(acc, rounds, seconds, fx, out)
    q1(runs, f111, out)
    B_h = q2(runs, acc, fx, out)
    q3(acc, out)
    q4_q5(acc, B_h, fx, first, out)
    mechanism(acc, rounds, runs, new_rounds, out)
    text = "\n".join(out) + "\n"
    with open(f"{OUT}/analysis.txt", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
