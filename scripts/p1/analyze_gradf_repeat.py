"""
Analysis of the GRADF repetition (results/p1_parity/gradf_repeat/PLAN.md), with
the reading rule frozen there.

  Sec. 1  completeness, and the regression of Random against ../mnist_rq3
  Sec. 2  repetition: GRADF - Random per execution, the robustness rule
  Sec. 3  GRADF - B_-s and Random - B_-s per execution, same fixed rules
  Sec. 4  mechanism (descriptive, no test): voted rules, Layer-4 rejections,
          Layer-1 rejections, GRADF against fixed FLTrust where GRADF nearly
          always votes FLTrust

Writes results/p1_parity/gradf_repeat/analysis/.

Usage: venv/bin/python scripts/p1/analyze_gradf_repeat.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import inference as inf  # noqa: E402
from analyze_parity import FA, load_parity_fixed, load_parity_systems  # noqa: E402

RAW = "results/p1_parity/gradf_repeat/raw"
OUT = "results/p1_parity/gradf_repeat/analysis"
EXECUTIONS = (1, 2)
SEEDS = list(range(42, 52))
BYZANTINE = {0, 1}  # byzantine_ids of exp10: 20% of 10 clients
NEARLY_ALWAYS = 0.9


def load_new():
    acc, rounds = [], []
    for e in EXECUTIONS:
        for s in SEEDS:
            with open(f"{RAW}/exec{e}_seed{s}.json") as f:
                d = json.load(f)
            acc += [{**a, "seed": s, "execution": e} for a in d["accuracy"]]
            rounds += [{**r, "seed": s, "execution": e, "round": i} for i, r in enumerate(d["rounds"])]
    return pd.DataFrame(acc), pd.DataFrame(rounds)


def fmt(r):
    return f"{100 * r['mean']:+.2f} ({100 * r['ci95_lo']:+.2f}; {100 * r['ci95_hi']:+.2f})"


def regression(acc, first, out):
    out += ["## Sec. 1 Completeness and regression", ""]
    out.append(f"JSONs: {len(glob.glob(f'{RAW}/*.json'))}/20; accuracy rows {len(acc)} (expected {2 * 10 * 21 * 2})")
    ref = first[first.system == "Random"].set_index(inf.CELL + ["seed"]).accuracy
    for e in EXECUTIONS:
        new = acc[(acc.execution == e) & (acc.system == "Random")].set_index(inf.CELL + ["seed"]).accuracy
        d = (new - ref).dropna().abs()
        out.append(f"Random exec {e} vs mnist_rq3: {len(d)} pairs, {(d > 0).sum()} differ, max |d| {d.max():.2e}")
        for k, v in d[d > 0].items():
            out.append(f"  differs: {k}, |d| = {v:.2e}")
    ref = first[first.system == "GRADF"].set_index(inf.CELL + ["seed"]).accuracy
    for e in EXECUTIONS:
        new = acc[(acc.execution == e) & (acc.system == "GRADF")].set_index(inf.CELL + ["seed"]).accuracy
        d = (new - ref).dropna().abs()
        out.append(f"GRADF exec {e} vs first run (non-deterministic, no expectation): {(d > 0).sum()} of {len(d)} pairs differ, "
                   f"mean |d| {100 * d.mean():.2f} p.p.")


def per_seed(sy, a, b):
    P = sy[sy.attack_type != FA].pivot_table(index=inf.CELL + ["seed"], columns="system", values="accuracy")
    return (P[a] - P[b]).groupby(level="seed").mean()


def repetition(runs, out):
    out += ["", "## Sec. 2 Repetition: GRADF - Random, 18 confirmatory cells (p.p.)", ""]
    rows = []
    for name, sy in runs.items():
        d = per_seed(sy, "GRADF", "Random")
        r = inf.describe(d.values)
        rows.append({"execution": name, **r})
        out.append(f"{name}: {fmt(r)} category {r['category']}, per seed {np.round(100 * d.values, 2).tolist()}")
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/repetition_gradf_vs_random.csv", index=False)
    mean, sd = t["mean"].mean(), t["mean"].std(ddof=1)
    c1, c2 = bool((t.ci95_lo > 0).all()), bool(sd < 0.5 * abs(mean))
    out += ["", f"Aggregate over executions: mean {100 * mean:+.2f}, SD {100 * sd:.2f}, half |mean| {100 * 0.5 * abs(mean):.2f}",
            f"Criterion 1 (CI95 above zero in all 3): {c1}", f"Criterion 2 (SD < half |mean|): {c2}",
            f"VERDICT: {'ROBUST' if c1 and c2 else 'NOT ROBUST'} to run-to-run variance"]


def versus_best(runs, fx, out):
    out += ["", "## Sec. 3 Against B_-s, same fixed rules, 18 confirmatory cells (p.p.)", ""]
    r100 = fx[(fx.root == 100) & (fx.attack_type != FA)]
    B, _ = inf.crossfit_references(r100)
    for name, sy in runs.items():
        P = sy[sy.attack_type != FA].pivot_table(index=inf.CELL + ["seed"], columns="system", values="accuracy")
        for s in ("GRADF", "Random"):
            r = inf.describe((P[s] - B).dropna().groupby(level="seed").mean().values)
            out.append(f"{name}, {s} - B_-s: {fmt(r)} category {r['category']}")


def mechanism(acc, rounds, fx, out):
    out += ["", "## Sec. 4 Mechanism (descriptive; both new executions, 21 cells)", ""]
    last = rounds[rounds["round"] % 15 == 14]
    r = rounds.copy()
    r["rejected_l4"] = ~r.accepted
    r["l1_byz"] = r.layer1_rejected.apply(lambda x: sum(i in BYZANTINE for i in x))
    r["l1_honest"] = r.layer1_rejected.apply(lambda x: sum(i not in BYZANTINE for i in x))
    r["l3_error"] = r.layer3_error.notna()
    shares = r.groupby(["system", "strategy"], dropna=False).size().rename("rounds").reset_index()
    shares["share"] = shares.rounds / shares.groupby("system").rounds.transform("sum")
    shares.to_csv(f"{OUT}/vote_shares.csv", index=False)
    out += ["Vote shares over all rounds:", shares.round(3).to_string(index=False), ""]

    keys = inf.CELL + ["system"]
    cell = r.groupby(keys).agg(l4_rejection=("rejected_l4", "mean"), l1_byz_per_round=("l1_byz", "mean"),
                               l1_honest_per_round=("l1_honest", "mean"), l3_errors=("l3_error", "sum")).reset_index()
    votes = r.groupby(keys).strategy.value_counts(normalize=True).unstack(fill_value=0).reset_index()
    cell = cell.merge(votes, on=keys).merge(
        acc.groupby(keys).accuracy.mean().reset_index(), on=keys)
    cell.to_csv(f"{OUT}/mechanism_per_cell.csv", index=False)
    out += ["Overall per system:",
            r.groupby("system").agg(l4_rejection=("rejected_l4", "mean"), l1_byz_per_round=("l1_byz", "mean"),
                                    l1_honest_per_round=("l1_honest", "mean"), l3_errors=("l3_error", "sum")).round(3).to_string(), ""]
    out += ["Layer-4 rejection by attack:",
            r.pivot_table(index="attack_type", columns="system", values="rejected_l4").round(3).to_string(), ""]

    fixed = fx[fx.root == 100].pivot_table(index=inf.CELL, columns="strategy", values="accuracy")
    g = cell[cell.system == "GRADF"].set_index(inf.CELL)
    ra = cell[cell.system == "Random"].set_index(inf.CELL)
    t = pd.DataFrame({"gradf_fltrust_share": g.fltrust, "gradf": g.accuracy, "fixed_fltrust": fixed.fltrust,
                      "random": ra.accuracy, "best_fixed": fixed[inf.D_R].max(axis=1),
                      "best_fixed_all7": fixed.max(axis=1)})
    t["gradf_minus_fltrust"] = t.gradf - t.fixed_fltrust
    t["gradf_minus_random"] = t.gradf - t.random
    t["gradf_minus_best"] = t.gradf - t.best_fixed
    t.to_csv(f"{OUT}/gradf_vs_fixed_fltrust.csv")
    out += ["GRADF against fixed FLTrust and Random per cell:", (100 * t).round(1).to_string(), ""]
    near = t[t.gradf_fltrust_share >= NEARLY_ALWAYS]
    out.append(f"Cells where GRADF votes FLTrust >= {NEARLY_ALWAYS:.0%}: {len(near)} of {len(t)}; "
               f"mean GRADF - fixed FLTrust {100 * near.gradf_minus_fltrust.mean():+.2f} p.p., "
               f"above fixed FLTrust in {(near.gradf_minus_fltrust > 0).sum()}")
    for col, label in (("best_fixed", "4 rules of D_R"), ("best_fixed_all7", "all 7 rules")):
        out.append(f"Above the cell-wise best fixed rule, {label}: GRADF {(t.gradf > t[col]).sum()} of {len(t)}, "
                   f"Random {(t.random > t[col]).sum()} of {len(t)}")
    out += ["", "Layer-1 rejections per round by attack (Byzantine; honest is zero where absent):",
            r.pivot_table(index="attack_type", columns="system", values=["l1_byz", "l1_honest"]).round(3).to_string(), ""]
    # Votes weighted by share over the fixed rules: what Random's mix would give without Layers 1, 2 and 4
    mix = (ra[fixed.columns] * fixed).sum(axis=1)
    out.append(f"Random - vote-weighted mean of fixed rules, 21 cells: {100 * (ra.accuracy - mix).mean():+.2f} p.p.; "
               f"18 cells: {100 * (ra.accuracy - mix)[mix.index.get_level_values('attack_type') != FA].mean():+.2f} p.p.")
    m = acc.merge(last[inf.CELL + ["system", "seed", "execution", "global_accuracy"]],
                  on=inf.CELL + ["system", "seed", "execution"])
    out.append(f"Check, final accuracy equals last-round global accuracy: max |d| {(m.accuracy - m.global_accuracy).abs().max():.6f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    acc, rounds = load_new()
    fx, first = load_parity_fixed(), load_parity_systems()
    first = first[first.system.isin(["Random", "GRADF"])]
    runs = {"first run": first}
    for e in EXECUTIONS:
        runs[f"exec {e}"] = acc[acc.execution == e]
    out = []
    regression(acc, first, out)
    a, b = (acc[(acc.execution == e) & (acc.system == "GRADF")].set_index(inf.CELL + ["seed"]).accuracy for e in EXECUTIONS)
    out.append(f"GRADF exec 1 vs exec 2: {((a - b).abs() > 0).sum()} of {len(a)} pairs differ")
    repetition(runs, out)
    versus_best(runs, fx, out)
    mechanism(acc, rounds, fx, out)
    text = "\n".join(out) + "\n"
    with open(f"{OUT}/analysis.txt", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
