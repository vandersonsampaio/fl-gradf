"""
Two descriptive calculations on existing runs, requested after the X5 analysis and not specified
in results/p1_campaign/ADDENDUM_3.md (post hoc; no new jobs):

  E2 over D_7 at factor 1/3  cross-fitted headroom over the 7 rules on the 21 E2 cells (and on
                             the 18 without fltrust_aligned), root 100, from e1_main, next to the
                             same quantity over D_R.
  Systems against B7_-s      Random, attack-aware, GRADF, FedStrategist and AdaAggRL against the
                             cross-fitted best of the 7 plain rules, at factor 1/3 (e3_main) and
                             factor 1 (x1_systems), on the 12 E3 cells, with inference.describe and
                             the seed bootstrap that repeats the selection; B_R,-s (plain) next to it.
No Holm correction and no confirmatory test. Writes results/p1_campaign/analysis_x5/extra.json.

Usage: venv/bin/python scripts/p1/analyze_x5_extra.py
"""

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import analyze_x5 as ax  # noqa: E402
import campaign_job as cj  # noqa: E402
import inference as inf  # noqa: E402

FA = "fltrust_aligned"
SYSTEMS = {"random": "Random", "attack_aware": "Attack-aware", "gradf": "GRADF",
           "fedstrategist": "FedStrategist", "adaaggrl": "AdaAggRL"}


def e2_over(fx, rules):
    acc, cells, seeds = inf.as_array(fx, rules)
    mean = acc.mean(axis=1)
    d_in = mean.mean(axis=0).argmax()
    agree = []
    for i in range(acc.shape[1]):
        sel = (acc.sum(axis=1) - acc[:, i, :]) / (acc.shape[1] - 1)
        agree.append(rules[sel.mean(axis=0).argmax()])
    winners = Counter(rules[j] for j in mean.argmax(axis=1))
    return {"n_cells": len(cells), **ax.headroom(fx, rules), "dstar_insample": rules[d_in],
            "dstar_loso_choices": Counter(agree), "cell_winners": winners,
            "n_cells_best_outside_DR": int(sum(v for k, v in winners.items() if k not in inf.D_R))}


def e2_part():
    fx = ax.read("e1_main")
    arms = {f"{r}_r100": r for r in ax.D_7}
    fx = fx[fx.arm.isin(arms)].copy()
    fx["strategy"] = fx.arm.map(arms)
    fx = fx[["alpha", "attack_type", "seed", "strategy", "accuracy"]]
    out = {}
    for name, sub in (("all_21", fx), ("no_fltrust_aligned_18", fx[fx.attack_type != FA])):
        out[name] = {"D_R": e2_over(sub, inf.D_R), "D_7": e2_over(sub, ax.D_7)}
    means = fx.groupby(["alpha", "attack_type", "strategy"]).accuracy.mean().unstack()[ax.D_7]
    means.to_csv(f"{ax.OUT}/extra_e2_factor_1_3_cell_means.csv")
    out["rule_means_21"] = means.mean().to_dict()
    return out


def systems_part():
    rows = []
    for factor, block in (("1/3", "e3_main"), ("1", "x1_systems")):
        fx = ax.fixed_rules(factor)
        df = ax.in_e3(ax.read(block))
        for rules, ref_name in ((ax.D_7, "B7_-s plain"), (inf.D_R, "BR_-s plain")):
            B, _, chosen = ax.crossfit(fx, rules)
            acc, cells, seeds = inf.as_array(fx, rules)
            for arm, name in SYSTEMS.items():
                s = df[df.arm == arm].set_index(inf.CELL + ["seed"]).accuracy.sort_index()
                d = (s - B).dropna().groupby(level="seed").mean().values
                sysv = np.array([[s[(*c, k)] for k in seeds] for c in cells])
                r = {"factor": factor, "system": name, "reference": ref_name,
                     "dstar": Counter(chosen.values()).most_common(1)[0][0],
                     "system_mean": float(s.mean()), **inf.describe(d), **ax.boot_minus_dstar(acc, sysv)}
                r["mean"] = inf.describe(d)["mean"]
                lo, hi = r["ci95_lo"], r["ci95_hi"]
                blo, bhi = r["boot_ci95"]
                r["describe_and_bootstrap_agree_on_0"] = (lo > 0) == (blo > 0) and (hi < 0) == (bhi < 0)
                rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(f"{ax.OUT}/extra_systems_vs_dstar.csv", index=False)
    return t


def main():
    flag = f"{ax.OUT}/.extra_done"
    if os.path.exists(flag):
        raise SystemExit(f"this analysis already ran once ({flag})")
    pd.set_option("display.width", 250)
    e2 = e2_part()
    t = systems_part()
    out = {"E2_factor_1_3": e2, "systems_vs_dstar": t.to_dict(orient="records")}
    ax.dump(out, f"{ax.OUT}/extra.json")
    for g in ("all_21", "no_fltrust_aligned_18"):
        for s in ("D_R", "D_7"):
            r = e2[g][s]
            print(g, s, "H_loso", round(r["H_loso"], 4), np.round(r["H_loso_ci95"], 4), "d*", r["dstar_insample"],
                  dict(r["dstar_loso_choices"]), "winners", dict(r["cell_winners"]))
    print(t[["factor", "system", "reference", "dstar", "system_mean", "mean", "ci95_lo", "ci95_hi", "boot_ci95",
             "p_wilcoxon", "p_tost", "category", "describe_and_bootstrap_agree_on_0"]].round(4).to_string(index=False))
    open(flag, "w").close()


if __name__ == "__main__":
    main()
