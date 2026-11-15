"""
MNIST: the 7-rule reference set and the hardened reference (results/p1_parity/mnist_trailing/ADDENDUM_3.md).
Descriptive re-analysis on the trailing mean (rounds 11-15), seeds 42-51, root 100; not blind.

  A  D_7: cross-fitted headroom over D_7 next to D_R (21, 18 and 12 cells), d*, cell winners;
     every system against B7_-s (plain), with inference.describe and the seed bootstrap that repeats
     the selection, and against B_R,-s (plain) as a consistency check with RESULTS.md.
  B  hardened: every system against the cross-fitted best hardened rule of D_R (B_H,-s), and the
     hardened reference against the plain one, on the 18 and 12 cells.

No Holm correction. Writes results/p1_parity/mnist_trailing/analysis_addendum3/.

Usage: ../fl-fedmed/venv/bin/python scripts/p1/analyze_mnist_addendum3.py
"""

import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import analyze_mnist_trailing as amt  # noqa: E402
import inference as inf  # noqa: E402

RAW = amt.RAW
OUT = f"{RAW}/analysis_addendum3"
D_R = inf.D_R
D_7 = D_R + ["fedavg", "fedprox", "clustering"]
FA = amt.FA
GRIDS = {"21": lambda d: d, "18": lambda d: d[d.attack_type != FA],
         "12": lambda d: d[d.attack_type.isin(amt.LEVEL1)]}
SYSTEMS = ["Random", "Oracle (decoupled)", "GRADF", "FedStrategist", "AdaAggRL"]
N_BOOT = 20000


def crossfit(fx, rules):
    acc, cells, seeds = inf.as_array(fx, rules)
    total, k = acc.sum(axis=1), len(seeds)
    B, chosen = {}, {}
    for i, s in enumerate(seeds):
        d = ((total - acc[:, i, :]) / (k - 1)).mean(axis=0).argmax()
        chosen[s] = rules[d]
        for j, c in enumerate(cells):
            B[(*c, s)] = acc[j, i, d]
    idx = pd.MultiIndex.from_tuples(list(B), names=inf.CELL + ["seed"])
    return pd.Series(list(B.values()), index=idx), chosen


def boot_minus_dstar(acc, sysv, seed=0):
    def stat(idx):
        sub, sv, vals = acc[:, idx, :], sysv[:, idx], []
        for i in range(len(idx)):
            keep = idx != idx[i]
            if keep.any():
                d = sub[:, keep, :].mean(axis=1).mean(axis=0).argmax()
                vals.append(np.mean(sv[:, i] - sub[:, i, d]))
        return np.mean(vals)
    n = acc.shape[1]
    rng = np.random.default_rng(seed)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 2:
            idx[0] = (idx[0] + 1) % n
        boots[b] = stat(idx)
    return stat(np.arange(n)), np.percentile(boots, [2.5, 97.5]).tolist()


def headroom(fx, rules, seed=0):
    acc, cells, seeds = inf.as_array(fx, rules)
    h, _, h_in = inf.loso_headroom(acc)
    rng = np.random.default_rng(seed)
    n = len(seeds)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 2:
            idx[0] = (idx[0] + 1) % n
        boots[b] = inf.loso_headroom(acc, idx)[0]
    mean = acc.mean(axis=1)
    _, chosen = crossfit(fx, rules)
    winners = Counter(rules[j] for j in mean.argmax(axis=1))
    return {"H_loso": h, "H_loso_ci95": np.percentile(boots, [2.5, 97.5]).tolist(), "H_insample": h_in,
            "dstar_insample": rules[mean.mean(axis=0).argmax()], "dstar_loso_choices": Counter(chosen.values()),
            "cell_winners": winners, "n_cells_best_outside_DR": int(sum(v for k, v in winners.items() if k not in D_R))}


def versus(fx, P, rules, ref_name, systems, boot=True):
    B, chosen = crossfit(fx, rules)
    acc, cells, seeds = inf.as_array(fx, rules)
    rows = []
    for s in systems:
        d = (P[s] - B).dropna().groupby(level="seed").mean().values
        r = {"system": s, "reference": ref_name, "dstar": Counter(chosen.values()).most_common(1)[0][0],
             "system_mean": float(P[s].mean()), **inf.describe(d)}
        if boot:
            sysv = np.array([[P[s][(*c, k)] for k in seeds] for c in cells])
            r["boot_mean"], r["boot_ci95"] = boot_minus_dstar(acc, sysv)
            r["agree_on_0"] = (r["ci95_lo"] > 0) == (r["boot_ci95"][0] > 0) and (r["ci95_hi"] < 0) == (r["boot_ci95"][1] < 0)
        rows.append(r)
    return rows, B


def main():
    os.makedirs(OUT, exist_ok=True)
    flag = f"{OUT}/.done"
    if os.path.exists(flag):
        raise SystemExit("this analysis already ran once")
    col = amt.METRICS["trailing"]
    fx_all = amt.load_fixed(RAW)
    fx = fx_all[fx_all.root == 100].drop(columns="root").assign(accuracy=lambda d: d[col])
    sy = pd.concat([amt.load_systems(RAW), amt.load_adaaggrl(RAW)], ignore_index=True).assign(accuracy=lambda d: d[col])
    hx = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RAW}/mnist_hardened/raw/*.csv")], ignore_index=True)
    hx = hx.assign(accuracy=hx[col], strategy=hx.strategy.str.replace("_hardened", "", regex=False))
    checks = {"fixed_root100_rows": len(fx), "fixed_rules": sorted(fx.strategy.unique()),
              "hardened_rows": len(hx), "hardened_rules": sorted(hx.strategy.unique()),
              "system_rows": sy.groupby("system").size().to_dict()}
    if len(fx) != 1470 or len(hx) != 840:
        raise SystemExit(f"incomplete: {checks}")
    out = {"checks": checks, "A_headroom": {}, "A_systems": [], "B_systems": []}
    for g, pick in GRIDS.items():
        out["A_headroom"][g] = {"D_R": headroom(pick(fx), D_R), "D_7": headroom(pick(fx), D_7)}
    out["A_rule_means_21"] = fx.groupby("strategy").accuracy.mean().to_dict()
    out["B_hardened_rule_means_21"] = hx.groupby("strategy").accuracy.mean().to_dict()
    for g in ("18", "12"):
        f, h, s = GRIDS[g](fx), GRIDS[g](hx), GRIDS[g](sy)
        P = s.pivot_table(index=inf.CELL + ["seed"], columns="system", values="accuracy")
        for rules, name in ((D_7, "B7_-s plain"), (D_R, "BR_-s plain")):
            rows, _ = versus(f, P, rules, name, SYSTEMS)
            out["A_systems"] += [{"cells": g, **r} for r in rows]
        rows, BH = versus(h, P, D_R, "BH_-s hardened", SYSTEMS)
        out["B_systems"] += [{"cells": g, **r} for r in rows]
        BR, _ = crossfit(f, D_R)
        out["B_systems"].append({"cells": g, "system": "hardened B_-s", "reference": "BR_-s plain",
                                 **inf.describe((BH - BR).groupby(level="seed").mean().values)})
    with open(f"{OUT}/analysis.json", "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item")
                  else (dict(o) if isinstance(o, Counter) else str(o)))
    pd.set_option("display.width", 250)
    lines = [f"checks: {checks}", ""]
    for g, r in out["A_headroom"].items():
        for k in ("D_R", "D_7"):
            x = r[k]
            lines.append(f"A {g} cells {k}: H_LOSO {100 * x['H_loso']:.2f} ({100 * x['H_loso_ci95'][0]:.2f}; "
                         f"{100 * x['H_loso_ci95'][1]:.2f}) d* {x['dstar_insample']} {dict(x['dstar_loso_choices'])} "
                         f"winners {dict(x['cell_winners'])}")
    lines.append(f"A rule means (21 cells, %): { {k: round(100 * v, 2) for k, v in out['A_rule_means_21'].items()} }")
    lines.append(f"B hardened rule means (21 cells, %): { {k: round(100 * v, 2) for k, v in out['B_hardened_rule_means_21'].items()} }")
    cols = ["cells", "system", "reference", "dstar", "system_mean", "mean", "ci95_lo", "ci95_hi", "boot_ci95",
            "p_wilcoxon", "p_tost", "category", "agree_on_0"]
    lines += ["", "A systems:", pd.DataFrame(out["A_systems"])[cols].round(4).to_string(index=False)]
    b = pd.DataFrame(out["B_systems"])
    lines += ["", "B hardened reference:", b.reindex(columns=cols).round(4).to_string(index=False)]
    text = "\n".join(lines)
    with open(f"{OUT}/analysis.txt", "w") as fh:
        fh.write(text + "\n")
    print(text)
    open(flag, "w").close()


if __name__ == "__main__":
    main()
