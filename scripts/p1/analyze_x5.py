"""
Descriptive analysis of X5, the action set against the reference set
(results/p1_campaign/ADDENDUM_3.md, Sec. 3). Two parts, each run once:

  --part existing  Runs that existed before X5:
                   X5c at factor 1/3, the 7 rules of e1_main against D_R on the 12 E3 cells;
                   X5b for Random, attack-aware and GRADF, from their stored Layer-3 logs.
  --part full      After the X5 blocks are complete:
                   X5c at factor 1 (with x5_fixed);
                   A_FedStrategist - B7_-s per factor, with inference.describe and a seed
                   bootstrap that repeats the selection of d*_7;
                   per-cell FedStrategist against the best of the 7 rules;
                   X5b for FedStrategist from the replays, and FedStrategist minus the rule it
                   chooses most, in the cells where it beats the plain d*_R;
                   the reading of ADDENDUM_3 Sec. 4.
Writes results/p1_campaign/analysis_x5/.

Usage: venv/bin/python scripts/p1/analyze_x5.py --part existing|full
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import campaign_job as cj  # noqa: E402
import inference as inf  # noqa: E402
import x1_job  # noqa: E402
import x5_job  # noqa: E402

R = cj.OUT
OUT = f"{R}/analysis_x5"
D_R = inf.D_R
D_7 = D_R + list(x5_job.OUTSIDE_DR)
E3 = set(cj.E3_CELLS)
N_BOOT = 20000
LAYERED = {"random": "Random", "attack_aware": "Attack-aware", "gradf": "GRADF"}


def read(block):
    rows = []
    for path in glob.glob(f"{R}/{block}/raw/*.json"):
        with open(path) as f:
            d = json.load(f)
        rows.append({**{k: d.get(k) for k in ("block", "arm", "alpha", "seed", "code_sha256", "machine")},
                     "attack_type": d["attack"], "accuracy": d["primary_trailing5"], "layers": d.get("layers"),
                     "actions": d.get("actions"), "identical": d.get("identical_params"),
                     "global_lr": d.get("global_lr")})
    return pd.DataFrame(rows)


def in_e3(df):
    return df[[(a, t) in E3 for a, t in zip(df.alpha, df.attack_type)]]


def fixed_frame(df, arms):
    sub = df[df.arm.isin(arms)].copy()
    sub["strategy"] = sub.arm.map(arms)
    return in_e3(sub)[["alpha", "attack_type", "seed", "strategy", "accuracy"]]


def fixed_rules(factor):
    """7-rule long frame on the 12 E3 cells, root 100."""
    if factor == "1/3":
        return fixed_frame(read("e1_main"), {f"{r}_r100": r for r in D_7})
    parts = [fixed_frame(read("e1_sens_factor"), {"fltrust_r100": "fltrust", "trimmed_mean_r100": "trimmed_mean"}),
             fixed_frame(read("x1_fixed"), {"median_r100": "median", "krum_r100": "krum"}),
             fixed_frame(read("x5_fixed"), {f"{r}_r100": r for r in x5_job.OUTSIDE_DR})]
    return pd.concat(parts)


def crossfit(fx, rules):
    """B_-s and O_-s over `rules`, as inference.crossfit_references does over D_R."""
    acc, cells, seeds = inf.as_array(fx, rules)
    total, k, rows = acc.sum(axis=1), len(seeds), np.arange(len(cells))
    B, O, chosen = {}, {}, {}
    for i, s in enumerate(seeds):
        sel = (total - acc[:, i, :]) / (k - 1)
        r_cell, d_glob = sel.argmax(axis=1), sel.mean(axis=0).argmax()
        chosen[s] = rules[d_glob]
        for j, c in enumerate(cells):
            B[(*c, s)] = acc[j, i, d_glob]
            O[(*c, s)] = acc[j, i, r_cell[j]]
    idx = pd.MultiIndex.from_tuples(list(B), names=inf.CELL + ["seed"])
    return pd.Series(list(B.values()), index=idx), pd.Series(list(O.values()), index=idx), chosen


def boot_minus_dstar(acc, sysv, n_boot=N_BOOT, seed=0):
    """Mean over seeds of A_sys - A_{d*} with d* selected without the evaluated seed; the
    bootstrap resamples seeds and repeats the selection, leaving every copy of the held-out
    seed out of it. acc: [cell, seed, rule], sysv: [cell, seed]."""
    def stat(idx):
        sub, sv = acc[:, idx, :], sysv[:, idx]
        vals = []
        for i in range(len(idx)):
            keep = idx != idx[i]
            if not keep.any():
                continue
            d = sub[:, keep, :].mean(axis=1).mean(axis=0).argmax()
            vals.append(np.mean(sv[:, i] - sub[:, i, d]))
        return np.mean(vals)
    n = acc.shape[1]
    point = stat(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 2:
            idx[0] = (idx[0] + 1) % n
        boots[b] = stat(idx)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": point, "boot_ci95": [lo, hi]}


def headroom(fx, rules, n_boot=N_BOOT, seed=0):
    acc, cells, seeds = inf.as_array(fx, rules)
    h, _, h_in = inf.loso_headroom(acc)
    rng = np.random.default_rng(seed)
    n = len(seeds)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 2:
            idx[0] = (idx[0] + 1) % n
        boots[b] = inf.loso_headroom(acc, idx)[0]
    return {"H_loso": h, "H_loso_ci95": np.percentile(boots, [2.5, 97.5]).tolist(), "H_insample": h_in}


def x5c(fx):
    means = fx.groupby(["alpha", "attack_type", "strategy"]).accuracy.mean().unstack()[D_7]
    glob_means = means.mean()
    best7, bestR = means.idxmax(axis=1), means[D_R].idxmax(axis=1)
    outside = means[~best7.isin(D_R)]
    B7, _, ch7 = crossfit(fx, D_7)
    BR, _, chR = crossfit(fx, D_R)
    return {
        "rule_means": glob_means.to_dict(),
        "dstar7_insample": glob_means.idxmax(), "dstarR_insample": glob_means[D_R].idxmax(),
        "dstar7_loso_choices": Counter(ch7.values()), "dstarR_loso_choices": Counter(chR.values()),
        "cells_best_outside_DR": {f"{a}|{t}": {"best7": best7[(a, t)], "acc_best7": means.loc[(a, t)].max(),
                                               "bestR": bestR[(a, t)], "acc_bestR": means.loc[(a, t), D_R].max()}
                                  for a, t in outside.index},
        "n_cells_best_outside_DR": int(len(outside)),
        "H_7": headroom(fx, D_7), "H_R": headroom(fx, D_R),
        "B7_minus_BR": inf.describe((B7 - BR).groupby(level="seed").mean().values),
    }, means


def shares(actions_by_cell):
    out = {}
    for cell, acts in actions_by_cell.items():
        c = Counter(acts)
        tot = sum(c.values())
        out[cell] = {r: round(c.get(r, 0) / tot, 4) for r in D_7}
    return out


def layered_frequencies(block):
    df = in_e3(read(block))
    res = {}
    for arm, name in LAYERED.items():
        sub = df[df.arm == arm]
        by_cell = {}
        for (a, t), g in sub.groupby(["alpha", "attack_type"]):
            by_cell[f"{a}|{t}"] = [x["strategy"] for lay in g.layers for x in lay if x.get("strategy")]
        res[name] = shares(by_cell)
    return res


def systems(block):
    df = in_e3(read(block))
    return df[df.arm == "fedstrategist"].set_index(inf.CELL + ["seed"]).accuracy.sort_index()


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item")
                  else (dict(o) if isinstance(o, Counter) else str(o)))


def part_existing():
    fx = fixed_rules("1/3")
    res, means = x5c(fx)
    out = {"X5c_factor_1_3": res,
           "X5b_layered_factor_1_3": layered_frequencies("e3_main"),
           "X5b_layered_factor_1": layered_frequencies("x1_systems")}
    means.to_csv(f"{OUT}/x5c_factor_1_3_cell_means.csv")
    dump(out, f"{OUT}/existing.json")
    return out


def reading(row):
    lo, hi = row["ci95_lo"], row["ci95_hi"]
    blo, bhi = row["boot_ci95"]
    if (lo > 0) != (blo > 0) or (hi < 0) != (bhi < 0):
        return "inconclusive (describe and bootstrap CIs disagree on 0)"
    if lo > 0:
        return "exploratory evidence of useful selection beyond the action set"
    if row["category"] in ("equivalence", "inferiority"):
        return "gain attributed to the action set"
    return "inconclusive (no evidence)"


def part_full():
    rep = read("x5_replay")
    xf = read("x5_fixed")
    # A replay that did not reproduce its stored run is run again on the stored run's machine
    # (x5_replay_same). Where that reproduction is identical, its choices replace the replay's.
    same = read("x5_replay_same") if os.path.isdir(f"{R}/x5_replay_same/raw") else pd.DataFrame()
    n_not_identical = int((~rep.identical.astype(bool)).sum())
    n_same_identical = 0
    if len(same):
        key = ["arm", "alpha", "attack_type", "seed"]
        good = same[same.identical.astype(bool)].set_index(key)
        for i in rep.index[~rep.identical.astype(bool)]:
            k = tuple(rep.loc[i, key])
            if k in good.index:
                rep.at[i, "actions"] = good.loc[k, "actions"]
                rep.at[i, "identical"] = True
                n_same_identical += 1
    checks = {"x5_fixed": len(xf), "x5_replay": len(rep),
              "code_hash_ok": bool(pd.concat([xf, rep]).code_sha256.eq(open(f"{R}/X5_CODE.sha256").read().split()[0]).all()),
              "x5_fixed_factor_one": bool((xf.global_lr == 1.0).all()),
              "replays_not_identical_cross_machine": n_not_identical,
              "replaced_by_identical_same_machine_reproduction": n_same_identical,
              "replays_used": int(rep.identical.astype(bool).sum()),
              "replays_excluded": int((~rep.identical.astype(bool)).sum()),
              "same_machine_reproductions": len(same),
              "same_machine_not_identical": int((~same.identical.astype(bool)).sum()) if len(same) else 0}
    for block in x5_job.BLOCKS:
        want = set(x5_job.job_keys(block))
        have = set((read(block)[["arm", "alpha", "attack_type", "seed"]]).itertuples(index=False, name=None))
        if want != have:
            raise SystemExit(f"block {block} is incomplete: {len(have & want)}/{len(want)}")
    out = {"checks": checks}
    tables = []
    for factor, sys_block, rep_arm in (("1/3", "e3_main", "e3_main"), ("1", "x1_systems", "x1_systems")):
        fx = fixed_rules(factor)
        x5c_res, means = x5c(fx)
        means.to_csv(f"{OUT}/x5c_factor_{factor.replace('/', '_')}_cell_means.csv")
        fs = systems(sys_block)
        B7, O7, _ = crossfit(fx, D_7)
        BR, _, _ = crossfit(fx, D_R)
        rows = []
        for name, ref in (("B7_-s", B7), ("BR_-s", BR)):
            d = (fs - ref).dropna().groupby(level="seed").mean().values
            acc, cells, seeds = inf.as_array(fx, D_7 if name == "B7_-s" else D_R)
            sysv = np.array([[fs[(*c, s)] for s in seeds] for c in cells])
            row = {"factor": factor, "reference": name, **inf.describe(d), **boot_minus_dstar(acc, sysv)}
            row["mean"] = inf.describe(d)["mean"]
            rows.append(row)
        rows[0]["reading"] = reading(rows[0])
        tables += rows
        gainR = (fs - BR).groupby(level=inf.CELL).mean()
        cell = pd.DataFrame({"fedstrategist": fs.groupby(level=inf.CELL).mean(),
                             "best7_crossfit_O7": O7.groupby(level=inf.CELL).mean(),
                             "fs_minus_O7": (fs - O7).groupby(level=inf.CELL).mean(),
                             "fs_minus_BR": gainR, "beats_plain_dstarR": gainR > 0})
        ok = rep[(rep.arm == rep_arm) & rep.identical.astype(bool)]
        by_cell = {f"{a}|{t}": [x for acts in g.actions for x in acts] for (a, t), g in ok.groupby(["alpha", "attack_type"])}
        freq = shares(by_cell)
        most = {}
        for (a, t) in cell.index:
            key = f"{a}|{t}"
            top = max(freq[key], key=freq[key].get) if key in freq else None
            cell.loc[(a, t), "most_chosen"] = top
            if top is not None and cell.loc[(a, t), "beats_plain_dstarR"]:
                ref = fx[(fx.alpha == a) & (fx.attack_type == t) & (fx.strategy == top)].set_index("seed").accuracy
                d = (fs.xs((a, t), level=inf.CELL) - ref).dropna().values
                most[key] = {"rule": top, "share": freq[key][top], **inf.describe(d)}
        cell.to_csv(f"{OUT}/fedstrategist_cells_factor_{factor.replace('/', '_')}.csv")
        out[f"factor_{factor}"] = {"X5c": x5c_res, "fedstrategist_frequencies": freq,
                                   "fs_minus_most_chosen_in_winning_cells": most,
                                   "cells_beating_plain_dstarR": int(cell.beats_plain_dstarR.sum()),
                                   "cells_beating_O7": int((cell.fs_minus_O7 > 0).sum())}
    t = pd.DataFrame(tables)
    t.to_csv(f"{OUT}/fedstrategist_vs_dstar.csv", index=False)
    out["main"] = t.to_dict(orient="records")
    dump(out, f"{OUT}/full.json")
    return out, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["existing", "full"])
    part = ap.parse_args().part
    os.makedirs(OUT, exist_ok=True)
    flag = f"{OUT}/.{part}_done"
    if os.path.exists(flag):
        raise SystemExit(f"the {part} part already ran once ({flag})")
    pd.set_option("display.width", 250)
    if part == "existing":
        out = part_existing()
    else:
        out, t = part_full()
        print(t[["factor", "reference", "mean", "ci95_lo", "ci95_hi", "boot_ci95", "p_wilcoxon", "p_tost",
                 "category", "reading"]].round(4).to_string(index=False))
    print(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item")
                     else (dict(o) if isinstance(o, Counter) else str(o))))
    open(flag, "w").close()


if __name__ == "__main__":
    main()
