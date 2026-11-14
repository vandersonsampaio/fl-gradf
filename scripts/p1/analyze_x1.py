"""
Exploratory analysis of X1, E2 and E3 at global factor 1 (results/p1_campaign/ADDENDUM_2.md,
Sec. 3), run once on the complete X1 blocks.

Reads the raw JSONs, checks the X1 blocks against the job lists of x1_job.py, the X1 code hash
and the factor recorded in each run, and applies scripts/p1/inference.py on the primary metric
(mean test accuracy over rounds 106-110), as the confirmatory analysis does:
  E2  D_R = {FLTrust, median, trimmed mean, Krum} at root 100 and factor 1: FLTrust and the
      trimmed mean from e1_sens_factor, median and Krum from x1_fixed. H_LOSO with the seed
      bootstrap, in-sample H, d* and LOSO agreement, on 21 and 18 cells.
  E3  12 cells: systems vs Random, and systems vs B_-s (GRADF and Random against the hardened
      rules of x1_hardened, the others against the plain rules); Holm within each family.
      The other reference, and attack-aware against the hardened rules, are descriptive.
  Side by side with factor 1/3, read from the confirmatory analysis (analysis/), descriptive.
Writes results/p1_campaign/analysis_x1/.

Usage: venv/bin/python scripts/p1/analyze_x1.py
"""

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

R = cj.OUT
OUT = f"{R}/analysis_x1"
CONF = f"{R}/analysis"
FA = "fltrust_aligned"
SYSTEMS = {"random": "Random", "attack_aware": "Attack-aware", "gradf": "GRADF",
           "fedstrategist": "FedStrategist", "adaaggrl": "AdaAggRL"}
MAIN = ["Attack-aware", "GRADF", "FedStrategist", "AdaAggRL"]
LAYERED = ["Random", "Attack-aware", "GRADF"]
COLS = ["family", "system", "reference", "mean", "ci95_lo", "ci95_hi", "p_wilcoxon", "p_wilcoxon_holm",
        "p_t", "p_t_holm", "d_z", "p_tost", "category"]
BLOCKS_USED = ("e1_sens_factor",) + tuple(x1_job.BLOCKS)


def load():
    rows = []
    for block in BLOCKS_USED:
        for path in glob.glob(f"{R}/{block}/raw/*.json"):
            with open(path) as f:
                d = json.load(f)
            layers = d["layers"]
            rows.append({"block": d["block"], "arm": d["arm"], "alpha": d["alpha"], "attack_type": d["attack"],
                         "seed": d["seed"], "accuracy": d["primary_trailing5"], "machine": d["machine"],
                         "code": d["code_sha256"], "campaign_code": d.get("campaign_code_sha256"),
                         "global_lr": d.get("global_lr"),
                         "pretraining_lrs": sorted({p.get("global_lr") for p in d.get("pretraining") or []
                                                    if isinstance(p, dict) and "global_lr" in p}),
                         "l4_reject": np.mean([not x["accepted"] for x in layers]) if layers else np.nan,
                         "l1_reject": np.mean([len(x["layer1_rejected"]) > 0 for x in layers]) if layers else np.nan})
    return pd.DataFrame(rows)


def check(df):
    def frozen(name):
        with open(f"{R}/{name}") as f:
            return f.read().split()[0]
    x1 = df[df.block.isin(x1_job.BLOCKS)]
    out = {"n_results_x1": len(x1),
           "x1_code_hash_ok": bool((x1.code == frozen("X1_CODE.sha256")).all()),
           "campaign_code_hash_ok": bool((x1.campaign_code == frozen("CODE.sha256")).all()),
           "factor_one_ok": bool((x1.global_lr == 1.0).all()
                                 and all(set(p) <= {1.0} for p in x1.pretraining_lrs)),
           "machines_x1": Counter(x1.machine)}
    for block, keys in [(b, x1_job.job_keys(b)) for b in x1_job.BLOCKS] + [("e1_sens_factor", cj.job_keys("e1_sens_factor"))]:
        want = set(keys)
        have = set(df[df.block == block][["arm", "alpha", "attack_type", "seed"]].itertuples(index=False, name=None))
        out[block] = {"expected": len(want), "found": len(have & want), "missing": len(want - have),
                      "extra": len(have - want)}
        if want != have:
            raise SystemExit(f"block {block} is incomplete or has extra results: {out[block]}")
    if not (out["x1_code_hash_ok"] and out["campaign_code_hash_ok"] and out["factor_one_ok"]):
        raise SystemExit(f"X1 runs fail the hash or factor check: {out}")
    with open(f"{R}/x1_dispatch.log") as f:
        log = f.read().split("\n")
    out["infra_requeued"] = sum(" INFRA " in x and x.endswith("requeued") for x in log)
    out["infra_dropped"] = sum(" INFRA " in x and x.endswith("dropped") for x in log)
    out["fail"] = sum(" FAIL " in x for x in log)
    return out


def fixed(df, block, arms, rename):
    sub = df[(df.block == block) & df.arm.isin(arms)].copy()
    sub["strategy"] = sub.arm.map(rename)
    return sub[["alpha", "attack_type", "seed", "strategy", "accuracy"]]


def e3_rows(P, fam, systems, ref_of):
    rows = []
    for s in systems:
        ref_name, ref = ref_of(s)
        d = (P[s] - ref).dropna()
        rows.append({"family": fam, "system": s, "reference": ref_name,
                     **inf.describe(d.groupby(level="seed").mean().values)})
    return rows


def finish(rows):
    return inf.add_holm(inf.add_holm(pd.DataFrame(rows)), col="p_t")


def analyze(df):
    """Every X1 output from the loaded runs (no I/O)."""
    res = {}
    # ---- E2 at factor 1 (root 100, D_R)
    fx1 = pd.concat([
        fixed(df, "e1_sens_factor", ["fltrust_r100", "trimmed_mean_r100"],
              {"fltrust_r100": "fltrust", "trimmed_mean_r100": "trimmed_mean"}),
        fixed(df, "x1_fixed", ["median_r100", "krum_r100"], {"median_r100": "median", "krum_r100": "krum"})])
    for name, fx in (("all_21", fx1), ("no_fltrust_aligned_18", fx1[fx1.attack_type != FA])):
        res[f"E2_{name}"] = {**inf.e2_headroom(fx), **inf.e2_selection_stability(fx)}
    cell_means_fx = fx1.groupby(["alpha", "attack_type", "strategy"]).accuracy.mean().unstack()[inf.D_R]
    res["E2_rule_means_21"] = cell_means_fx.mean().to_dict()
    res["E2_cell_winners_21"] = Counter(cell_means_fx.idxmax(axis=1))

    # ---- E3 at factor 1 (12 cells)
    e3c = set(cj.E3_CELLS)
    fx12 = fx1[[(a, t) in e3c for a, t in zip(fx1.alpha, fx1.attack_type)]]
    hx12 = fixed(df, "x1_hardened", [f"{r}_hardened" for r in inf.D_R], {f"{r}_hardened": r for r in inf.D_R})
    sy = df[df.block == "x1_systems"].copy()
    sy["system"] = sy.arm.map(SYSTEMS)
    P = sy.pivot_table(index=inf.CELL + ["seed"], columns="system", values="accuracy")
    B_plain, O_plain = inf.crossfit_references(fx12)
    B_hard, O_hard = inf.crossfit_references(hx12)
    conf = e3_rows(P, "vs Random", MAIN, lambda s: ("Random", P["Random"]))
    conf += e3_rows(P, "vs B_-s", ["Random"] + MAIN,
                    lambda s: ("hardened", B_hard) if s in ("Random", "GRADF") else ("plain", B_plain))
    t_main = finish(conf)
    desc = e3_rows(P, "vs B_-s plain (descriptive)", ["Random", "GRADF"], lambda s: ("plain", B_plain))
    desc += e3_rows(P, "vs B_-s hardened (descriptive)", ["Attack-aware"], lambda s: ("hardened", B_hard))
    desc += e3_rows(P, "hardened B_-s vs plain B_-s", ["Random"], lambda s: ("plain", B_plain))
    desc[-1]["system"] = "hardened B_-s"
    desc[-1].update(inf.describe((B_hard - B_plain).groupby(level="seed").mean().values))
    t_desc = finish(desc)

    Hc_plain = (O_plain - B_plain).groupby(level=inf.CELL).mean()
    Hc_hard = (O_hard - B_hard).groupby(level=inf.CELL).mean()
    cap = {}
    for s in P.columns:
        Bref, Hc = (B_hard, Hc_hard) if s in LAYERED else (B_plain, Hc_plain)
        gain = (P[s] - Bref).groupby(level=inf.CELL).mean()
        cap[s] = {"cells_beating_B": int((gain > 0).sum()),
                  "captured_in_headroom_cells": int(((gain > 0) & (Hc > 0)).sum())}
    acc = fx12.groupby("strategy").accuracy.mean()
    hacc = hx12.groupby("strategy").accuracy.mean()
    res["E3_descriptive"] = {
        "cells_with_crossfit_headroom_plain": int((Hc_plain > 0).sum()),
        "cells_with_crossfit_headroom_hardened": int((Hc_hard > 0).sum()),
        "dstar_plain": inf.D_R[int(np.argmax([acc[r] for r in inf.D_R]))],
        "dstar_hardened": inf.D_R[int(np.argmax([hacc[r] for r in inf.D_R]))],
        "plain_rule_means": acc[inf.D_R].to_dict(), "hardened_rule_means": hacc[inf.D_R].to_dict(),
        "system_means": P.mean().to_dict(), "capture": cap,
        "layer4_reject_rate": sy.groupby("system").l4_reject.mean().dropna().to_dict(),
        "layer1_reject_rate": sy.groupby("system").l1_reject.mean().dropna().to_dict(),
        "hardened_layer4_reject_rate": df[df.block == "x1_hardened"].groupby("arm").l4_reject.mean().to_dict(),
        "hardened_layer1_reject_rate": df[df.block == "x1_hardened"].groupby("arm").l1_reject.mean().to_dict(),
    }
    cell_tab = P.groupby(level=inf.CELL).mean().join(
        fx12.groupby(["alpha", "attack_type", "strategy"]).accuracy.mean().unstack()[inf.D_R]).join(
        hx12.groupby(["alpha", "attack_type", "strategy"]).accuracy.mean().unstack()[inf.D_R], rsuffix="_hardened")
    res["machine_share_by_block"] = {b: Counter(g.machine) for b, g in df.groupby("block")}
    return res, t_main, t_desc, cell_means_fx, cell_tab


def side_by_side(res, t_main):
    """Factor 1/3 (confirmatory analysis, read from its outputs) next to factor 1."""
    with open(f"{CONF}/analysis.json") as f:
        third = json.load(f)
    t3 = pd.read_csv(f"{CONF}/e3_confirmatory.csv")
    key = ["family", "system", "reference"]
    e3 = t3[key + ["mean", "ci95_lo", "ci95_hi", "p_wilcoxon_holm", "category"]].merge(
        t_main[key + ["mean", "ci95_lo", "ci95_hi", "p_wilcoxon_holm", "category"]],
        on=key, how="outer", suffixes=("_f1_3", "_f1"))
    e2 = {}
    for g in ("all_21", "no_fltrust_aligned_18"):
        e2[g] = {f: {k: r[k] for k in ("dstar_insample", "dstar_agreement", "H_loso", "H_loso_ci95",
                                        "H_insample", "category")}
                 for f, r in (("factor_1_3", third[f"E2_{g}"]), ("factor_1", res[f"E2_{g}"]))}
    e3d = {f: {k: r[k] for k in ("dstar_plain", "dstar_hardened", "cells_with_crossfit_headroom_plain",
                                 "cells_with_crossfit_headroom_hardened")}
           for f, r in (("factor_1_3", third["E3_descriptive"]), ("factor_1", res["E3_descriptive"]))}
    return {"E2": e2, "E3_dstar": e3d}, e3


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item")
                  else (dict(o) if isinstance(o, Counter) else str(o)))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()
    checks = check(df)
    res, t_main, t_desc, cell_fx, cell_tab = analyze(df)
    res = {"checks": checks, **res}
    sbs, e3_sbs = side_by_side(res, t_main)
    res["side_by_side"] = sbs
    df.drop(columns="pretraining_lrs").to_csv(f"{OUT}/x1_runs.csv", index=False)
    t_main.to_csv(f"{OUT}/e3_factor1.csv", index=False)
    t_desc.to_csv(f"{OUT}/e3_factor1_descriptive.csv", index=False)
    e3_sbs.to_csv(f"{OUT}/e3_side_by_side.csv", index=False)
    cell_fx.to_csv(f"{OUT}/e2_factor1_cell_means_r100.csv")
    cell_tab.to_csv(f"{OUT}/e3_factor1_cell_means.csv")
    dump(res, f"{OUT}/analysis_x1.json")

    short = lambda r: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()  # noqa: E731
                       if k not in ("H_loso_per_seed", "sd")}
    pd.set_option("display.width", 250)
    print("checks", checks)
    for g in ("all_21", "no_fltrust_aligned_18"):
        print("E2 factor 1", g, short(res[f"E2_{g}"]))
    print("E2 rule means (21)", {k: round(v, 4) for k, v in res["E2_rule_means_21"].items()},
          "cell winners", dict(res["E2_cell_winners_21"]))
    print("\nE3 factor 1\n", t_main[COLS].round(4).to_string(index=False))
    print("\nE3 factor 1 descriptive\n", t_desc[COLS].round(4).to_string(index=False))
    print("\nE3 side by side\n", e3_sbs.round(4).to_string(index=False))
    print("\n", json.dumps(res["side_by_side"], indent=1, default=float))
    print("\n", json.dumps(res["E3_descriptive"], indent=1, default=float))


if __name__ == "__main__":
    main()
