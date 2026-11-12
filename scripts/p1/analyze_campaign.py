"""
Confirmatory analysis of the P1 CIFAR-10 campaign (results/p1_campaign,
PREREGISTRATION.md Secs. 4-5), run once on the complete blocks.

Reads every raw JSON, checks the blocks against the frozen job lists and the
code hash, and applies scripts/p1/inference.py on the primary metric (mean
test accuracy over rounds 106-110):
  E1  D at factor 1/3 (H1) and at factor 1 (sensitivity), 21 and 18 cells;
      the server-steps factorial at factor 1 (descriptive).
  E2  H_LOSO with the seed bootstrap, root 100, factor 1/3 (H2).
  E3  12 cells: systems vs Random, and systems vs B_-s; GRADF and Random
      against the hardened rules (confirmatory) and the plain ones.
Writes results/p1_campaign/analysis/.

Usage: venv/bin/python scripts/p1/analyze_campaign.py
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

R = cj.OUT
OUT = f"{R}/analysis"
FA = "fltrust_aligned"
SYSTEMS = {"random": "Random", "attack_aware": "Attack-aware", "gradf": "GRADF",
           "fedstrategist": "FedStrategist", "adaaggrl": "AdaAggRL",
           "adaaggrl_pretrained": "AdaAggRL (pretrained)", "fedstrategist_detector": "FedStrategist (GRADF detector)"}
LAYERED = ["Random", "Attack-aware", "GRADF"]
COLS = ["family", "system", "reference", "mean", "ci95_lo", "ci95_hi", "p_wilcoxon", "p_wilcoxon_holm",
        "p_t", "p_t_holm", "d_z", "p_tost", "category"]


def load():
    rows = []
    for path in glob.glob(f"{R}/*/raw/*.json"):
        with open(path) as f:
            d = json.load(f)
        layers = d["layers"]
        rows.append({"block": d["block"], "arm": d["arm"], "alpha": d["alpha"], "attack_type": d["attack"],
                     "seed": d["seed"], "accuracy": d["primary_trailing5"], "final": d["final"],
                     "machine": d["machine"], "code": d["code_sha256"],
                     "l4_reject": np.mean([not x["accepted"] for x in layers]) if layers else np.nan,
                     "l1_reject": np.mean([len(x["layer1_rejected"]) > 0 for x in layers]) if layers else np.nan})
    return pd.DataFrame(rows)


def check(df):
    with open(f"{R}/CODE.sha256") as f:
        frozen = f.read().split()[0]
    out = {"n_results": len(df), "code_hash_ok": bool((df.code == frozen).all()),
           "machines": Counter(df.machine)}
    for block in cj.BLOCKS:
        want = {(a, al, at, s) for a, al, at, s in cj.job_keys(block)}
        have = set(df[df.block == block][["arm", "alpha", "attack_type", "seed"]].itertuples(index=False, name=None))
        out[block] = {"expected": len(want), "found": len(have & want), "missing": len(want - have),
                      "extra": len(have - want)}
        if want != have:
            raise SystemExit(f"block {block} is incomplete or has extra results: {out[block]}")
    with open(f"{R}/dispatch.log") as f:
        log = f.read().split("\n")
    out["infra_requeued"] = sum(" INFRA " in x and x.endswith("requeued") for x in log)
    out["infra_dropped"] = sum(" INFRA " in x and x.endswith("dropped") for x in log)
    out["fail"] = sum(" FAIL " in x for x in log)
    return out


def fixed(df, block, arms, rename=None):
    """Long fixed-rule frame (strategy column) from a block's arms."""
    sub = df[(df.block == block) & df.arm.isin(arms)].copy()
    sub["strategy"] = sub.arm.map(rename) if rename else sub.arm
    return sub[["alpha", "attack_type", "seed", "strategy", "accuracy"]]


def e1_frames(df, block, tm_big_block):
    """fx_small, fx_big for E1: FLTrust and TM at each root. TM at root 2,500
    is run only under fltrust_aligned (where the attack sees the root); on the
    other cells TM does not use the root and its root-100 run stands for both."""
    small = fixed(df, block, ["fltrust_r100", "trimmed_mean_r100"],
                  {"fltrust_r100": "fltrust", "trimmed_mean_r100": "trimmed_mean"})
    ft_big = fixed(df, block, ["fltrust_r2500"], {"fltrust_r2500": "fltrust"})
    tm_fa = fixed(df, tm_big_block, ["trimmed_mean_r2500"], {"trimmed_mean_r2500": "trimmed_mean"})
    tm_fa = tm_fa[tm_fa.attack_type == FA]
    tm_rest = small[(small.strategy == "trimmed_mean") & (small.attack_type != FA)]
    return small, pd.concat([ft_big, tm_fa, tm_rest])


def e1_block(fx_small, fx_big):
    res = {}
    for name, excl in (("all_21", ()), ("no_fltrust_aligned_18", (FA,))):
        r, per_seed = inf.e1_root_dependence(fx_small, fx_big, exclude=excl)
        res[name] = {**r, "D_s": per_seed.round(5).to_dict()}
        for root, fx in (("r100", fx_small), ("r2500", fx_big)):
            m = inf.cube(fx[~fx.attack_type.isin(excl)])
            v = (m["fltrust"] - m["trimmed_mean"]).groupby(level="seed").mean()
            res[name][f"V_{root}"] = inf.describe(v.values)
    for root, fx in (("r100", fx_small), ("r2500", fx_big)):
        m = inf.cube(fx).groupby(level=inf.CELL).mean()
        res[f"fltrust_wins_cells_{root}"] = int((m["fltrust"] > m["trimmed_mean"]).sum())
    return res


def e3_rows(P, fam, systems, ref_of):
    rows = []
    for s in systems:
        ref_name, ref = ref_of(s)
        d = (P[s] - ref).dropna()
        rows.append({"family": fam, "system": s, "reference": ref_name,
                     **inf.describe(d.groupby(level="seed").mean().values)})
    return rows


def finish(rows):
    t = inf.add_holm(pd.DataFrame(rows))
    return inf.add_holm(t, col="p_t")


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()
    res = {"checks": check(df)}
    df.to_csv(f"{OUT}/all_runs.csv", index=False)

    # ---- invariance of the trimmed mean between roots (first output of E1)
    tm100 = fixed(df, "e1_main", ["trimmed_mean_r100"], {"trimmed_mean_r100": "trimmed_mean"})
    tm2500 = fixed(df, "e1_invariant", ["trimmed_mean_r2500"], {"trimmed_mean_r2500": "trimmed_mean"})
    res["TM_invariance_outside_FA"] = inf.tm_invariance(tm100[tm100.attack_type != FA], tm2500[tm2500.attack_type != FA])
    res["TM_root_difference_FA"] = inf.tm_invariance(tm100[tm100.attack_type == FA], tm2500[tm2500.attack_type == FA])

    # ---- E1
    s3, b3 = e1_frames(df, "e1_main", "e1_invariant")
    res["E1_factor_third"] = e1_block(s3, b3)
    s1, b1 = e1_frames(df, "e1_sens_factor", "e1_sens_factor")
    res["E1_factor_one"] = e1_block(s1, b1)
    # server-steps factorial at factor 1: FLTrust native vs parity, each root
    nat = df[df.block == "e1_sens_steps"]
    par = df[(df.block == "e1_sens_factor") & df.arm.str.startswith("fltrust")]
    tm1 = df[(df.block == "e1_sens_factor") & (df.arm == "trimmed_mean_r100")]
    steps = {}
    for root in ("r100", "r2500"):
        a = nat[nat.arm == f"fltrust_{root}"].set_index(inf.CELL + ["seed"]).accuracy
        b = par[par.arm == f"fltrust_{root}"].set_index(inf.CELL + ["seed"]).accuracy
        t = tm1.set_index(inf.CELL + ["seed"]).accuracy
        steps[root] = {"fltrust_native_mean": float(a.mean()), "fltrust_parity_mean": float(b.mean()),
                       "native_minus_parity": inf.describe((a - b).groupby(level="seed").mean().values),
                       "native_minus_TM100": inf.describe((a - t).dropna().groupby(level="seed").mean().values)}
    nat_small = nat[nat.arm == "fltrust_r100"].assign(strategy="fltrust")[s1.columns]
    nat_big = nat[nat.arm == "fltrust_r2500"].assign(strategy="fltrust")[s1.columns]
    steps["D_native"] = e1_block(pd.concat([nat_small, s1[s1.strategy == "trimmed_mean"]]),
                                 pd.concat([nat_big, b1[b1.strategy == "trimmed_mean"]]))
    res["E1_server_steps_factor_one"] = steps

    # ---- E2 (root 100, factor 1/3, D_R)
    fx100 = fixed(df, "e1_main", [f"{r}_r100" for r in cj.RULES], {f"{r}_r100": r for r in cj.RULES})
    for name, fx in (("all_21", fx100), ("no_fltrust_aligned_18", fx100[fx100.attack_type != FA])):
        res[f"E2_{name}"] = {**inf.e2_headroom(fx), **inf.e2_selection_stability(fx)}
    means = fx100.groupby(["alpha", "attack_type", "strategy"]).accuracy.mean().unstack()
    means.to_csv(f"{OUT}/e1_main_cell_means_r100.csv")

    # ---- E3 (12 cells)
    e3c = set(cj.E3_CELLS)
    in_e3 = lambda x: x[[(a, t) in e3c for a, t in zip(x.alpha, x.attack_type)]]  # noqa: E731
    fx12 = in_e3(fx100)
    hx12 = fixed(df, "e3_hardened", [f"{r}_hardened" for r in inf.D_R], {f"{r}_hardened": r for r in inf.D_R})
    sy = df[df.block.isin(["e3_main", "e3_sens"])].copy()
    sy["system"] = sy.arm.map(SYSTEMS)
    P = sy.pivot_table(index=inf.CELL + ["seed"], columns="system", values="accuracy")
    B_plain, O_plain = inf.crossfit_references(fx12)
    B_hard, O_hard = inf.crossfit_references(hx12)
    main = ["Attack-aware", "GRADF", "FedStrategist", "AdaAggRL"]
    conf = e3_rows(P, "vs Random", main, lambda s: ("Random", P["Random"]))
    conf += e3_rows(P, "vs B_-s", ["Random"] + main,
                    lambda s: ("hardened", B_hard) if s in ("Random", "GRADF") else ("plain", B_plain))
    t_conf = finish(conf)
    desc = e3_rows(P, "vs B_-s plain (descriptive)", ["Random", "GRADF"], lambda s: ("plain", B_plain))
    desc += e3_rows(P, "vs B_-s hardened (descriptive)", ["Attack-aware"], lambda s: ("hardened", B_hard))
    desc += e3_rows(P, "hardened B_-s vs plain B_-s", ["Random"], lambda s: ("plain", B_plain))
    desc[-1]["system"] = "hardened B_-s"
    d = (B_hard - B_plain).groupby(level="seed").mean().values
    desc[-1].update(inf.describe(d))
    t_desc = finish(desc)
    sens = e3_rows(P, "sensitivity vs Random", ["AdaAggRL (pretrained)", "FedStrategist (GRADF detector)"],
                   lambda s: ("Random", P["Random"]))
    sens += e3_rows(P, "sensitivity vs B_-s", ["AdaAggRL (pretrained)", "FedStrategist (GRADF detector)"],
                    lambda s: ("plain", B_plain))
    t_sens = finish(sens)
    for name, t in (("e3_confirmatory", t_conf), ("e3_descriptive", t_desc), ("e3_sensitivity", t_sens)):
        t.to_csv(f"{OUT}/{name}.csv", index=False)

    Hc_plain = (O_plain - B_plain).groupby(level=inf.CELL).mean()
    Hc_hard = (O_hard - B_hard).groupby(level=inf.CELL).mean()
    cap = {}
    for s in P.columns:
        Bref, Hc = (B_hard, Hc_hard) if s in LAYERED else (B_plain, Hc_plain)
        gain = (P[s] - Bref).groupby(level=inf.CELL).mean()
        cap[s] = {"cells_beating_B": int((gain > 0).sum()), "captured_in_headroom_cells": int(((gain > 0) & (Hc > 0)).sum())}
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
        "hardened_layer4_reject_rate": df[df.block == "e3_hardened"].groupby("arm").l4_reject.mean().to_dict(),
    }
    cell_tab = P.groupby(level=inf.CELL).mean().join(fx12.groupby(["alpha", "attack_type", "strategy"]).accuracy.mean().unstack()[inf.D_R])
    cell_tab.join(hx12.groupby(["alpha", "attack_type", "strategy"]).accuracy.mean().unstack()[inf.D_R], rsuffix="_hardened").to_csv(f"{OUT}/e3_cell_means.csv")

    # ---- machine, descriptive: deterministic arms on several machines
    res["machine_share_by_block"] = {b: Counter(g.machine) for b, g in df.groupby("block")}

    with open(f"{OUT}/analysis.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else (dict(o) if isinstance(o, Counter) else str(o)))

    short = lambda r: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()  # noqa: E731
                       if k not in ("D_s", "H_loso_per_seed", "sd")}
    print("checks", res["checks"])
    print("TM invariance outside FA", res["TM_invariance_outside_FA"], "| FA", res["TM_root_difference_FA"])
    for k in ("E1_factor_third", "E1_factor_one"):
        for g in ("all_21", "no_fltrust_aligned_18"):
            print(k, g, "D:", short(res[k][g]))
            print("   V_r100", short(res[k][g]["V_r100"]), "\n   V_r2500", short(res[k][g]["V_r2500"]))
        print(k, "FLTrust wins over TM:", res[k]["fltrust_wins_cells_r100"], "/21 at r100,", res[k]["fltrust_wins_cells_r2500"], "/21 at r2500")
    for root in ("r100", "r2500"):
        s = steps[root]
        print("steps", root, round(s["fltrust_native_mean"], 4), round(s["fltrust_parity_mean"], 4), short(s["native_minus_parity"]))
    print("steps D_native", short(steps["D_native"]["all_21"]))
    for g in ("all_21", "no_fltrust_aligned_18"):
        print("E2", g, short(res[f"E2_{g}"]))
    pd.set_option("display.width", 250)
    print("\nE3 confirmatory\n", t_conf[COLS].round(4).to_string(index=False))
    print("\nE3 descriptive\n", t_desc[COLS].round(4).to_string(index=False))
    print("\nE3 sensitivity\n", t_sens[COLS].round(4).to_string(index=False))
    print("\n", json.dumps(res["E3_descriptive"], indent=1, default=float))


if __name__ == "__main__":
    main()
