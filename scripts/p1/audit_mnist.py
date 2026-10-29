"""
P-1 audit of the P1 manuscript (FGCS submission, MNIST): recomputes every
reported number from the raw per-seed CSVs and compares it with the value
printed in the manuscript.

Pipeline: raw CSVs (results/tables/) -> this script -> results/p1_audit/
(regenerated tables as CSV + a claim-by-claim check). Nothing is typed in by
hand except the manuscript values being checked (MANUSCRIPT_CLAIMS), each
tagged with its line in docs/P1/fgcs_src/elsarticle-template-num-names.tex.

Raw inputs:
  exp9_dominance_grid_10seeds_ALL_root100_raw.csv   D_R rules, root 100, 10 seeds
  exp9_dominance_grid_10seeds_root{3000,50}_raw.csv FLTrust 10 seeds; others 3 seeds
  exp9_nonfltrust_extraseeds_raw.csv                median/TM/Krum, seeds 45-51
  exp10_FULL_variantb_raw.csv                       Random, attack-aware, GRADF,
                                                    FedStrategist, AdaAggRL (root 100)
  exp10_FULL_varianta_fedstrategist_raw.csv         FedStrategist with GRADF's detector
  exp10_passo_zero_combined_raw.csv                 AdaAggRL with pretrained extractor

Usage: venv/bin/python scripts/p1/audit_mnist.py
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

TAB = "results/tables"
OUT = "results/p1_audit"
D_R = ["fltrust", "median", "trimmed_mean", "krum"]
ALPHAS = [0.5, 0.1, 0.05]
ATTACKS = ["fltrust_aligned", "gaussian_noise", "krum_collusion", "label_flipping",
           "low_mag_backdoor", "sign_flipping", "trim_attack"]
SYSTEMS = {"Random": "Random", "Attack-aware": "Oracle (decoupled)", "GRADF": "GRADF",
           "FedStrat.": "FedStrategist", "AdaAggRL": "AdaAggRL"}
CELL = ["alpha", "attack_type"]


def read(name):
    return pd.read_csv(os.path.join(TAB, name))


# ---------------------------------------------------------------- inputs

def load_fixed():
    """Returns {root: long DataFrame of D_R rules}. Two versions of the
    enlarged/reduced roots are kept: 'as_published' (the 10seeds_root* files,
    whose median/TM/Krum rows only cover seeds 42-44) and 'ten_seeds'
    (median/TM/Krum completed with seeds 45-51, which were run once, at the
    full 3,000-sample server set, and reused for every root size)."""
    r100 = read("exp9_dominance_grid_10seeds_ALL_root100_raw.csv")
    extra = read("exp9_nonfltrust_extraseeds_raw.csv")
    fixed = {"100": r100}
    for r in ("3000", "50"):
        pub = read(f"exp9_dominance_grid_10seeds_root{r}_raw.csv")
        fixed[f"{r}_as_published"] = pub
        fixed[f"{r}_ten_seeds"] = pd.concat([pub, extra], ignore_index=True)
    return fixed


def load_systems():
    sy = read("exp10_FULL_variantb_raw.csv")
    pz = read("exp10_passo_zero_combined_raw.csv")
    fsa = read("exp10_FULL_varianta_fedstrategist_raw.csv")
    pre = pz[pz.system == "AdaAggRL (pretrained)"]
    fsa = fsa.assign(system="FedStrategist (GRADF detector)")
    cols = CELL + ["system", "seed", "accuracy"]
    return pd.concat([sy[cols], pre[cols], fsa[cols]], ignore_index=True)


def cell_means(df, key):
    return df.groupby(CELL + [key]).accuracy.mean().unstack()


def cell_stds(df, key):
    return df.groupby(CELL + [key]).accuracy.std().unstack()


def per_seed(df, key):
    """(cell, seed) x key matrix of accuracies."""
    return df.pivot_table(index=CELL + ["seed"], columns=key, values="accuracy")


# ---------------------------------------------------------------- statistics

def paired(diff_by_seed):
    """Seed-paired t-test on per-seed differences; d is Cohen's d_z."""
    x = np.asarray(diff_by_seed, dtype=float)
    t = stats.ttest_1samp(x, 0.0)
    return {"delta": x.mean(), "p": t.pvalue, "d": x.mean() / x.std(ddof=1), "n": len(x)}


# ---------------------------------------------------------------- RQ1

def winners(fx):
    m = cell_means(fx[fx.strategy.isin(D_R)], "strategy")[D_R]
    s = cell_stds(fx[fx.strategy.isin(D_R)], "strategy")[D_R]
    rows = []
    for c in m.index:
        ranked = m.loc[c].sort_values(ascending=False)
        best, second = ranked.index[0], ranked.index[1]
        gap = ranked.iloc[0] - ranked.iloc[1]
        rows.append({"alpha": c[0], "attack_type": c[1], "winner": best, "runner_up": second,
                     "gap": gap, "sum_std": s.loc[c, best] + s.loc[c, second]})
    return pd.DataFrame(rows)


def table1(fixed):
    out = []
    for root, fx in fixed.items():
        w = winners(fx)
        row = {"root": root}
        for r in D_R:
            row[r] = int((w.winner == r).sum())
        robust = (w.winner != "fltrust") & (w.gap > w.sum_std)
        row["non_fltrust_robust"] = int(robust.sum())
        row["non_fltrust_any"] = int((w.winner != "fltrust").sum())
        out.append(row)
    return pd.DataFrame(out)


def table2(fixed, roots=("3000_as_published", "100")):
    rows = []
    for a in ALPHAS:
        row = {"alpha": a}
        for r in roots:
            w = winners(fixed[r])
            row[r] = int(((w.alpha == a) & (w.winner == "fltrust")).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- RQ2/RQ3

def headroom(fx100, sy, dstar=None, drop_fa=False):
    fx, s = fx100, sy
    if drop_fa:
        fx, s = fx[fx.attack_type != "fltrust_aligned"], s[s.attack_type != "fltrust_aligned"]
    A = cell_means(fx, "strategy")[D_R]
    S = cell_means(s, "system")
    glob = fx.groupby("strategy").accuracy.mean()[D_R]
    dstar = dstar or glob.idxmax()
    O, B = A.max(axis=1), A[dstar]
    H = O > B
    cap = {k: ((S[v] > B) & H) for k, v in SYSTEMS.items()}
    for v in ("AdaAggRL (pretrained)",):
        cap[v] = (S[v] > B) & H
    disc = cap["GRADF"] | cap["FedStrat."]
    deploy = disc | cap["AdaAggRL"]
    return {
        "dstar": dstar, "global_means": glob.to_dict(), "n_cells": int(len(A)),
        "headroom_cells": int(H.sum()),
        "captured": {k: int(v.sum()) for k, v in cap.items()},
        "discrete_any": int(disc.sum()), "deployable_any": int(deploy.sum()),
        "discrete_subset_of_adaaggrl": bool((disc & ~cap["AdaAggRL"]).sum() == 0),
        "H_global": float((O - B).mean()),
        "R": {k: ((S[v] - B) / (O - B)).where(H) for k, v in SYSTEMS.items()},
        "O_minus_B": (O - B).where(H),
    }


def legacy_capture(fx100, sy):
    """The pre-revision criterion still used in Sec. 6.3: A_s(c) - O(c) larger
    than the SUM of the two across-seed standard deviations."""
    A = cell_means(fx100, "strategy")[D_R]
    As = cell_stds(fx100, "strategy")[D_R]
    O, Or = A.max(axis=1), A.idxmax(axis=1)
    Ostd = pd.Series([As.loc[c, Or[c]] for c in A.index], index=A.index)
    S, Ss = cell_means(sy, "system"), cell_stds(sy, "system")
    return {s: int(((S[s] - O) > (Ss[s] + Ostd)).sum()) for s in S.columns}


def table3(fx100, sy):
    A = cell_means(fx100, "strategy")[D_R]
    S = cell_means(sy, "system")
    t = pd.DataFrame({k: S[v] for k, v in SYSTEMS.items()})
    t["best_fixed"] = A.max(axis=1)
    t["rule"] = A.idxmax(axis=1)
    for k in ("GRADF", "FedStrat.", "AdaAggRL"):
        t[f"bold_{k}"] = t[k] > t["best_fixed"]
    return t.reindex(pd.MultiIndex.from_product([ALPHAS, ATTACKS], names=CELL))


def vs_reference(sy, ref, by=None):
    """Per seed, mean over cells (optionally within groups `by`) of
    A_sys - ref, then the seed-paired test. `ref` is a (cell, seed) Series."""
    P = per_seed(sy, "system")
    out = {}
    for name in P.columns:
        d = (P[name] - ref).dropna().rename("d").reset_index()
        if by is None:
            out[name] = paired(d.groupby("seed").d.mean())
        else:
            out[name] = {g: paired(x.groupby("seed").d.mean()) for g, x in d.groupby(by)}
    return out


# ---------------------------------------------------------------- claims

# (id, tex line, description, printed value, decimals used in print)
MANUSCRIPT_CLAIMS = [
    ("T1_3000_fltrust", 292, "Tab.1 root 3000 FLTrust wins", 18, 0),
    ("T1_3000_median", 292, "Tab.1 root 3000 median wins", 1, 0),
    ("T1_3000_tm", 292, "Tab.1 root 3000 trimmed-mean wins", 1, 0),
    ("T1_3000_krum", 292, "Tab.1 root 3000 Krum wins", 1, 0),
    ("T1_3000_robust", 292, "Tab.1 root 3000 non-FLTrust (robust)", 2, 0),
    ("T1_100_fltrust", 293, "Tab.1 root 100 FLTrust wins", 8, 0),
    ("T1_100_median", 293, "Tab.1 root 100 median wins", 1, 0),
    ("T1_100_tm", 293, "Tab.1 root 100 trimmed-mean wins", 8, 0),
    ("T1_100_krum", 293, "Tab.1 root 100 Krum wins", 4, 0),
    ("T1_100_robust", 293, "Tab.1 root 100 non-FLTrust (robust)", 9, 0),
    ("T1_50_fltrust", 294, "Tab.1 root 50 FLTrust wins", 6, 0),
    ("T1_50_median", 294, "Tab.1 root 50 median wins", 1, 0),
    ("T1_50_tm", 294, "Tab.1 root 50 trimmed-mean wins", 10, 0),
    ("T1_50_krum", 294, "Tab.1 root 50 Krum wins", 4, 0),
    ("T1_50_robust", 294, "Tab.1 root 50 non-FLTrust (robust)", 11, 0),
    ("T2_0.5_3000", 316, "Tab.2 FLTrust wins a=0.5 root 3000", 5, 0),
    ("T2_0.5_100", 316, "Tab.2 FLTrust wins a=0.5 root 100", 0, 0),
    ("T2_0.1_3000", 317, "Tab.2 FLTrust wins a=0.1 root 3000", 7, 0),
    ("T2_0.1_100", 317, "Tab.2 FLTrust wins a=0.1 root 100", 4, 0),
    ("T2_0.05_3000", 318, "Tab.2 FLTrust wins a=0.05 root 3000", 6, 0),
    ("T2_0.05_100", 318, "Tab.2 FLTrust wins a=0.05 root 100", 4, 0),
    ("fltrust_sd_ratio", 323, "FLTrust mean SD grows ~fivefold (3000 -> 100)", 5, 0),
    ("dstar_tm_mean", 329, "global mean trimmed mean", 0.7099, 4),
    ("dstar_fltrust_mean", 329, "global mean FLTrust", 0.7095, 4),
    ("alt_dstar_headroom", 329, "d*=FLTrust: headroom cells", 13, 0),
    ("alt_dstar_captured", 329, "d*=FLTrust: captured (deployable)", 13, 0),
    ("alt_dstar_discrete", 329, "d*=FLTrust: discrete selection", 12, 0),
    ("alt_dstar_continuous", 329, "d*=FLTrust: continuous weighting", 11, 0),
    ("headroom_21", 331, "headroom cells of 21", 13, 0),
    ("headroom_18", 331, "headroom cells of 18 (no FLTrust aligned)", 10, 0),
    ("captured_21", 369, "deployable captures of 21", 10, 0),
    ("adaaggrl_21", 369, "AdaAggRL captures of 21", 10, 0),
    ("gradf_21", 369, "GRADF captures of 21 ('7 cells each')", 7, 0),
    ("fedstrat_21", 369, "FedStrategist captures of 21 ('7 cells each')", 7, 0),
    ("adaaggrl_18", 369, "AdaAggRL captures of 18", 10, 0),
    ("discrete_18", 369, "discrete captures of 18", 7, 0),
    ("T4_ada_delta", 394, "Tab.4 AdaAggRL delta", 0.030, 3),
    ("T4_ada_p", 394, "Tab.4 AdaAggRL p", 0.0065, 4),
    ("T4_ada_d", 394, "Tab.4 AdaAggRL d", 1.11, 2),
    ("T4_aw_delta", 395, "Tab.4 attack-aware delta", 0.012, 3),
    ("T4_aw_p", 395, "Tab.4 attack-aware p", 0.0212, 4),
    ("T4_aw_d", 395, "Tab.4 attack-aware d", 0.88, 2),
    ("T4_gradf_delta", 396, "Tab.4 GRADF delta", -0.042, 3),
    ("T4_gradf_p", 396, "Tab.4 GRADF p", 0.0031, 4),
    ("T4_gradf_d", 396, "Tab.4 GRADF d", -1.27, 2),
    ("T4_fs_delta", 397, "Tab.4 FedStrat. delta", -0.044, 3),
    ("T4_fs_p", 397, "Tab.4 FedStrat. p", 3.7e-5, 6),
    ("T4_fs_d", 397, "Tab.4 FedStrat. d", -2.37, 2),
    ("extractor_delta", 236, "random - pretrained extractor delta", 0.019, 3),
    ("extractor_p", 236, "random vs pretrained extractor p", 0.007, 3),
    ("extractor_d", 236, "random vs pretrained extractor d", 1.11, 2),
    ("pretrained_vs_random_delta", 437, "pretrained AdaAggRL - Random delta", 0.012, 3),
    ("pretrained_vs_random_p", 437, "pretrained AdaAggRL vs Random p", 0.24, 2),
    ("legacy_floor", 437, "'from nine' (AdaAggRL, random extractor)", 9, 0),
    ("legacy_pretrained", 437, "'to seven' (AdaAggRL, pretrained)", 7, 0),
    ("legacy_gradf", 437, "'none' (GRADF)", 0, 0),
    ("legacy_fedstrat", 437, "'two' (FedStrategist)", 2, 0),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    fixed = load_fixed()
    sy = load_systems()
    fx100 = fixed["100"]
    got = {}

    # --- Table 1 (both seed compositions) and Table 2
    t1 = table1(fixed)
    t1.to_csv(f"{OUT}/table1_winners_by_root.csv", index=False)
    pub = t1.set_index("root")
    for r, key in (("3000", "3000_as_published"), ("100", "100"), ("50", "50_as_published")):
        for col, short in (("fltrust", "fltrust"), ("median", "median"), ("trimmed_mean", "tm"),
                           ("krum", "krum"), ("non_fltrust_robust", "robust")):
            got[f"T1_{r}_{short}"] = pub.loc[key, col]
    t2 = table2(fixed)
    t2.to_csv(f"{OUT}/table2_fltrust_wins_by_alpha.csv", index=False)
    for _, row in t2.iterrows():
        got[f"T2_{row.alpha}_3000"] = row["3000_as_published"]
        got[f"T2_{row.alpha}_100"] = row["100"]
    sd3000 = cell_stds(fixed["3000_as_published"].query("strategy=='fltrust'"), "strategy")["fltrust"].mean()
    sd100 = cell_stds(fx100.query("strategy=='fltrust'"), "strategy")["fltrust"].mean()
    got["fltrust_sd_ratio"] = sd100 / sd3000

    # --- headroom and capture counts
    h = headroom(fx100, sy)
    # The manuscript's 18-cell counts keep d* from the 21-cell grid; recomputing
    # d* on the 18 cells flips it to FLTrust (reported in extra_numbers.json).
    h18 = headroom(fx100, sy, dstar=h["dstar"], drop_fa=True)
    h18_own = headroom(fx100, sy, drop_fa=True)
    hf = headroom(fx100, sy, dstar="fltrust")
    got.update({
        "dstar_tm_mean": h["global_means"]["trimmed_mean"],
        "dstar_fltrust_mean": h["global_means"]["fltrust"],
        "alt_dstar_headroom": hf["headroom_cells"], "alt_dstar_captured": hf["deployable_any"],
        "alt_dstar_discrete": hf["discrete_any"], "alt_dstar_continuous": hf["captured"]["AdaAggRL"],
        "headroom_21": h["headroom_cells"], "headroom_18": h18["headroom_cells"],
        "captured_21": h["deployable_any"], "adaaggrl_21": h["captured"]["AdaAggRL"],
        "gradf_21": h["captured"]["GRADF"], "fedstrat_21": h["captured"]["FedStrat."],
        "adaaggrl_18": h18["captured"]["AdaAggRL"], "discrete_18": h18["discrete_any"],
    })
    leg = legacy_capture(fx100, sy)
    got.update({"legacy_floor": leg["AdaAggRL"], "legacy_pretrained": leg["AdaAggRL (pretrained)"],
                "legacy_gradf": leg["GRADF"], "legacy_fedstrat": leg["FedStrategist"]})

    # --- Table 3
    t3 = table3(fx100, sy)
    t3.to_csv(f"{OUT}/table3_grid_root100.csv")

    # --- Tables 4/5: vs Random, and vs the global best fixed rule B(c)
    P = per_seed(sy, "system")
    rnd = P["Random"]
    t4 = vs_reference(sy, rnd)
    t5 = vs_reference(sy, rnd, by="attack_type")
    F = per_seed(fx100, "strategy")
    t4b = vs_reference(sy, F[h["dstar"]])
    t4o = vs_reference(sy, F[D_R].max(axis=1))  # per-seed cell max: not B(c), kept for reference
    for k, v in SYSTEMS.items():
        if k == "Random":
            continue
        tag = {"AdaAggRL": "ada", "Attack-aware": "aw", "GRADF": "gradf", "FedStrat.": "fs"}[k]
        got[f"T4_{tag}_delta"], got[f"T4_{tag}_p"], got[f"T4_{tag}_d"] = (
            t4[v]["delta"], t4[v]["p"], t4[v]["d"])
    pre = P["AdaAggRL (pretrained)"]
    ext = paired((P["AdaAggRL"] - pre).groupby(level="seed").mean())
    got["extractor_delta"], got["extractor_p"], got["extractor_d"] = ext["delta"], ext["p"], ext["d"]
    got["pretrained_vs_random_delta"] = t4["AdaAggRL (pretrained)"]["delta"]
    got["pretrained_vs_random_p"] = t4["AdaAggRL (pretrained)"]["p"]

    rows = []
    for t, by in ((t4, "Random"), (t4b, f"B(c)=acc({h['dstar']})")):
        for s, r in t.items():
            rows.append({"reference": by, "system": s, **r})
    pd.DataFrame(rows).to_csv(f"{OUT}/table4_pooled.csv", index=False)
    rows = [{"system": s, "attack_type": a, **r} for s, g in t5.items() for a, r in g.items()]
    pd.DataFrame(rows).to_csv(f"{OUT}/table5_per_attack_vs_random.csv", index=False)

    # --- figure data
    fig3 = sy.groupby(["attack_type", "system"]).accuracy.mean().unstack()
    fig3.to_csv(f"{OUT}/fig3_mean_by_attack.csv")

    # --- claim check
    checks = []
    for cid, line, desc, printed, dec in MANUSCRIPT_CLAIMS:
        val = got[cid]
        if cid == "fltrust_sd_ratio":
            ok = 4.5 <= val < 5.5
        elif cid.endswith("_p") and printed < 1e-3:
            ok = f"{val:.1e}" == f"{printed:.1e}"
        else:
            ok = round(float(val), dec) == round(float(printed), dec)
        checks.append({"id": cid, "tex_line": line, "claim": desc, "manuscript": printed,
                       "recomputed": val, "match": ok})
    ck = pd.DataFrame(checks)
    ck.to_csv(f"{OUT}/claims_check.csv", index=False)

    extra = {
        "dstar": h["dstar"], "H_global_insample": h["H_global"],
        "discrete_subset_of_adaaggrl": h["discrete_subset_of_adaaggrl"],
        "captured_by_system_21": h["captured"], "captured_by_system_alt_dstar": hf["captured"],
        "no_fltrust_aligned_dstar_recomputed": {
            "dstar": h18_own["dstar"], "headroom": h18_own["headroom_cells"],
            "captured": h18_own["captured"], "discrete_any": h18_own["discrete_any"]},
        "legacy_criterion_counts": leg,
        "extractor_random_minus_pretrained": ext,
        "random_minus_floor_is_table4_adaaggrl": t4["AdaAggRL"],
        "vs_global_best_fixed_B": t4b, "vs_per_seed_cell_max": t4o,
        "fedstrategist_gradf_detector_vs_random": t4["FedStrategist (GRADF detector)"],
        "fltrust_mean_sd": {"3000": sd3000, "100": sd100},
        "discrete_capture_R_and_OminusB": {
            k: pd.DataFrame({"R": h["R"][k], "O_minus_B": h["O_minus_B"]})
                 [h["R"][k] > 0].round(3).reset_index().to_dict("records")
            for k in ("GRADF", "FedStrat.")},
        "mean_per_cell_sd": cell_stds(sy, "system").mean().to_dict(),
    }
    with open(f"{OUT}/extra_numbers.json", "w") as f:
        json.dump(extra, f, indent=1, default=float)

    pd.set_option("display.width", 200)
    print(ck.to_string(index=False))
    print(f"\n{int(ck.match.sum())}/{len(ck)} claims reproduced")
    print(t1.to_string(index=False))


if __name__ == "__main__":
    main()
