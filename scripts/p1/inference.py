"""
P1 confirmatory analyses (plan docs/plano_p1_completo.md, Secs. 4.3-4.5 and
E1-E3), written once and applied to MNIST now and to CIFAR-10 later.

Inputs are long DataFrames with columns alpha, attack_type, seed, accuracy and
`strategy` (fixed rules) or `system` (adaptive systems). The unit of analysis
is the seed: every aggregate hypothesis reduces each seed to one number (the
mean over the cells of the grid in use) before testing.

  E1  V(c,s,r) = Acc_FLTrust(c,s,r) - Acc_TM(c,s);
      D_s = mean_c [V(c,s,r_big) - V(c,s,r_small)]; paired test over seeds.
  E2  cross-fitted (leave-one-seed-out) headroom H_LOSO, with a seed-level
      bootstrap that repeats the whole selection + evaluation in each resample;
      the in-sample headroom is reported as an optimistic upper bound.
  E3  A_sys(c,s) - B_{-s}(c,s) and A_sys(c,s) - A_Random(c,s), aggregated over
      cells per seed; Holm within each family.

Every comparison reports mean, 95% CI, t-test, Wilcoxon, Cohen's d_z, TOST at
the 1.0 p.p. margin, and one of four outcome categories (superiority,
inferiority, equivalence, no evidence).
"""

import numpy as np
import pandas as pd
from scipy import stats

D_R = ["fltrust", "median", "trimmed_mean", "krum"]
CELL = ["alpha", "attack_type"]
MARGIN = 0.010  # equivalence margin (1.0 p.p.), frozen in the plan


# ---------------------------------------------------------------- tests

def tost_p(x, margin=MARGIN):
    """Two one-sided t-tests for |mean| < margin; returns the larger p."""
    x = np.asarray(x, dtype=float)
    lo = stats.ttest_1samp(x, -margin, alternative="greater").pvalue
    hi = stats.ttest_1samp(x, margin, alternative="less").pvalue
    return max(lo, hi)


def describe(x, margin=MARGIN):
    """Full report for per-seed values x (one value per seed)."""
    x = np.asarray(x, dtype=float)
    n, m, sd = len(x), x.mean(), x.std(ddof=1)
    se = sd / np.sqrt(n)
    q95, q90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    ci95 = (m - q95 * se, m + q95 * se)
    ci90 = (m - q90 * se, m + q90 * se)
    wil = stats.wilcoxon(x).pvalue if np.any(x != 0) else 1.0
    if ci95[0] > 0:
        cat = "superiority"
    elif ci95[1] < 0:
        cat = "inferiority"
    elif -margin < ci90[0] and ci90[1] < margin:
        cat = "equivalence"
    else:
        cat = "no evidence"
    return {"n": n, "mean": m, "sd": sd, "ci95_lo": ci95[0], "ci95_hi": ci95[1],
            "p_wilcoxon": wil, "p_t": stats.ttest_1samp(x, 0.0).pvalue,
            "d_z": m / sd if sd > 0 else np.nan, "p_tost": tost_p(x, margin),
            "category": cat}


def holm(pvals):
    """Holm-adjusted p-values, same order as the input."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def add_holm(df, col="p_wilcoxon", family="family"):
    df = df.copy()
    df[f"{col}_holm"] = np.nan
    for _, idx in df.groupby(family).groups.items():
        df.loc[idx, f"{col}_holm"] = holm(df.loc[idx, col].values)
    return df


# ---------------------------------------------------------------- reshaping

def cube(fx):
    """(cell, seed) x rule matrix of fixed-rule accuracies."""
    return fx.pivot_table(index=CELL + ["seed"], columns="strategy", values="accuracy")


def as_array(fx, rules=D_R):
    """3-D array [cell, seed, rule] plus the cell index and seed list."""
    m = cube(fx)[rules]
    cells = m.index.droplevel("seed").unique()
    seeds = sorted(m.index.get_level_values("seed").unique())
    m = m.reindex(pd.MultiIndex.from_tuples(
        [(*c, s) for c in cells for s in seeds], names=CELL + ["seed"]))
    if m.isna().any().any():
        raise ValueError("fixed-rule grid is incomplete")
    return m.values.reshape(len(cells), len(seeds), len(rules)), cells, seeds


# ---------------------------------------------------------------- E1

def e1_root_dependence(fx_small, fx_big, exclude=()):
    """D_s = mean_c [V(c,s,big) - V(c,s,small)], V = FLTrust - TM."""
    def v(fx):
        m = cube(fx[~fx.attack_type.isin(exclude)])
        return (m["fltrust"] - m["trimmed_mean"]).rename("V")
    d = (v(fx_big) - v(fx_small)).dropna()
    per_seed = d.groupby(level="seed").mean()
    return describe(per_seed.values), per_seed


def tm_invariance(fx_small, fx_big):
    """Per (cell, seed) differences of the trimmed mean between two roots."""
    a = cube(fx_small)["trimmed_mean"]
    b = cube(fx_big)["trimmed_mean"]
    diff = (a - b).dropna()
    return {"n_pairs": int(len(diff)), "max_abs": float(diff.abs().max()),
            "n_nonzero": int((diff != 0).sum())}


# ---------------------------------------------------------------- E2

def loso_headroom(acc, seed_idx=None):
    """Cross-fitted headroom. acc: [cell, seed, rule]. `seed_idx` lists the
    seed columns in use (with repeats, for the bootstrap). For each position i
    the selection uses every position holding a DIFFERENT seed (copies of the
    held-out seed are left out too, so nothing leaks from evaluation into
    selection), and the evaluation uses i.
    Returns (H_LOSO, per-position H_s, in-sample H)."""
    idx = np.arange(acc.shape[1]) if seed_idx is None else np.asarray(seed_idx)
    sub = acc[:, idx, :]                         # [cell, k, rule]
    k = sub.shape[1]
    h_s = np.full(k, np.nan)
    rows = np.arange(sub.shape[0])
    for i in range(k):
        keep = idx != idx[i]
        if not keep.any():
            continue
        sel = sub[:, keep, :].mean(axis=1)       # [cell, rule]
        r_cell = sel.argmax(axis=1)
        d_glob = sel.mean(axis=0).argmax()
        h_s[i] = np.mean(sub[rows, i, r_cell] - sub[:, i, d_glob])
    mean = sub.mean(axis=1)
    h_in = np.mean(mean.max(axis=1) - mean[:, mean.mean(axis=0).argmax()])
    return np.nanmean(h_s), h_s, h_in


def e2_headroom(fx, n_boot=20000, seed=0):
    acc, cells, seeds = as_array(fx)
    h, h_s, h_in = loso_headroom(acc)
    rng = np.random.default_rng(seed)
    n = len(seeds)
    boot_loso, boot_in = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 2:
            idx[0] = (idx[0] + 1) % n
        boot_loso[b], _, boot_in[b] = loso_headroom(acc, idx)
    ci = np.percentile(boot_loso, [2.5, 97.5])
    ci_in = np.percentile(boot_in, [2.5, 97.5])
    return {
        "n_cells": len(cells), "n_seeds": n,
        "H_insample": h_in, "H_insample_ci95": ci_in.tolist(),
        "H_loso": h, "H_loso_ci95": ci.tolist(),
        "H_loso_per_seed": dict(zip(seeds, h_s.round(5).tolist())),
        "category": "positive" if ci[0] > 0 else ("negative" if ci[1] < 0 else "no evidence"),
    }


def e2_selection_stability(fx):
    """How often the LOSO-selected cell winner and d* agree with the
    in-sample choice (descriptive)."""
    acc, cells, seeds = as_array(fx)
    mean = acc.mean(axis=1)
    r_in, d_in = mean.argmax(axis=1), mean.mean(axis=0).argmax()
    agree_cell, agree_d = [], []
    for i in range(acc.shape[1]):
        sel = (acc.sum(axis=1) - acc[:, i, :]) / (acc.shape[1] - 1)
        agree_cell.append(np.mean(sel.argmax(axis=1) == r_in))
        agree_d.append(sel.mean(axis=0).argmax() == d_in)
    return {"dstar_insample": D_R[d_in], "dstar_agreement": float(np.mean(agree_d)),
            "cell_winner_agreement": float(np.mean(agree_cell))}


# ---------------------------------------------------------------- E3

def crossfit_references(fx):
    """B_{-s}(c,s) and O_{-s}(c,s) as Series indexed by (cell, seed)."""
    acc, cells, seeds = as_array(fx)
    k = len(seeds)
    total = acc.sum(axis=1)
    rows = np.arange(len(cells))
    B, O = {}, {}
    for i, s in enumerate(seeds):
        sel = (total - acc[:, i, :]) / (k - 1)
        r_cell, d_glob = sel.argmax(axis=1), sel.mean(axis=0).argmax()
        for j, c in enumerate(cells):
            B[(*c, s)] = acc[j, i, d_glob]
            O[(*c, s)] = acc[j, i, r_cell[j]]
    idx = pd.MultiIndex.from_tuples(list(B), names=CELL + ["seed"])
    return pd.Series(list(B.values()), index=idx), pd.Series(list(O.values()), index=idx)


def e3_capture(fx, sy, systems, random_name="Random"):
    """Confirmatory E3 families: each system vs Random, and each system vs
    the cross-fitted global best fixed rule B_{-s}. Plus descriptive per-cell
    capture counts with cross-fitted references."""
    B, O = crossfit_references(fx)
    P = sy.pivot_table(index=CELL + ["seed"], columns="system", values="accuracy")
    rows = []
    for s in systems:
        refs = [("vs B_-s (cross-fitted)", B)]
        if s != random_name:
            refs.insert(0, ("vs Random", P[random_name]))
        for fam, ref in refs:
            d = (P[s] - ref).dropna()
            r = describe(d.groupby(level="seed").mean().values)
            rows.append({"family": fam, "system": s, **r})
    res = add_holm(pd.DataFrame(rows))
    res = add_holm(res, col="p_t")
    # descriptive: cells where the seed-mean system beats the seed-mean B_-s
    # among cells whose cross-fitted headroom O_-s - B_-s is positive on average
    Hc = (O - B).groupby(level=CELL).mean()
    cap = {}
    for s in systems:
        gain = (P[s] - B).groupby(level=CELL).mean()
        cap[s] = int(((gain > 0) & (Hc > 0)).sum())
    return res, {"cells_with_crossfit_headroom": int((Hc > 0).sum()), "captured": cap}
