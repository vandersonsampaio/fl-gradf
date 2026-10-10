"""
Step 2 analysis, following results/frente1_passo2_oficial/PREREGISTRO.md §5-6.

Written before any grid result existed. Unit = (attack, seed)
pair; D = metric(cond) - metric(td3).

  Primary: mean accuracy over the last `window` rounds (default 50, rounds
           451-500), taken from the environment steps (1 step = 1 round).
  H1: fixed - td3. Paired TOST (t), margin ±1.0 p.p., alpha 0.05;
      paired Wilcoxon for a difference.
  H2: random - td3, same procedure.
  Per attack: Δ, 95% CI, d, Wilcoxon + Holm (descriptive).
  Exploratory: distance of the TD3 actions to the center (0.475) after round
  100, weight mass on real attackers, resets, sim_lc rule firings.

Usage:
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar.py
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

MARGIN = 0.010  # 1.0 p.p.
ALPHA = 0.05
CENTER = 0.475
ATTACKS = ["LMP", "EB"]
SEEDS = [100, 101, 102, 103, 104]


def load(raw_dir: str, rounds: int) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(raw_dir, f"*_R{rounds}.json"))):
        d = json.load(open(f))
        # SB3 collects in blocks of train_freq=3, so td3 runs 501 steps
        # (so does the official main.py). Truncating at `rounds` equalizes the windows
        # across conditions (addendum 1 of the PREREGISTRO).
        n_raw = len(d["steps"])
        steps = d["steps"][:rounds]
        acc = np.array([s["acc"] for s in steps])
        mass = np.array([np.nan if s["att_weight_mass"] is None else s["att_weight_mass"] for s in steps])
        real = np.array([s["n_att_real"] for s in steps])
        acts = np.array([s["action"] for s in steps])
        rows.append({
            "attack": d["attack"], "condition": d["condition"], "seed": d["seed"],
            "n_steps": len(steps), "n_steps_raw": n_raw, "acc": acc,
            "final_acc": float(acc[-1]),
            "att_mass_mean": float(np.nanmean(mass[real > 0])) if (real > 0).any() else np.nan,
            "n_resets_extra": len(d["resets"]) - 1,
            "simlc_rows_per_round": float(np.mean([s["simlc_rule_rows"] for s in steps])),
            "act_dist_center_post100": float(np.mean(np.abs(acts[100:] - CENTER))) if len(acts) > 100 else np.nan,
            "act_std_post100": float(np.mean(acts[100:].std(axis=0))) if len(acts) > 100 else np.nan,
        })
    return pd.DataFrame(rows)


def tost_paired(d: np.ndarray, margin: float):
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return (0.0, 0.0) if abs(m) < margin else (1.0, 1.0)
    p_low = 1 - stats.t.cdf((m + margin) / se, n - 1)   # H0: diff <= -margin
    p_high = stats.t.cdf((m - margin) / se, n - 1)      # H0: diff >= +margin
    return p_low, p_high


def describe(d: np.ndarray) -> dict:
    n = len(d)
    m = d.mean()
    sd = d.std(ddof=1) if n > 1 else np.nan
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    try:
        wp = stats.wilcoxon(d).pvalue if n > 1 and np.any(d != 0) else 1.0
    except ValueError:
        wp = np.nan
    return {"n": n, "delta_pp": 100 * m, "ci95_pp": (100 * (m - half), 100 * (m + half)),
            "d": m / sd if sd and sd > 0 else np.nan, "wilcoxon_p": wp}


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def verdict(desc: dict, p_tost: float) -> str:
    if p_tost < ALPHA:
        return "EQUIVALENT (TOST)"
    if desc["wilcoxon_p"] < ALPHA:
        return "td3 BETTER" if desc["delta_pp"] < 0 else "condition BETTER than td3 (do not claim 'td3 hurts')"
    return "INCONCLUSIVE"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="results/frente1_passo2_oficial/raw")
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--out", default="results/frente1_passo2_oficial/analise.txt")
    a = p.parse_args()

    df = load(a.raw_dir, a.rounds)
    df["primary"] = df["acc"].apply(lambda x: float(np.mean(x[-a.window:])))
    lines = [f"Step 2 — {len(df)} runs loaded (expected {len(ATTACKS) * 3 * len(SEEDS)})", ""]
    incomplete = df[df["n_steps"] != a.rounds]
    if len(incomplete):
        lines.append(f"WARNING: runs with number of steps != {a.rounds}:\n{incomplete[['attack','condition','seed','n_steps']]}")

    tab = df.pivot_table(index=["attack", "seed"], columns="condition", values="primary")
    lines += ["Primary metric (mean accuracy over the last %d rounds):" % a.window, tab.round(4).to_string(), ""]

    for cond, label in [("fixed", "H1: fixed − td3"), ("random", "H2: random − td3")]:
        if cond not in tab or "td3" not in tab:
            continue
        pair = tab[[cond, "td3"]].dropna()
        d = (pair[cond] - pair["td3"]).to_numpy()
        if len(d) < 2:
            continue
        desc = describe(d)
        pl, ph = tost_paired(d, MARGIN)
        p_tost = max(pl, ph)
        lines += [f"== {label} (pooled, attack×seed pairs) ==",
                  f"  n={desc['n']}  Δ={desc['delta_pp']:+.2f} p.p.  CI95=({desc['ci95_pp'][0]:+.2f}, {desc['ci95_pp'][1]:+.2f})  d={desc['d']:+.2f}",
                  f"  TOST ±{100*MARGIN:.1f} p.p.: p={p_tost:.4f}   Wilcoxon: p={desc['wilcoxon_p']:.4f}",
                  f"  VERDICT: {verdict(desc, p_tost)}"]
        per = []
        for att in ATTACKS:
            if att in pair.index.get_level_values(0):
                da = (pair.loc[att][cond] - pair.loc[att]["td3"]).to_numpy()
                if len(da) >= 2:
                    per.append((att, describe(da)))
        if per:
            adj = holm(np.array([x[1]["wilcoxon_p"] for x in per]))
            for (att, ds), pa in zip(per, adj):
                lines.append(f"    {att}: n={ds['n']} Δ={ds['delta_pp']:+.2f} p.p. CI95=({ds['ci95_pp'][0]:+.2f}, {ds['ci95_pp'][1]:+.2f}) d={ds['d']:+.2f} Wilcoxon p={ds['wilcoxon_p']:.4f} Holm={pa:.4f}")
        lines.append("")

    lines += ["== Secondary / exploratory (mean per attack × condition) ==",
              df.groupby(["attack", "condition"])[["final_acc", "att_mass_mean", "n_resets_extra",
                                                    "simlc_rows_per_round", "act_dist_center_post100",
                                                    "act_std_post100"]].mean().round(4).to_string()]

    curves = {f"{r.attack}_{r.condition}_{r.seed}": r.acc for r in df.itertuples()}
    out_dir = os.path.dirname(a.out)
    pd.DataFrame({k: pd.Series(v) for k, v in curves.items()}).to_csv(os.path.join(out_dir, "curvas_acc.csv"), index_label="rodada_menos_1")
    df.drop(columns=["acc"]).to_csv(os.path.join(out_dir, "resumo_runs.csv"), index=False)
    text = "\n".join(lines)
    open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
