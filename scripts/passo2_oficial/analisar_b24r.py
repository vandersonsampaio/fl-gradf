"""
B2.4r analysis, following results/b24r_limiar_oficial/PLANO.md. Written before the grid.

For each a₅ ∈ {0; 0.1; 0.25; 0.95}: Δ = metric(a₅) − metric(center, a₅ = 0.475),
paired by seed (100-104), t CI95. Center: re-run in this grid (ADENDO1:
the official code on GPU is not bit-reproducible across runs, so the Step 2 center
did not reproduce); the Step 2 center appears only as a descriptive reference.
  primary    median accuracy over rounds 401-500
  secondary  mean accuracy over the 500 rounds (robust to resets)
  Q1 causal effect: some a₅ with |Δ| > 2 p.p. and CI95 excluding 0 (primary OR secondary)
  Q2 better constant: some a₅ with Δ > 0 and CI95 > 0 on the primary AND Δ > 0 on the secondary
Mechanism: resets, mass on attackers, excluded clients and excluded attackers per round.

Usage (official venv): external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_b24r.py
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/b24r_limiar_oficial"
CENTER_RAW = "results/frente1_passo2_oficial/raw"
A5 = [0.0, 0.1, 0.25, 0.95]
SEEDS = list(range(100, 105))
ROUNDS = 500
THRESH_PP = 2.0


def _load(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    st = d["steps"][:ROUNDS]
    acc = np.array([x["acc"] for x in st])
    real = np.array([x["n_att_real"] for x in st])
    mass = np.array([np.nan if x["att_weight_mass"] is None else x["att_weight_mass"] for x in st])
    exc = np.array([np.nan if x["n_excluded"] is None else x["n_excluded"] for x in st])
    aexc = np.array([np.nan if x.get("n_att_excluded") is None else x["n_att_excluded"] for x in st])
    sel = (real > 0) & ~np.isnan(mass)
    return {"primaria": float(np.median(acc[400:500])), "secundaria": float(np.mean(acc)),
            "resets": len(d["resets"]) - 1, "massa_atac": float(np.mean(mass[sel])) if sel.any() else np.nan,
            "excluidos": float(np.nanmean(exc)), "atac_excluidos": float(np.nanmean(aexc)),
            "a_fixed": d.get("a_fixed")}


def _ci(d):
    n = len(d)
    m = d.mean()
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, m - h, m + h


def main():
    rows = []
    for s in SEEDS:
        r = _load(f"{CENTER_RAW}/MNIST_EB_q0.5_fixed_seed{s}_R{ROUNDS}.json")
        if r:
            rows.append({"a5": "centro_passo2", "seed": s, **r})
        for a5 in A5 + [0.475]:
            r = _load(f"{OUT}/raw/a5_{a5:g}/MNIST_EB_q0.5_fixed_seed{s}_R{ROUNDS}.json")
            if r:
                rows.append({"a5": a5, "seed": s, **r})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/resumo_runs.csv", index=False)

    lines = [f"B2.4r — threshold a₅ in the official AdaAggRL, EB, seeds {SEEDS[0]}–{SEEDS[-1]}, {ROUNDS} rounds",
             f"runs: {len(df[df.a5 != 'centro_passo2'])} (expected 25, with the re-run center) + Step 2 center (descriptive): {len(df[df.a5 == 'centro_passo2'])}", "",
             "Descriptive by a₅ (mean across seeds):",
             df.astype({"a5": str}).groupby("a5")[["primaria", "secundaria", "resets", "massa_atac", "excluidos", "atac_excluidos"]]
             .mean().round(4).to_string(), ""]
    center = df[df.a5 == 0.475].set_index("seed")
    q1, q2 = [], []
    res = []
    for a5 in A5:
        g = df[df.a5 == a5].set_index("seed")
        out = {"a5": a5}
        for m in ["primaria", "secundaria"]:
            d = (g[m] - center[m]).dropna().to_numpy()
            if len(d) < 2:
                continue
            mean, lo, hi = _ci(d)
            wp = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            out.update({f"{m}_delta_pp": 100 * mean, f"{m}_ic95_lo": 100 * lo, f"{m}_ic95_hi": 100 * hi,
                        f"{m}_wilcoxon_p": wp, "n": len(d)})
            if abs(100 * mean) > THRESH_PP and (lo > 0 or hi < 0):
                q1.append((a5, m))
        if out.get("primaria_ic95_lo", -1) > 0 and out.get("secundaria_delta_pp", -1) > 0:
            q2.append(a5)
        res.append(out)
    r = pd.DataFrame(res)
    r.to_csv(f"{OUT}/delta_vs_centro.csv", index=False)
    lines += ["Δ vs. the center (a₅ = 0.475), paired by seed, t CI95 (descriptive Wilcoxon; n = 5 → min. p 0.0625):",
              r.round(3).to_string(index=False), "",
              f"Q1 (causal effect: |Δ| > {THRESH_PP} p.p. with CI95 excluding 0): "
              + ("YES — " + ", ".join(f"a₅={a} ({m})" for a, m in q1) if q1 else "NO"),
              "Q2 (constant better than the center): " + ("YES — a₅ ∈ " + str(q2) if q2 else "NO"),
              "Exploratory, no multiplicity correction (4 comparisons)."]
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/analise.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
