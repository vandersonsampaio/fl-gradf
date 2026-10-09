"""
B3.3 analysis (results/b33_limiar_bloodmnist/PLANO.md §3 and §5). Written before the grid.

For each a₅ ∈ {0.75; 0.95}: Δ = metric(a₅) − metric(center, a₅ = 0.475), paired by
seed (145-149), t CI95 (4 d.f.); Wilcoxon descriptive only (n = 5 → min. p 0.0625).
  primary          median accuracy over rounds 401-500
  AUC sensitivity: mean 1-500 and mean 251-500
Criterion: some a₅ with Δ > 2 p.p. and CI95 > 0 on the primary → "some constant beats the center";
if met on the primary but Δ ≤ 0 on AUC 1-500 → "reset-sensitive".
Δ < −2 p.p. with CI95 < 0 → "higher threshold is worse" (descriptive).
Mechanism (descriptive): resets per run, median interval between resets, mass on attackers,
excluded clients and excluded attackers per round.
Exploratory, no multiplicity correction (2 comparisons).

Usage (official venv): external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_b33.py
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/b33_limiar_bloodmnist"
A5 = [0.75, 0.95]
CENTER = 0.475
SEEDS = list(range(145, 150))
ROUNDS = 500
THRESH_PP = 2.0
METRICS = ["primaria", "auc_1_500", "auc_251_500"]


def _load(path, a5):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    a_fixed = d.get("a_fixed")
    assert a_fixed is not None and abs(a_fixed[4] - a5) < 1e-9 and all(abs(x - 0.475) < 1e-9 for x in a_fixed[:4]), \
        f"unexpected a_fixed in {path}: {a_fixed}"
    st = d["steps"][:ROUNDS]
    acc = np.array([x["acc"] for x in st])
    real = np.array([x["n_att_real"] for x in st])
    mass = np.array([np.nan if x["att_weight_mass"] is None else x["att_weight_mass"] for x in st])
    exc = np.array([np.nan if x["n_excluded"] is None else x["n_excluded"] for x in st])
    aexc = np.array([np.nan if x.get("n_att_excluded") is None else x["n_att_excluded"] for x in st])
    sel = (real > 0) & ~np.isnan(mass)
    rs = [r["env_step"] for r in d["resets"][1:]]
    return {"primaria": float(np.median(acc[400:500])), "auc_1_500": float(acc.mean()),
            "auc_251_500": float(acc[250:500].mean()), "resets": len(rs),
            "intervalo_reset_mediano": float(np.median(np.diff([0] + rs))) if rs else np.nan,
            "reset_401_500": any(400 <= r <= 500 for r in rs),
            "massa_atac": float(np.mean(mass[sel])) if sel.any() else np.nan,
            "excluidos": float(np.nanmean(exc)), "atac_excluidos": float(np.nanmean(aexc))}


def _ci(d):
    n = len(d)
    m = d.mean()
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def main():
    rows = []
    for a5 in [CENTER] + A5:
        for s in SEEDS:
            r = _load(f"{OUT}/raw/a5_{a5:g}/BloodMNIST_EB_q0.5_fixed_seed{s}_R{ROUNDS}.json", a5)
            if r:
                rows.append({"a5": a5, "seed": s, **r})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/resumo_runs.csv", index=False)
    cols = METRICS + ["resets", "intervalo_reset_mediano", "reset_401_500", "massa_atac", "excluidos", "atac_excluidos"]
    lines = [f"B3.3 — threshold a₅ in the official AdaAggRL, BloodMNIST, EB, seeds {SEEDS[0]}–{SEEDS[-1]}, {ROUNDS} rounds",
             f"runs: {len(df)} (expected {len(SEEDS) * (1 + len(A5))})", "",
             "Descriptive by a₅ (mean across seeds):",
             df.groupby("a5")[cols].mean().round(4).to_string(), ""]
    center = df[df.a5 == CENTER].set_index("seed")
    res, supera, piora = [], [], []
    for a5 in A5:
        g = df[df.a5 == a5].set_index("seed")
        out = {"a5": a5}
        for m in METRICS:
            d = (g[m] - center[m]).dropna().to_numpy()
            if len(d) < 2:
                continue
            mean, lo, hi = _ci(d)
            wp = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            out.update({f"{m}_delta_pp": 100 * mean, f"{m}_ic95_lo": 100 * lo, f"{m}_ic95_hi": 100 * hi,
                        f"{m}_wilcoxon_p": wp, "n": len(d)})
        if out.get("primaria_delta_pp", -np.inf) > THRESH_PP and out.get("primaria_ic95_lo", -np.inf) > 0:
            supera.append((a5, out.get("auc_1_500_delta_pp", np.nan) <= 0))
        if out.get("primaria_delta_pp", np.inf) < -THRESH_PP and out.get("primaria_ic95_hi", np.inf) < 0:
            piora.append(a5)
        res.append(out)
    r = pd.DataFrame(res)
    r.to_csv(f"{OUT}/delta_vs_centro.csv", index=False)
    lines += ["Δ vs. the center (a₅ = 0.475), paired by seed, t CI95 (descriptive Wilcoxon; n = 5 → min. p 0.0625):",
              r.round(3).to_string(index=False), ""]
    if supera:
        partes = [f"a₅={a}" + (" (RESET-SENSITIVE: Δ ≤ 0 on AUC 1-500)" if sens else "") for a, sens in supera]
        veredito = "SOME CONSTANT BEATS THE CENTER — " + ", ".join(partes)
    else:
        veredito = "NO CONSTANT BEATS THE CENTER in this sweep"
    lines += [f"Criterion (Δ > {THRESH_PP} p.p. with CI95 > 0 on the primary): {veredito}"]
    if piora:
        lines.append(f"Higher threshold is worse (Δ < −{THRESH_PP} p.p. with CI95 < 0, descriptive): a₅ ∈ {piora}")
    lines.append("Exploratory, no multiplicity correction (2 comparisons).")
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/analise.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
