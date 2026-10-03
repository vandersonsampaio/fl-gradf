"""
Análise do B2.4r, conforme results/b24r_limiar_oficial/PLANO.md. Escrita antes da grade.

Para cada a₅ ∈ {0; 0,1; 0,25; 0,95}: Δ = métrica(a₅) − métrica(centro, a₅ = 0,475),
pareado por semente (100-104), IC95 t. Centro: rodado de novo nesta grade (ADENDO1:
o código oficial na GPU não é bit-reprodutível entre runs, então o centro do Passo 2
não reproduziu); o centro do Passo 2 aparece só como referência descritiva.
  primária   mediana da acurácia nas rodadas 401-500
  secundária média da acurácia nas 500 rodadas (robusta a reset)
  Q1 efeito causal: algum a₅ com |Δ| > 2 p.p. e IC95 excluindo 0 (primária OU secundária)
  Q2 constante melhor: algum a₅ com Δ > 0 e IC95 > 0 na primária E Δ > 0 na secundária
Mecanismo: resets, massa nos atacantes, excluídos e atacantes excluídos por rodada.

Uso (venv oficial): external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_b24r.py
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

    lines = [f"B2.4r — limiar a₅ no AdaAggRL oficial, EB, sementes {SEEDS[0]}–{SEEDS[-1]}, {ROUNDS} rodadas",
             f"runs: {len(df[df.a5 != 'centro_passo2'])} (esperado 25, com o centro refeito) + centro do Passo 2 (descritivo): {len(df[df.a5 == 'centro_passo2'])}", "",
             "Descritivo por a₅ (média entre sementes):",
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
    lines += ["Δ contra o centro (a₅ = 0,475), pareado por semente, IC95 t (Wilcoxon descritivo; n = 5 → p mín. 0,0625):",
              r.round(3).to_string(index=False), "",
              f"Q1 (efeito causal: |Δ| > {THRESH_PP} p.p. com IC95 excluindo 0): "
              + ("SIM — " + ", ".join(f"a₅={a} ({m})" for a, m in q1) if q1 else "NÃO"),
              "Q2 (constante melhor que o centro): " + ("SIM — a₅ ∈ " + str(q2) if q2 else "NÃO"),
              "Exploratório, sem correção de multiplicidade (4 comparações)."]
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/analise.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
