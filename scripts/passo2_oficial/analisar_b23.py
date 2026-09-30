"""
Análise do B2.3 (steelman do TD3), conforme results/b23_steelman_oficial/PLANO.md.
Escrita antes de existir qualquer resultado. Exploratória (sementes gastas 100-104).

Pareamento por (ataque, semente) com os runs do Passo 2:
  fixed, td3 (oficial): results/frente1_passo2_oficial/raw/
  steelman:             results/b23_steelman_oficial/raw/

Métricas (runs truncados em 500 passos):
  primária: mediana da acurácia nas rodadas 401-500 (a mesma do B2.1)
  secundária: média 451-500 (a do Passo 2)
Comparações (n=10 pares, Wilcoxon bilateral, Δ, IC95, d; Holm por ataque):
  C1 steelman − fixed   <- define o Portão B-a
  C2 steelman − td3 oficial
Mecanismo (política determinística reconstruída dos checkpoints):
  drift_t = média |π_t(s) − π_0(s)| nos estados das rodadas 401-500, t = 50..500
  S_swap  = média |π_500(s_ataque,t) − π_500(s_outro,t)|, t em 401-500
  comparados com σ_a = 0,0475 (ruído de exploração).
Critério do Portão B-a (fixado no PLANO antes dos dados):
  "o steelman aprende e supera a fixa" = C1 com Δ > 0 e Wilcoxon p < 0,05 (primária)
  E mediana de drift_500 > σ_a E mediana de S_swap > σ_a.

Uso:
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_b23.py
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analisar as P2  # noqa: E402
import analisar_b21 as B21A  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ATTACKS = ["LMP", "EB"]
SEEDS = list(range(100, 105))
ROUNDS = 500
SIGMA_A = B21A.SIGMA_A
P2_RAW = "results/frente1_passo2_oficial/raw"


def _tag(att, cond, seed):
    return f"MNIST_{att}_q0.5_{cond}_seed{seed}_R{ROUNDS}"


def _metrics(path):
    d = json.load(open(path))
    acc = np.array([x["acc"] for x in d["steps"][:ROUNDS]])
    return {"median_401_500": float(np.median(acc[400:500])), "mean_451_500": float(np.mean(acc[450:500])),
            "n_resets_extra": len(d["resets"]) - 1, "n_steps": len(d["steps"][:ROUNDS])}


def compare(tab, a, b, label, lines):
    pair = tab[[a, b]].dropna()
    d = (pair[a] - pair[b]).to_numpy()
    desc = P2.describe(d)
    lines += [f"== {label}: {a} − {b} (pooled, n={desc['n']}) ==",
              f"  Δ={desc['delta_pp']:+.2f} p.p.  IC95=({desc['ci95_pp'][0]:+.2f}, {desc['ci95_pp'][1]:+.2f})  d={desc['d']:+.2f}  Wilcoxon p={desc['wilcoxon_p']:.4f}"]
    per = []
    for att in ATTACKS:
        if att in pair.index.get_level_values(0):
            da = (pair.loc[att][a] - pair.loc[att][b]).to_numpy()
            if len(da) >= 2:
                per.append((att, P2.describe(da)))
    if per:
        adj = P2.holm(np.array([x[1]["wilcoxon_p"] for x in per]))
        for (att, ds), pa in zip(per, adj):
            lines.append(f"    {att}: Δ={ds['delta_pp']:+.2f} p.p. IC95=({ds['ci95_pp'][0]:+.2f}, {ds['ci95_pp'][1]:+.2f}) d={ds['d']:+.2f} Wilcoxon p={ds['wilcoxon_p']:.4f} Holm={pa:.4f}")
    lines.append("")
    return desc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="results/b23_steelman_oficial/raw")
    p.add_argument("--out", default="results/b23_steelman_oficial/analise.txt")
    a = p.parse_args()

    rows = []
    for att in ATTACKS:
        for s in SEEDS:
            for cond, path in [("fixed", os.path.join(P2_RAW, _tag(att, "fixed", s) + ".json")),
                               ("td3", os.path.join(P2_RAW, _tag(att, "td3", s) + ".json")),
                               ("steelman", os.path.join(a.raw_dir, _tag(att, "td3", s) + ".json"))]:
                if os.path.exists(path):
                    rows.append({"attack": att, "seed": s, "condition": cond, **_metrics(path)})
    df = pd.DataFrame(rows)
    n_st = int((df["condition"] == "steelman").sum())
    lines = [f"B2.3 steelman — runs steelman carregados: {n_st} (esperados 10); pareados com Passo 2 (fixed, td3)", ""]

    tab = df.pivot_table(index=["attack", "seed"], columns="condition", values="median_401_500")
    lines += ["Primária (mediana 401–500):", tab.round(4).to_string(), ""]
    c1 = compare(tab, "steelman", "fixed", "C1 [primária]", lines)
    compare(tab, "steelman", "td3", "C2 [primária]", lines)
    tab2 = df.pivot_table(index=["attack", "seed"], columns="condition", values="mean_451_500")
    compare(tab2, "steelman", "fixed", "C1 [secundária: média 451–500]", lines)
    lines += ["Resets extras (média por ataque × condição):",
              df.groupby(["attack", "condition"])["n_resets_extra"].mean().round(2).to_string(), ""]

    # Mecanismo do steelman
    policy = B21A.make_actor_evaluator()
    mech, curve = [], []
    for att in ATTACKS:
        other = "LMP" if att == "EB" else "EB"
        for s in SEEDS:
            tag, otag = _tag(att, "td3", s), _tag(other, "td3", s)
            act = lambda t, T=tag: os.path.join(a.raw_dir, "actors", f"{T}_step{t:03d}.pt")  # noqa: E731
            obs_p, oobs_p = (os.path.join(a.raw_dir, "obs", f"{x}.npy") for x in (tag, otag))
            if not (os.path.exists(act(0)) and os.path.exists(act(500)) and os.path.exists(obs_p) and os.path.exists(oobs_p)):
                continue
            S = np.load(obs_p)[400:500]
            S_o = np.load(oobs_p)[400:500]
            pi0 = policy(act(0), S)
            for t in range(50, 501, 50):
                if os.path.exists(act(t)):
                    curve.append({"attack": att, "seed": s, "step": t,
                                  "drift": float(np.mean(np.abs(policy(act(t), S) - pi0)))})
            pi5 = policy(act(500), S)
            mech.append({"attack": att, "seed": s,
                         "drift_500": float(np.mean(np.abs(pi5 - pi0))),
                         "S_swap": float(np.mean(np.abs(pi5 - policy(act(500), S_o)))),
                         "pi500_dist_center": float(np.mean(np.abs(pi5 - 0.475)))})
    m = pd.DataFrame(mech)
    cv = pd.DataFrame(curve)
    if len(m):
        lines += [f"== Mecanismo do steelman (σ_a = {SIGMA_A:.4f}) ==",
                  m.round(4).to_string(index=False),
                  f"  mediana drift_500={m['drift_500'].median():.4f}  mediana S_swap={m['S_swap'].median():.4f}",
                  "  drift ao longo do treino (mediana entre runs):",
                  cv.groupby("step")["drift"].median().round(4).to_string(), ""]

    moved = len(m) and m["drift_500"].median() > SIGMA_A
    uses_input = len(m) and m["S_swap"].median() > SIGMA_A
    beats = c1["delta_pp"] > 0 and c1["wilcoxon_p"] < 0.05
    lines += ["== Portão B-a (critério do PLANO) ==",
              f"  C1 steelman > fixed (Δ>0 e p<0,05): {'SIM' if beats else 'NÃO'}",
              f"  política se moveu (drift_500 > σ_a): {'SIM' if moved else 'NÃO'}",
              f"  política usa a entrada (S_swap > σ_a): {'SIM' if uses_input else 'NÃO'}",
              "  VEREDITO: " + ("o steelman APRENDE E SUPERA a fixa" if (beats and moved and uses_input)
                               else "o steelman NÃO supera a fixa (tese do P2 fortalecida)" if not beats
                               else "supera a fixa, mas sem evidência de aprendizado dependente da entrada (investigar)")]

    out_dir = os.path.dirname(a.out)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "resumo_runs.csv"), index=False)
    m.to_csv(os.path.join(out_dir, "mecanismo.csv"), index=False)
    cv.to_csv(os.path.join(out_dir, "drift_por_checkpoint.csv"), index=False)
    text = "\n".join(lines)
    open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
