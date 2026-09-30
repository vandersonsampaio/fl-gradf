"""
Análise do B2.1 + B2.2, conforme results/b21_replicacao_oficial/PREREGISTRO.md.
Escrita antes de existir qualquer resultado da grade. Reusa os testes de
`analisar.py` (Passo 2) sem alterá-lo.

B2.1 (unidade = par ataque×semente, n=20):
  Primária: mediana da acurácia nas rodadas 401-500 (runs truncados em 500 passos).
  H1: fixed - td3. TOST pareado (t), margem ±1,0 p.p., alfa 0,05; Wilcoxon
      pareado para diferença. Por ataque: Δ, IC95, d, Wilcoxon + Holm.
  Secundárias: média 451-500 (métrica do Passo 2), nº de resets, massa nos atacantes.

B2.2 (família de 3 hipóteses, Holm sobre H3-H5, alfa 0,05; testes unilaterais):
  σ_a = 0,1 × 0,95/2 = 0,0475 (desvio do ruído de exploração do SB3 em unidades de ação).
  H3: corr. passo a passo das ações EXECUTADAS do td3 (rodadas 101-500, 5 dims)
      entre EB e LMP da mesma semente é > 0,9. Wilcoxon unilateral em
      atanh(r) - atanh(0,9) > 0, n=10.
  H4: a política determinística final quase não difere da inicial:
      drift = média |π_500(s) - π_0(s)| sobre os estados s das rodadas 401-500 e
      as 5 dims. Wilcoxon unilateral drift - σ_a < 0, n=20 runs td3.
  H5: a política final quase não depende da entrada:
      S_swap = média |π_500(s_ataque,t) - π_500(s_outro_ataque,t)|, t em 401-500,
      mesma semente. Wilcoxon unilateral S_swap - σ_a < 0, n=20.
  Reportados sem teste: S_shuffle (estado de outra rodada do mesmo run), e
  drift/S_swap do ator inicial π_0 como referência.

Uso:
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_b21.py
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analisar as P2  # noqa: E402  (tost_paired, describe, holm)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

MARGIN = 0.010
ALPHA = 0.05
ATTACKS = ["LMP", "EB"]
SEEDS = list(range(105, 115))
SIGMA_A = 0.1 * 0.95 / 2
R_THRESH = 0.9
ROUNDS = 500


def _tag(att, cond, seed):
    return f"MNIST_{att}_q0.5_{cond}_seed{seed}_R{ROUNDS}"


def load_runs(raw):
    rows = []
    for att in ATTACKS:
        for cond in ["fixed", "td3"]:
            for s in SEEDS:
                f = os.path.join(raw, _tag(att, cond, s) + ".json")
                if not os.path.exists(f):
                    continue
                d = json.load(open(f))
                steps = d["steps"][:ROUNDS]  # Adendo 1 do Passo 2: td3 do SB3 roda 501 passos
                acc = np.array([x["acc"] for x in steps])
                real = np.array([x["n_att_real"] for x in steps])
                mass = np.array([np.nan if x["att_weight_mass"] is None else x["att_weight_mass"] for x in steps])
                rows.append({
                    "attack": att, "condition": cond, "seed": s, "n_steps": len(steps),
                    "n_steps_raw": len(d["steps"]),
                    "primary_median_401_500": float(np.median(acc[400:500])),
                    "mean_451_500": float(np.mean(acc[450:500])),
                    "n_resets_extra": len(d["resets"]) - 1,
                    "att_mass_mean": float(np.nanmean(mass[real > 0])) if (real > 0).any() else np.nan,
                    "actions": np.array([x["action"] for x in steps]),
                })
    return pd.DataFrame(rows)


def paired_block(tab, metric_label, lines):
    pair = tab[["fixed", "td3"]].dropna()
    d = (pair["fixed"] - pair["td3"]).to_numpy()
    desc = P2.describe(d)
    p_tost = max(P2.tost_paired(d, MARGIN))
    lines += [f"== H1 ({metric_label}): fixed − td3 (pooled, pares ataque×semente) ==",
              f"  n={desc['n']}  Δ={desc['delta_pp']:+.2f} p.p.  IC95=({desc['ci95_pp'][0]:+.2f}, {desc['ci95_pp'][1]:+.2f})  d={desc['d']:+.2f}",
              f"  TOST ±{100*MARGIN:.1f} p.p.: p={p_tost:.4f}   Wilcoxon: p={desc['wilcoxon_p']:.4f}",
              f"  VEREDITO: {P2.verdict(desc, p_tost)}"]
    per = []
    for att in ATTACKS:
        if att in pair.index.get_level_values(0):
            da = (pair.loc[att]["fixed"] - pair.loc[att]["td3"]).to_numpy()
            if len(da) >= 2:
                per.append((att, P2.describe(da)))
    if per:
        adj = P2.holm(np.array([x[1]["wilcoxon_p"] for x in per]))
        for (att, ds), pa in zip(per, adj):
            lines.append(f"    {att}: n={ds['n']} Δ={ds['delta_pp']:+.2f} p.p. IC95=({ds['ci95_pp'][0]:+.2f}, {ds['ci95_pp'][1]:+.2f}) d={ds['d']:+.2f} Wilcoxon p={ds['wilcoxon_p']:.4f} Holm={pa:.4f}")
    lines.append("")


# ---------------------------------------------------------------------------
# B2.2: reconstrução do ator do SB3 para avaliar a política determinística
# ---------------------------------------------------------------------------

def make_actor_evaluator():
    import gymnasium
    import torch
    from stable_baselines3 import TD3

    class _Dummy(gymnasium.Env):
        observation_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(10, 4), dtype=np.float32)
        action_space = gymnasium.spaces.Box(0.0, 0.95, shape=(5,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            return np.zeros((10, 4), np.float32), {}

        def step(self, a):
            return np.zeros((10, 4), np.float32), 0.0, False, False, {}

    model = TD3("MlpPolicy", _Dummy(), policy_kwargs={"net_arch": [256, 128]}, device="cpu", verbose=0)

    def policy(actor_path, obs):
        model.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
        acts, _ = model.predict(obs.astype(np.float32), deterministic=True)
        return np.asarray(acts, dtype=np.float64)

    return policy


def one_sided_wilcoxon_less(x):
    """p de H1: mediana(x) < 0."""
    return float(stats.wilcoxon(x, alternative="less").pvalue)


def one_sided_wilcoxon_greater(x):
    return float(stats.wilcoxon(x, alternative="greater").pvalue)


def b22(df, raw, lines):
    policy = make_actor_evaluator()
    td3 = df[df["condition"] == "td3"].set_index(["attack", "seed"])
    rng = np.random.RandomState(0)

    # H3: correlação das ações executadas entre ataques, mesma semente
    rs = []
    for s in SEEDS:
        if ("EB", s) in td3.index and ("LMP", s) in td3.index:
            a, b = td3.loc[("EB", s), "actions"][100:500], td3.loc[("LMP", s), "actions"][100:500]
            rs.append((s, float(np.corrcoef(a.ravel(), b.ravel())[0, 1])))
    r = np.array([x[1] for x in rs])
    p3 = one_sided_wilcoxon_greater(np.arctanh(np.clip(r, -0.999999, 0.999999)) - np.arctanh(R_THRESH))

    # H4 e H5: política determinística
    rows = []
    for (att, s) in td3.index:
        other = "LMP" if att == "EB" else "EB"
        tag, otag = _tag(att, "td3", s), _tag(other, "td3", s)
        p0 = os.path.join(raw, "actors", f"{tag}_step000.pt")
        p500 = os.path.join(raw, "actors", f"{tag}_step500.pt")
        obs = np.load(os.path.join(raw, "obs", f"{tag}.npy"))[:ROUNDS]
        oobs_path = os.path.join(raw, "obs", f"{otag}.npy")
        if not (os.path.exists(p0) and os.path.exists(p500) and os.path.exists(oobs_path)):
            continue
        oobs = np.load(oobs_path)[:ROUNDS]
        S = obs[400:500]
        S_other = oobs[400:500]
        S_shuf = obs[rng.randint(0, len(obs), size=len(S))]
        pi0, pi0_o = policy(p0, S), policy(p0, S_other)
        pi5, pi5_o, pi5_sh = policy(p500, S), policy(p500, S_other), policy(p500, S_shuf)
        rows.append({
            "attack": att, "seed": s,
            "drift": float(np.mean(np.abs(pi5 - pi0))),
            "S_swap": float(np.mean(np.abs(pi5 - pi5_o))),
            "S_shuffle": float(np.mean(np.abs(pi5 - pi5_sh))),
            "S_swap_pi0": float(np.mean(np.abs(pi0 - pi0_o))),
            "pi0_dist_center": float(np.mean(np.abs(pi0 - 0.475))),
        })
    m = pd.DataFrame(rows)
    p4 = one_sided_wilcoxon_less(m["drift"].to_numpy() - SIGMA_A) if len(m) else np.nan
    p5 = one_sided_wilcoxon_less(m["S_swap"].to_numpy() - SIGMA_A) if len(m) else np.nan
    adj = P2.holm(np.array([p3, p4, p5]))

    lines += ["== B2.2: hipóteses mecanísticas (Holm sobre H3–H5) ==",
              f"  σ_a (ruído de exploração, unidades de ação) = {SIGMA_A:.4f}",
              f"  H3 corr(EB, LMP) das ações > {R_THRESH}: n={len(r)}  r por semente={np.round(r, 3).tolist()}",
              f"     mediana r={np.median(r):.3f}  Wilcoxon unilateral p={p3:.4f}  Holm={adj[0]:.4f}  -> {'CONFIRMADA' if adj[0] < ALPHA else 'NÃO confirmada'}",
              f"  H4 drift |π500−π0| < σ_a: n={len(m)}  mediana={m['drift'].median():.4f}  máx={m['drift'].max():.4f}",
              f"     Wilcoxon unilateral p={p4:.4f}  Holm={adj[1]:.4f}  -> {'CONFIRMADA' if adj[1] < ALPHA else 'NÃO confirmada'}",
              f"  H5 S_swap |π500(s)−π500(s')| < σ_a: n={len(m)}  mediana={m['S_swap'].median():.4f}  máx={m['S_swap'].max():.4f}",
              f"     Wilcoxon unilateral p={p5:.4f}  Holm={adj[2]:.4f}  -> {'CONFIRMADA' if adj[2] < ALPHA else 'NÃO confirmada'}",
              "  Referências (sem teste): mediana S_shuffle={:.4f}; mediana S_swap do π0={:.4f}; mediana |π0−centro|={:.4f}".format(
                  m["S_shuffle"].median(), m["S_swap_pi0"].median(), m["pi0_dist_center"].median()),
              ""]
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="results/b21_replicacao_oficial/raw")
    p.add_argument("--out", default="results/b21_replicacao_oficial/analise.txt")
    a = p.parse_args()

    df = load_runs(a.raw_dir)
    lines = [f"B2.1 + B2.2 — {len(df)} runs carregados (esperados {len(ATTACKS) * 2 * len(SEEDS)})", ""]
    tab = df.pivot_table(index=["attack", "seed"], columns="condition", values="primary_median_401_500")
    lines += ["Métrica primária (mediana da acurácia nas rodadas 401–500):", tab.round(4).to_string(), ""]
    paired_block(tab, "primária: mediana 401–500", lines)
    tab2 = df.pivot_table(index=["attack", "seed"], columns="condition", values="mean_451_500")
    paired_block(tab2, "secundária: média 451–500, métrica do Passo 2", lines)
    lines += ["== Secundárias descritivas (média por ataque × condição) ==",
              df.groupby(["attack", "condition"])[["n_resets_extra", "att_mass_mean"]].mean().round(4).to_string(), ""]

    m = b22(df, a.raw_dir, lines)

    out_dir = os.path.dirname(a.out)
    df.drop(columns=["actions"]).to_csv(os.path.join(out_dir, "resumo_runs.csv"), index=False)
    m.to_csv(os.path.join(out_dir, "b22_mecanismo.csv"), index=False)
    text = "\n".join(lines)
    open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
