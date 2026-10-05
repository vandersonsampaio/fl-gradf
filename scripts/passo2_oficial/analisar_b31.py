"""
Análise do B3.1 + B3.2 (results/b31_medmnist_oficial/PREREGISTRO.md §5), escrita
antes da grade. Margem M e ataques incluídos vêm do Adendo 1 (argumentos).

B3.1 (unidade = par ataque × semente): D = métrica(fixed) − métrica(td3)
  primária: mediana da acurácia nas rodadas 401-500
  sensibilidade por AUC: média 1-500 e média 251-500
  TOST pareado ±M (IC90 e IC95), Wilcoxon; por ataque com Holm. Primária e AUCs
  discordando no veredito -> "sensível a resets".
  Resets: contam como resultado; nº por run e fração de runs com reset em 401-500.
B3.2 (Holm sobre H3-H5, unilaterais, σ_a = 0,0475), como no B2.2.

Uso (venv oficial):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_b31.py --margem 1.25 --ataques LMP EB
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analisar as P2  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

OUT = "results/b31_medmnist_oficial"
RAW = f"{OUT}/raw"
SEEDS = list(range(135, 145))
ROUNDS = 500
SIGMA_A = 0.1 * 0.95 / 2
R_THRESH = 0.9
ALPHA = 0.05


def _tag(att, cond, s):
    return f"BloodMNIST_{att}_q0.5_{cond}_seed{s}_R{ROUNDS}"


def load(ataques):
    rows = []
    for att in ataques:
        for cond in ["fixed", "td3"]:
            for s in SEEDS:
                f = f"{RAW}/{_tag(att, cond, s)}.json"
                if not os.path.exists(f):
                    continue
                d = json.load(open(f))
                st = d["steps"][:ROUNDS]
                acc = np.array([x["acc"] for x in st])
                real = np.array([x["n_att_real"] for x in st])
                mass = np.array([np.nan if x["att_weight_mass"] is None else x["att_weight_mass"] for x in st])
                rs = [r["env_step"] for r in d["resets"][1:]]
                rows.append({"attack": att, "condition": cond, "seed": s,
                             "primaria": float(np.median(acc[400:500])), "auc_1_500": float(acc.mean()),
                             "auc_251_500": float(acc[250:500].mean()), "resets": len(rs),
                             "reset_401_500": any(400 <= r <= 500 for r in rs),
                             "massa_atac": float(np.nanmean(mass[real > 0])) if (real > 0).any() else np.nan,
                             "actions": np.array([x["action"] for x in st])})
    return pd.DataFrame(rows)


def _ci(d, level):
    n = len(d)
    m = d.mean()
    h = stats.t.ppf(0.5 + level / 2, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return 100 * (m - h), 100 * (m + h)


def h1(df, metric, margem, ataques, lines):
    tab = df.pivot_table(index=["attack", "seed"], columns="condition", values=metric).dropna()
    d = (tab["fixed"] - tab["td3"]).to_numpy()
    desc = P2.describe(d)
    p_tost = max(P2.tost_paired(d, margem / 100))
    v = P2.verdict(desc, p_tost)
    c90, c95 = _ci(d, 0.90), _ci(d, 0.95)
    lines += [f"== H1 ({metric}): fixed − td3, n={desc['n']} ==",
              f"  Δ={desc['delta_pp']:+.2f} p.p.  IC90=({c90[0]:+.2f}, {c90[1]:+.2f})  IC95=({c95[0]:+.2f}, {c95[1]:+.2f})  d={desc['d']:+.2f}",
              f"  TOST ±{margem:.2f} p.p.: p={p_tost:.4f}   Wilcoxon: p={desc['wilcoxon_p']:.4f}   VEREDITO: {v}"]
    per = []
    for att in ataques:
        if att in tab.index.get_level_values(0):
            da = (tab.loc[att]["fixed"] - tab.loc[att]["td3"]).to_numpy()
            if len(da) >= 2:
                per.append((att, P2.describe(da)))
    if per:
        adj = P2.holm(np.array([x[1]["wilcoxon_p"] for x in per]))
        for (att, ds), pa in zip(per, adj):
            lines.append(f"    {att}: n={ds['n']} Δ={ds['delta_pp']:+.2f} p.p. IC95=({ds['ci95_pp'][0]:+.2f}, {ds['ci95_pp'][1]:+.2f}) "
                         f"d={ds['d']:+.2f} Wilcoxon p={ds['wilcoxon_p']:.4f} Holm={pa:.4f}")
    lines.append("")
    return v


def b32(df, ataques, lines):
    from analisar_b21 import make_actor_evaluator
    policy = make_actor_evaluator()
    td3 = df[df.condition == "td3"].set_index(["attack", "seed"])
    pvals, labels = [], []
    if len(ataques) == 2:
        rs = []
        for s in SEEDS:
            if all((a, s) in td3.index for a in ataques):
                a, b = td3.loc[(ataques[0], s), "actions"][100:500], td3.loc[(ataques[1], s), "actions"][100:500]
                rs.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
        r = np.array(rs)
        pvals.append(float(stats.wilcoxon(np.arctanh(np.clip(r, -0.999999, 0.999999)) - np.arctanh(R_THRESH),
                                          alternative="greater").pvalue))
        labels.append(f"H3 corr(ações entre ataques) > {R_THRESH}: mediana r={np.median(r):.3f}")
    rows = []
    for (att, s) in td3.index:
        tag = _tag(att, "td3", s)
        p0, p5 = f"{RAW}/actors/{tag}_step000.pt", f"{RAW}/actors/{tag}_step500.pt"
        ob = f"{RAW}/obs/{tag}.npy"
        if not all(os.path.exists(x) for x in (p0, p5, ob)):
            continue
        S = np.load(ob)[400:500]
        pi0, pi5 = policy(p0, S), policy(p5, S)
        row = {"attack": att, "seed": s, "drift": float(np.mean(np.abs(pi5 - pi0))),
               "sd_estados": float(np.mean(pi5.std(axis=0)))}
        other = [a for a in ataques if a != att]
        if other:
            ob2 = f"{RAW}/obs/{_tag(other[0], 'td3', s)}.npy"
            if os.path.exists(ob2):
                row["S_swap"] = float(np.mean(np.abs(pi5 - policy(p5, np.load(ob2)[400:500]))))
        rng = np.random.RandomState(0)
        full = np.load(ob)[:ROUNDS]
        row["S_shuffle"] = float(np.mean(np.abs(pi5 - policy(p5, full[rng.randint(0, len(full), size=len(S))]))))
        rows.append(row)
    m = pd.DataFrame(rows)
    m.to_csv(f"{OUT}/mecanismo.csv", index=False)
    pvals.append(float(stats.wilcoxon(m.drift - SIGMA_A, alternative="less").pvalue))
    labels.append(f"H4 drift < σ_a: mediana={m.drift.median():.4f} máx={m.drift.max():.4f}")
    if "S_swap" in m:
        pvals.append(float(stats.wilcoxon(m.S_swap.dropna() - SIGMA_A, alternative="less").pvalue))
        labels.append(f"H5 S_swap < σ_a: mediana={m.S_swap.median():.4f}")
    adj = P2.holm(np.array(pvals))
    lines += [f"== B3.2 (Holm; σ_a = {SIGMA_A:.4f}) =="]
    for lab, p, pa in zip(labels, pvals, adj):
        lines.append(f"  {lab}  p={p:.4f} Holm={pa:.4f} -> {'CONFIRMADA' if pa < ALPHA else 'NÃO confirmada'}")
    lines.append(f"  descritivo: sd_estados mediana={m.sd_estados.median():.4f}; S_shuffle mediana={m.S_shuffle.median():.4f}")
    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margem", type=float, required=True, help="M em p.p. (Adendo 1)")
    ap.add_argument("--ataques", nargs="+", required=True)
    a = ap.parse_args()
    df = load(a.ataques)
    df.drop(columns="actions").to_csv(f"{OUT}/resumo_runs.csv", index=False)
    lines = [f"B3.1 + B3.2 — BloodMNIST, AdaAggRL oficial; ataques {a.ataques}; M = ±{a.margem:.2f} p.p.",
             f"runs: {len(df)} (esperado {len(a.ataques) * 2 * len(SEEDS)})", ""]
    vs = {m: h1(df, m, a.margem, a.ataques, lines) for m in ["primaria", "auc_1_500", "auc_251_500"]}
    if len(set(vs.values())) > 1:
        lines.append(f"SENSÍVEL A RESETS: vereditos diferem entre primária e AUCs ({vs})")
    else:
        lines.append(f"Primária e AUCs concordam: {vs['primaria']}")
    g = df.groupby("condition")
    lines += ["", "Resets (contam como resultado):",
              g.agg(resets_media=("resets", "mean"), frac_reset_401_500=("reset_401_500", "mean"),
                    massa_atac=("massa_atac", "median")).round(3).to_string()]
    t = df.pivot_table(index=["attack", "seed"], columns="condition", values="resets").dropna()
    dr = (t["fixed"] - t["td3"]).to_numpy()
    lines += [f"  Wilcoxon pareado do nº de resets (descritivo): p={P2.describe(dr.astype(float))['wilcoxon_p']:.4f}", ""]
    b32(df, a.ataques, lines)
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/analise.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
