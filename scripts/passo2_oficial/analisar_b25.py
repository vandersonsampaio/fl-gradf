"""
Análise do B2.5 (IPM real no AdaAggRL oficial), conforme
results/b25_ipm_oficial/PLANO.md. Escrita antes de qualquer run do sanity e da grade.

Subcomandos:
  sanity   semente 100, 100 rodadas: aplica o critério de escolha do ε
           (só FedAvg e fixed; o td3 nunca entra na escolha)
  grade    sementes 105-114, 500 rodadas, ε escolhido: H1 (fixed − td3) com a
           métrica primária do B2.1, e mecanismo descritivo (drift, sd_estados)

Uso (venv oficial):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_b25.py sanity
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_b25.py grade --eps 10
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analisar as P2  # noqa: E402  (tost_paired, describe, holm, verdict)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

OUT = "results/b25_ipm_oficial"
RAW = os.path.join(OUT, "raw")
SANITY_RAW = os.path.join(OUT, "sanity")
EPS_CANDIDATES = [2.0, 10.0]
SANITY_SEED = 100
SANITY_ROUNDS = 100
DEGRADE_PP = 0.05     # FedAvg com ataque ≥ 5 p.p. abaixo do sem ataque (média 81-100)
MASS_MIN = 0.01       # massa média nos atacantes sob o fixed > 0,01
MARGIN = 0.010
ALPHA = 0.05
SEEDS = list(range(105, 115))
ROUNDS = 500
SIGMA_A = 0.1 * 0.95 / 2


def _alabel(eps):
    return "none" if eps is None else f"IPMr{eps:g}"


def _load(raw, eps, cond, seed, rounds):
    f = os.path.join(raw, f"MNIST_{_alabel(eps)}_q0.5_{cond}_seed{seed}_R{rounds}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    steps = d["steps"][:rounds]  # td3 do SB3 roda 501 passos (Adendo 1 do Passo 2)
    acc = np.array([x["acc"] for x in steps])
    real = np.array([x["n_att_real"] for x in steps])
    mass = np.array([np.nan if x["att_weight_mass"] is None else x["att_weight_mass"] for x in steps])
    sel = (real > 0) & ~np.isnan(mass)
    return {"d": d, "steps": steps, "acc": acc, "real": real,
            "mass_mean": float(np.mean(mass[sel])) if sel.any() else np.nan,
            "n_rounds_att": int(sel.sum()), "n_resets_extra": len(d["resets"]) - 1}


def sanity():
    lines = [f"B2.5 sanity — semente {SANITY_SEED}, {SANITY_ROUNDS} rodadas; métrica = média da acurácia nas rodadas 81–100",
             f"critério: degrada o FedAvg se ≥ {100*DEGRADE_PP:.0f} p.p. abaixo do FedAvg sem ataque; "
             f"não é totalmente excluído se a massa média nos atacantes sob o fixed > {MASS_MIN}", ""]
    base = _load(SANITY_RAW, None, "fedavg", SANITY_SEED, SANITY_ROUNDS)
    if base is None:
        raise SystemExit("falta o run fedavg sem ataque")
    acc0 = float(base["acc"][80:100].mean())
    lines.append(f"FedAvg sem ataque: {100*acc0:.2f}%  (resets extras: {base['n_resets_extra']})")
    rows = []
    for eps in EPS_CANDIDATES:
        fa = _load(SANITY_RAW, eps, "fedavg", SANITY_SEED, SANITY_ROUNDS)
        fx = _load(SANITY_RAW, eps, "fixed", SANITY_SEED, SANITY_ROUNDS)
        if fa is None or fx is None:
            raise SystemExit(f"faltam runs de ε={eps:g}")
        acc_fa = float(fa["acc"][80:100].mean())
        degrade = (acc0 - acc_fa) >= DEGRADE_PP
        incl = fx["mass_mean"] > MASS_MIN
        rows.append((eps, degrade, incl))
        lines.append(f"ε={eps:g}: FedAvg {100*acc_fa:.2f}% (queda {100*(acc0-acc_fa):+.2f} p.p., degrada={degrade}, "
                     f"resets extras {fa['n_resets_extra']}); fixed: massa média nos atacantes={fx['mass_mean']:.4f} "
                     f"em {fx['n_rounds_att']} rodadas com atacantes reais (não excluído={incl}), "
                     f"acc 81–100 {100*float(fx['acc'][80:100].mean()):.2f}%; "
                     f"massa FedAvg (referência)={fa['mass_mean']:.4f}")
    both = [e for e, dg, inc in rows if dg and inc]
    deg = [e for e, dg, _ in rows if dg]
    if both:
        choice, note = min(both), "satisfaz os dois critérios (desempate: menor ε)"
    elif deg:
        choice, note = min(deg), ("nenhum ε satisfaz os dois critérios; menor ε que degrada o FedAvg. "
                                  "DECLARAR: o resultado da grade pode ser trivial (o fixed exclui os atacantes)")
    else:
        choice, note = None, "nenhum ε degrada o FedAvg: caso não previsto, consultar o autor antes da grade"
    lines += ["", f"ESCOLHA: ε = {choice} — {note}"]
    text = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "sanity.txt"), "w").write(text)
    print(text)


def _ci(d, level):
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * se
    return 100 * (m - h), 100 * (m + h)


def _make_actor_evaluator():
    import analisar_b21 as A
    return A.make_actor_evaluator()


def grade(eps):
    rows = []
    for cond in ["fixed", "td3"]:
        for s in SEEDS:
            r = _load(RAW, eps, cond, s, ROUNDS)
            if r is None:
                continue
            rows.append({"condition": cond, "seed": s, "n_steps": len(r["steps"]),
                         "primary_median_401_500": float(np.median(r["acc"][400:500])),
                         "mean_451_500": float(np.mean(r["acc"][450:500])),
                         "n_resets_extra": r["n_resets_extra"], "att_mass_mean": r["mass_mean"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "resumo_runs.csv"), index=False)
    lines = [f"B2.5 — IPM real ε={eps:g}, MNIST q=0,5, sementes {SEEDS[0]}–{SEEDS[-1]}, {ROUNDS} rodadas",
             f"runs: {len(df)} (esperado 20)", ""]
    for metric, label in [("primary_median_401_500", "PRIMÁRIA: mediana 401–500"),
                          ("mean_451_500", "secundária: média 451–500")]:
        tab = df.pivot(index="seed", columns="condition", values=metric)[["fixed", "td3"]].dropna()
        d = (tab["fixed"] - tab["td3"]).to_numpy()
        desc = P2.describe(d)
        p_tost = max(P2.tost_paired(d, MARGIN))
        c90, c95 = _ci(d, 0.90), _ci(d, 0.95)
        lines += [f"== H1 ({label}): fixed − td3, n={desc['n']} sementes ==",
                  f"  Δ={desc['delta_pp']:+.2f} p.p.  IC90=({c90[0]:+.2f}, {c90[1]:+.2f})  IC95=({c95[0]:+.2f}, {c95[1]:+.2f})  d={desc['d']:+.2f}",
                  f"  TOST ±{100*MARGIN:.1f} p.p.: p={p_tost:.4f}   Wilcoxon: p={desc['wilcoxon_p']:.4f}",
                  f"  VEREDITO: {P2.verdict(desc, p_tost)}", ""]
    g = df.groupby("condition")
    lines += ["== Descritivo por condição (mediana entre sementes) ==",
              g[["primary_median_401_500", "mean_451_500", "n_resets_extra", "att_mass_mean"]].median().round(4).to_string(), ""]

    policy = _make_actor_evaluator()
    mech = []
    for s in SEEDS:
        tag = f"MNIST_{_alabel(eps)}_q0.5_td3_seed{s}_R{ROUNDS}"
        p0 = os.path.join(RAW, "actors", f"{tag}_step000.pt")
        p5 = os.path.join(RAW, "actors", f"{tag}_step500.pt")
        ob = os.path.join(RAW, "obs", f"{tag}.npy")
        if not all(os.path.exists(x) for x in (p0, p5, ob)):
            continue
        S = np.load(ob)[400:500]
        pi0, pi5 = policy(p0, S), policy(p5, S)
        mech.append({"seed": s, "drift": float(np.mean(np.abs(pi5 - pi0))),
                     "sd_estados": float(np.mean(pi5.std(axis=0)))})
    m = pd.DataFrame(mech)
    if len(m):
        m.to_csv(os.path.join(OUT, "mecanismo.csv"), index=False)
        lines += ["== Mecanismo (descritivo, sem teste) ==",
                  f"  σ_a = {SIGMA_A:.4f}; n={len(m)} runs td3",
                  f"  drift |π500−π0|: mediana={m['drift'].median():.4f} máx={m['drift'].max():.4f}",
                  f"  sd_estados de π500: mediana={m['sd_estados'].median():.4f} máx={m['sd_estados'].max():.4f}", ""]
    text = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "analise.txt"), "w").write(text)
    print(text)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    g = sub.add_parser("grade")
    g.add_argument("--eps", type=float, required=True)
    a = p.parse_args()
    if a.cmd == "sanity":
        sanity()
    else:
        grade(a.eps)


if __name__ == "__main__":
    main()
