"""
W1 rerun confirmatory analysis (results/w1_protocolo_independente_oficial/PLANO.md §5, as changed by
results/w1_rerun_global_eval/PLAN.md: the held-out series is the global model; fresh seeds).
Run ONCE, with the full grid (86/86), with the official venv:
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_w1.py

  primary   median HELD-OUT test accuracy over rounds 401-500 (from the `held_out` log)
  secondary mean held-out test accuracy over rounds 1-500 (AUC)
  H1        per dataset, LMP + EB pooled (n = 20): Δ = fixed − td3, paired by (attack, seed);
            TOST margins ±1 p.p. (MNIST) / ±3 p.p. (BloodMNIST); Holm over the 2 datasets for each
            test type; verdict: superiority/inferiority by the two-sided t test (CI95), else
            equivalence by TOST (CI90 inside the margin), else inconclusive
  H2        the same on the AUC 1-500, no correction
  descriptive  per-attack Δ, val−test gap, resets, attackers' weight mass, drift and sd_estados
               (deterministic actor at steps 0 and 500 on the states of rounds 401-500, as in B3.2),
               fedavg/none reference
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analisar as P2  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

OUT = "results/w1_rerun_global_eval"
RAW = f"{OUT}/raw"
SEEDS = {"MNIST": list(range(220, 230)), "BloodMNIST": list(range(230, 240))}
FEDAVG_SEEDS = {"MNIST": [220, 221, 222], "BloodMNIST": [230, 231, 232]}
CHECK_TOL = 0.001
MARGIN = {"MNIST": 1.0, "BloodMNIST": 3.0}
ATTACKS = ["LMP", "EB"]
ROUNDS = 500
ALPHA = 0.05


def _tag(d, att, cond, s):
    return f"{d}_{att}_q0.5_{cond}_seed{s}_R{ROUNDS}"


def load():
    rows, missing = [], []
    jobs = [(d, a, c, s) for d in SEEDS for a in ATTACKS for c in ("fixed", "td3") for s in SEEDS[d]]
    jobs += [(d, "none", "fedavg", s) for d in FEDAVG_SEEDS for s in FEDAVG_SEEDS[d]]
    for d, att, cond, s in jobs:
        f = f"{RAW}/{_tag(d, att, cond, s)}.json"
        if not os.path.exists(f):
            missing.append(f)
            continue
        r = json.load(open(f))
        if r.get("eval_target") != "global" or not r["global_checks"] or any(
                abs(c["eval_val_acc"] - c["logged_val_acc"]) > CHECK_TOL for c in r["global_checks"]):
            raise SystemExit(f"{f}: the held-out series is not verified as the global model (PLAN §4)")
        st = r["steps"][:ROUNDS]
        test = np.array([x["test_acc"] for x in r["held_out"][:ROUNDS]])
        val = np.array([x["acc"] for x in st])
        real = np.array([x["n_att_real"] for x in st])
        mass = np.array([np.nan if x["att_weight_mass"] is None else x["att_weight_mass"] for x in st])
        rs = [x["env_step"] for x in r["resets"][1:]]
        rows.append({"dataset": d, "attack": att, "condition": cond, "seed": s,
                     "primary": float(np.median(test[400:500])), "auc_1_500": float(test.mean()),
                     "frac_below_50_401_500": float(np.mean(test[400:500] < 0.5)),
                     "val_primary": float(np.median(val[400:500])),
                     "gap_val_minus_test": float(np.median(val[400:500]) - np.median(test[400:500])),
                     "resets": len(rs), "reset_401_500": any(400 <= x <= 500 for x in rs),
                     "att_mass": float(np.nanmean(mass[real > 0])) if (real > 0).any() else np.nan})
    return pd.DataFrame(rows), missing


def _stats(d, margin_pp):
    n, m, sd = len(d), d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    t95, t90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    return {"n": n, "delta_pp": 100 * m, "ci95": (100 * (m - t95 * se), 100 * (m + t95 * se)),
            "ci90": (100 * (m - t90 * se), 100 * (m + t90 * se)), "d": m / sd if sd > 0 else np.nan,
            "p_diff": float(stats.ttest_1samp(d, 0).pvalue),
            "p_tost": float(max(P2.tost_paired(d, margin_pp / 100))),
            "wilcoxon_p": P2.describe(d)["wilcoxon_p"]}


def family(df, metric, holm=True):
    res = {}
    for d in SEEDS:
        tab = df[df.dataset == d].pivot_table(index=["attack", "seed"], columns="condition", values=metric).dropna()
        res[d] = _stats((tab["fixed"] - tab["td3"]).to_numpy(), MARGIN[d])
    keys = list(res)
    adj_d = P2.holm(np.array([res[k]["p_diff"] for k in keys])) if holm else [res[k]["p_diff"] for k in keys]
    adj_t = P2.holm(np.array([res[k]["p_tost"] for k in keys])) if holm else [res[k]["p_tost"] for k in keys]
    for k, pd_, pt in zip(keys, adj_d, adj_t):
        r = res[k]
        r["p_diff_adj"], r["p_tost_adj"] = float(pd_), float(pt)
        if pd_ < ALPHA:
            r["verdict"] = "fixed BETTER (superiority of fixed)" if r["delta_pp"] > 0 else "td3 BETTER"
        elif pt < ALPHA:
            r["verdict"] = "EQUIVALENT"
        else:
            r["verdict"] = "INCONCLUSIVE"
    return res


def behaviour(df):
    from analisar_b21 import make_actor_evaluator
    policy = make_actor_evaluator()
    rows = []
    for _, r in df[df.condition == "td3"].iterrows():
        tag = _tag(r.dataset, r.attack, "td3", r.seed)
        p0, p5, ob = f"{RAW}/actors/{tag}_step000.pt", f"{RAW}/actors/{tag}_step500.pt", f"{RAW}/obs/{tag}.npy"
        if not all(os.path.exists(x) for x in (p0, p5, ob)):
            continue
        S = np.load(ob)[400:500]
        pi0, pi5 = policy(p0, S), policy(p5, S)
        rows.append({"dataset": r.dataset, "attack": r.attack, "seed": r.seed,
                     "drift": float(np.mean(np.abs(pi5 - pi0))), "sd_estados": float(np.mean(pi5.std(axis=0))),
                     "sd_estados_pi0": float(np.mean(pi0.std(axis=0)))})
    return pd.DataFrame(rows)


def main():
    df, missing = load()
    lines = [f"W1 analysis: {len(df)}/86 runs" + (f"; MISSING {len(missing)}" if missing else "")]
    if missing:
        lines += ["  " + m for m in missing] + ["INCOMPLETE GRID: no confirmatory verdict (PLANO §7)"]
        print("\n".join(lines))
        return
    df.to_csv(f"{OUT}/resumo_runs.csv", index=False)
    att = df[df.condition != "fedavg"]
    for metric, title, holm in (("primary", "H1 (primary: median held-out acc 401-500), Holm over 2 datasets", True),
                                ("auc_1_500", "H2 (secondary: mean held-out acc 1-500), no correction", False)):
        lines.append(f"\n== {title}; Δ = fixed − td3 ==")
        for d, r in family(att, metric, holm).items():
            lines.append(f"  {d}: n={r['n']} Δ={r['delta_pp']:+.2f} p.p. CI90=({r['ci90'][0]:+.2f}; {r['ci90'][1]:+.2f}) "
                         f"CI95=({r['ci95'][0]:+.2f}; {r['ci95'][1]:+.2f}) d={r['d']:+.2f} | TOST ±{MARGIN[d]:.0f} p={r['p_tost']:.4f} "
                         f"(adj {r['p_tost_adj']:.4f}) | t p={r['p_diff']:.4f} (adj {r['p_diff_adj']:.4f}) "
                         f"| Wilcoxon p={r['wilcoxon_p']:.4f} -> {r['verdict']}")
    lines.append("\n== Descriptive: per attack, Δ = fixed − td3 on the primary ==")
    for d in SEEDS:
        for a in ATTACKS:
            tab = att[(att.dataset == d) & (att.attack == a)].pivot_table(index="seed", columns="condition",
                                                                           values="primary").dropna()
            ds = P2.describe((tab["fixed"] - tab["td3"]).to_numpy())
            lines.append(f"  {d} {a}: n={ds['n']} Δ={ds['delta_pp']:+.2f} p.p. CI95=({ds['ci95_pp'][0]:+.2f}; "
                         f"{ds['ci95_pp'][1]:+.2f}) Wilcoxon p={ds['wilcoxon_p']:.4f}")
    lines.append("\n== Descriptive: means per dataset × attack × condition ==")
    g = df.groupby(["dataset", "attack", "condition"])
    lines.append(g.agg(primary=("primary", "mean"), auc=("auc_1_500", "mean"), val_primary=("val_primary", "mean"),
                       gap_val_test=("gap_val_minus_test", "mean"), resets=("resets", "mean"),
                       frac_reset_401_500=("reset_401_500", "mean"), att_mass=("att_mass", "median"),
                       frac_below_50_401_500=("frac_below_50_401_500", "mean"))
                 .round(4).to_string())
    b = behaviour(df)
    if len(b):
        b.to_csv(f"{OUT}/mecanismo.csv", index=False)
        lines.append("\n== Descriptive: TD3 behaviour (σ_a = 0.0475) ==")
        lines.append(b.groupby(["dataset", "attack"])[["drift", "sd_estados", "sd_estados_pi0"]].median()
                     .round(4).to_string())
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/analise.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
