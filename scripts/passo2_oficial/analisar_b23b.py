"""
B2.3b analysis (last steelman), following results/b23b_steelman_normalizado/PLANO.md.
Written before any result existed. Exploratory (already-used seeds 100-104).
Reuses `analisar_b23.py` and `analisar_b21.py` without changing them.

Pairing by (attack, seed):
  fixed, td3 (official): results/frente1_passo2_oficial/raw/
  steelman B2.3:         results/b23_steelman_oficial/raw/
  steelman B2.3b:        results/b23b_steelman_normalizado/raw/

Primary: median 401-500 (the same as B2.1/B2.3). Secondary: mean 451-500.
  C1 b23b − fixed   <- defines the definitive Gate B-a
  C2 b23b − official td3
  C3 b23b − steelman B2.3
Mechanism (deterministic policy rebuilt from the checkpoints):
  drift_500    = mean |π_500(s) − π_0(s)|, states of rounds 401-500
  sd_estados   = mean over the 5 dims of the standard deviation of π_500(s) across those
                 states  <- constancy statistic (S_swap saturates with the tanh)
  S_swap       = reported, no criterion
Criterion (PLANO §5): "learns and beats" = C1 with Δ > 0 and p < 0.05
  AND median drift_500 > σ_a AND median sd_estados > σ_a  (σ_a = 0.0475).

Usage:
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_b23b.py
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analisar_b21 as B21A  # noqa: E402
import analisar_b23 as B23A  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ATTACKS, SEEDS = B23A.ATTACKS, B23A.SEEDS
SIGMA_A = B21A.SIGMA_A
B23_RAW = "results/b23_steelman_oficial/raw"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="results/b23b_steelman_normalizado/raw")
    p.add_argument("--out", default="results/b23b_steelman_normalizado/analise.txt")
    a = p.parse_args()

    rows = []
    for att in ATTACKS:
        for s in SEEDS:
            for cond, path in [("fixed", os.path.join(B23A.P2_RAW, B23A._tag(att, "fixed", s) + ".json")),
                               ("td3", os.path.join(B23A.P2_RAW, B23A._tag(att, "td3", s) + ".json")),
                               ("steelman_b23", os.path.join(B23_RAW, B23A._tag(att, "td3", s) + ".json")),
                               ("steelman_b23b", os.path.join(a.raw_dir, B23A._tag(att, "td3", s) + ".json"))]:
                if os.path.exists(path):
                    rows.append({"attack": att, "seed": s, "condition": cond, **B23A._metrics(path)})
    df = pd.DataFrame(rows)
    n_st = int((df["condition"] == "steelman_b23b").sum())
    lines = [f"B2.3b (last steelman) — runs loaded: {n_st} (expected 10); paired with Step 2 and B2.3", ""]

    tab = df.pivot_table(index=["attack", "seed"], columns="condition", values="median_401_500")
    lines += ["Primary (median 401–500):", tab.round(4).to_string(), ""]
    c1 = B23A.compare(tab, "steelman_b23b", "fixed", "C1 [primary]", lines)
    B23A.compare(tab, "steelman_b23b", "td3", "C2 [primary]", lines)
    B23A.compare(tab, "steelman_b23b", "steelman_b23", "C3 [primary]", lines)
    tab2 = df.pivot_table(index=["attack", "seed"], columns="condition", values="mean_451_500")
    B23A.compare(tab2, "steelman_b23b", "fixed", "C1 [secondary: mean 451–500]", lines)
    lines += ["Extra resets (mean per attack × condition):",
              df.groupby(["attack", "condition"])["n_resets_extra"].mean().round(2).to_string(), ""]

    policy = B21A.make_actor_evaluator()
    mech, curve = [], []
    for att in ATTACKS:
        other = "LMP" if att == "EB" else "EB"
        for s in SEEDS:
            tag, otag = B23A._tag(att, "td3", s), B23A._tag(other, "td3", s)
            act = lambda t, T=tag: os.path.join(a.raw_dir, "actors", f"{T}_step{t:03d}.pt")  # noqa: E731
            obs_p, oobs_p = (os.path.join(a.raw_dir, "obs", f"{x}.npy") for x in (tag, otag))
            if not (os.path.exists(act(0)) and os.path.exists(act(500)) and os.path.exists(obs_p)):
                continue
            S = np.load(obs_p)[400:500]
            pi0 = policy(act(0), S)
            for t in range(50, 501, 50):
                if os.path.exists(act(t)):
                    curve.append({"attack": att, "seed": s, "step": t,
                                  "drift": float(np.mean(np.abs(policy(act(t), S) - pi0)))})
            pi5 = policy(act(500), S)
            row = {"attack": att, "seed": s,
                   "drift_500": float(np.mean(np.abs(pi5 - pi0))),
                   "sd_estados": float(np.mean(pi5.std(axis=0))),
                   "acao_media": np.round(pi5.mean(axis=0), 3).tolist()}
            if os.path.exists(oobs_p):
                row["S_swap"] = float(np.mean(np.abs(pi5 - policy(act(500), np.load(oobs_p)[400:500]))))
            mech.append(row)
    m, cv = pd.DataFrame(mech), pd.DataFrame(curve)
    if len(m):
        lines += [f"== B2.3b mechanism (σ_a = {SIGMA_A:.4f}) ==", m.round(4).to_string(index=False),
                  f"  median drift_500={m['drift_500'].median():.4f}  median sd_estados={m['sd_estados'].median():.4f}"
                  + (f"  median S_swap={m['S_swap'].median():.4f}" if "S_swap" in m else ""),
                  "  drift over training (median across runs):",
                  cv.groupby("step")["drift"].median().round(4).to_string(), ""]

    beats = c1["delta_pp"] > 0 and c1["wilcoxon_p"] < 0.05
    moved = bool(len(m)) and m["drift_500"].median() > SIGMA_A
    state_dep = bool(len(m)) and m["sd_estados"].median() > SIGMA_A
    lines += ["== Definitive Gate B-a (PLANO §5 criterion) ==",
              f"  C1 b23b > fixed (Δ>0 and p<0.05): {'YES' if beats else 'NO'}",
              f"  policy moved (drift_500 > σ_a): {'YES' if moved else 'NO'}",
              f"  policy depends on the state (sd_estados > σ_a): {'YES' if state_dep else 'NO'}",
              "  VERDICT: " + ("the steelman LEARNS AND BEATS the fixed action -> confirm in B2.3c (seeds 115–124)"
                               if (beats and moved and state_dep)
                               else "the steelman does NOT beat the fixed action -> B-a definitive; configuration search ends"
                               if not beats
                               else "beats the fixed action without a state-dependent policy (investigate; does not trigger B2.3c)")]

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
