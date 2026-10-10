"""
A0 item (d): summary table of the three TD3 regimes in the official code,
for a P2 figure. Plan in results/a0_analises/PLANO.md.
Only reads already saved checkpoints and states.

Regimes:
  published   B2.1 (seeds 105-114), lr 1e-5, learning_starts 100
  B2.3        steelman lr 1e-3, raw reward (seeds 100-104)
  B2.3b       steelman lr 1e-4, normalized reward (seeds 100-104)
For each td3 run and each checkpoint t (0, 50, ..., 500), over the states of
rounds 401-500 of the run itself:
  drift_t     = mean |π_t(s) − π_0(s)|
  sd_estados  = mean over the 5 dims of the standard deviation of π_t(s) across states
Uses the intermediate checkpoints available locally (the repository only versions steps 0 and 500).

Usage (official venv):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/a0_regimes_td3.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analisar_b21 as A  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REGIMES = [
    ("published (B2.1)", "results/b21_replicacao_oficial/raw", range(105, 115)),
    ("B2.3: lr 1e-3, raw reward", "results/b23_steelman_oficial/raw", range(100, 105)),
    ("B2.3b: lr 1e-4, normalized reward", "results/b23b_steelman_normalizado/raw", range(100, 105)),
]
OUT = "results/a0_analises"


def main():
    policy = A.make_actor_evaluator()
    rows = []
    for name, raw, seeds in REGIMES:
        for att in ["LMP", "EB"]:
            for s in seeds:
                tag = f"MNIST_{att}_q0.5_td3_seed{s}_R500"
                obs_p = os.path.join(raw, "obs", f"{tag}.npy")
                p0 = os.path.join(raw, "actors", f"{tag}_step000.pt")
                if not (os.path.exists(obs_p) and os.path.exists(p0)):
                    continue
                S = np.load(obs_p)[400:500]
                pi0 = policy(p0, S)
                for t in range(0, 501, 50):
                    pt = os.path.join(raw, "actors", f"{tag}_step{t:03d}.pt")
                    if not os.path.exists(pt):
                        continue
                    pi = policy(pt, S)
                    rows.append({"regime": name, "attack": att, "seed": s, "step": t,
                                 "drift": float(np.mean(np.abs(pi - pi0))),
                                 "sd_estados": float(np.mean(pi.std(axis=0)))})
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/regimes_td3_por_checkpoint.csv", index=False)
    med = df.groupby(["regime", "step"])[["drift", "sd_estados"]].median().round(4).unstack("regime")
    n = df.groupby("regime")["seed"].nunique() * 2
    text = ("A0 (d) — three TD3 regimes: median across runs per checkpoint (σ_a = 0.0475)\n"
            + "runs per regime: " + ", ".join(f"{k}: {v}" for k, v in n.items()) + "\n\n" + med.to_string() + "\n")
    open(f"{OUT}/analise_regimes_td3.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
