"""
A0 item (d) (references/roadmap_tese_gradf_v4.md §4.2): tabela-síntese dos três
regimes do TD3 no código oficial, para figura do P2. Plano em
results/a0_analises/PLANO.md. Só lê checkpoints e estados já gravados.

Regimes:
  publicado   B2.1 (sementes 105-114), lr 1e-5, learning_starts 100
  B2.3        steelman lr 1e-3, recompensa bruta (sementes 100-104)
  B2.3b       steelman lr 1e-4, recompensa normalizada (sementes 100-104)
Para cada run td3 e cada checkpoint t (0, 50, ..., 500), sobre os estados das
rodadas 401-500 do próprio run:
  drift_t     = média |π_t(s) − π_0(s)|
  sd_estados  = média sobre as 5 dims do desvio-padrão de π_t(s) entre estados
Usa os checkpoints intermediários disponíveis localmente (nem todos estão no git).

Uso (venv oficial):
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
    ("publicado (B2.1)", "results/b21_replicacao_oficial/raw", range(105, 115)),
    ("B2.3: lr 1e-3, recompensa bruta", "results/b23_steelman_oficial/raw", range(100, 105)),
    ("B2.3b: lr 1e-4, recompensa normalizada", "results/b23b_steelman_normalizado/raw", range(100, 105)),
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
    text = ("A0 (d) — três regimes do TD3: mediana entre runs por checkpoint (σ_a = 0,0475)\n"
            + "runs por regime: " + ", ".join(f"{k}: {v}" for k, v in n.items()) + "\n\n" + med.to_string() + "\n")
    open(f"{OUT}/analise_regimes_td3.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
