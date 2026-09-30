"""
C.0 (references/roadmap_tese_gradf_v2.md §5): espaço restante por célula no
framework próprio. Descritivo, sem hipótese a testar.

Para cada célula (alpha x ataque) da grade da ablação (MNIST, 10 clientes,
2 bizantinos, 15 rodadas, root 100, sementes 42-51):
  teto   = FedAvg SEM ataque (mesmo regime, mesma métrica global_accuracy)
  melhor = melhor média entre {fixed, sr_only, td3_ref (AdaAggRL)}
  gap    = teto - melhor, pareado por semente (média, IC95)
Contexto: melhor regra fixa do exp9 (root 100) e cosserver_only (exploratório).

O único cálculo novo é o teto (30 runs de regressão logística, só CPU).
As demais acurácias vêm de:
  results/frente1_ablacao_adaaggrl/grade_combined_raw.csv
  results/tables/exp9_dominance_grid_10seeds_ALL_root100_raw.csv

Uso (CPU, baixa prioridade, sem GPU):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.c0_espaco_restante
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/c0_espaco_restante"
SEEDS = list(range(42, 52))
ALPHAS = [0.5, 0.1, 0.05]
ABL = "results/frente1_ablacao_adaaggrl/grade_combined_raw.csv"
EXP9 = "results/tables/exp9_dominance_grid_10seeds_ALL_root100_raw.csv"
CANDIDATES = ["fixed", "sr_only", "td3_ref"]


def ceiling() -> pd.DataFrame:
    """FedAvg sem ataque, mesmo regime da ablação."""
    path = os.path.join(OUT, "teto_fedavg_sem_ataque_raw.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    from src.experiments.exp9_dominance_grid import _split_name
    from src.fl.attacked_learner import AttackedFederatedLearner
    from src.utils.data_loader import load_dataset_participants

    rows = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            participants, root = load_dataset_participants(
                "mnist", _split_name(alpha), 10, root_size=100, root_seed=seed,
            )
            learner = AttackedFederatedLearner(
                n_rounds=15, n_classes=10, attack_type="none", byzantine_ids=[0, 1],
                seed=seed, aggregation="fedavg",
            )
            acc = learner.train(participants, root_data=root, verbose=False)[-1].global_accuracy
            rows.append({"alpha": alpha, "seed": seed, "accuracy": acc})
            print(f"teto alpha={alpha} seed={seed} acc={acc:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def ci95(x: np.ndarray):
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def main():
    os.makedirs(OUT, exist_ok=True)
    teto = ceiling().rename(columns={"accuracy": "teto"})
    abl = pd.read_csv(ABL)
    ex9 = pd.read_csv(EXP9)

    rows = []
    for (alpha, attack), g in abl.groupby(["alpha", "attack_type"]):
        wide = g.pivot_table(index="seed", columns="mode", values="accuracy")
        means = wide[CANDIDATES].mean()
        best = means.idxmax()
        t = teto[teto.alpha == alpha].set_index("seed")["teto"]
        gap = (t - wide[best]).dropna().to_numpy()
        m, lo, hi = ci95(gap)
        fx = ex9[(ex9.alpha == alpha) & (ex9.attack_type == attack)].groupby("strategy")["accuracy"].mean()
        rows.append({
            "alpha": alpha, "attack_type": attack,
            "teto_fedavg_sem_ataque": t.mean(),
            "melhor_metodo": best, "melhor_acc": means[best],
            "gap_pp": 100 * m, "gap_ic95_lo_pp": 100 * lo, "gap_ic95_hi_pp": 100 * hi,
            **{f"acc_{c}": means[c] for c in CANDIDATES},
            "acc_cosserver_only": wide["cosserver_only"].mean() if "cosserver_only" in wide else np.nan,
            "melhor_regra_fixa_exp9": fx.idxmax(), "acc_melhor_regra_fixa_exp9": fx.max(),
        })
    res = pd.DataFrame(rows).sort_values("gap_pp", ascending=False)
    res.to_csv(os.path.join(OUT, "espaco_restante_por_celula.csv"), index=False)

    lines = ["C.0 — espaço restante (teto = FedAvg sem ataque; melhor entre fixed, sr_only, AdaAggRL)", ""]
    lines.append("Teto por alpha (média de 10 sementes): " + ", ".join(
        f"α={a}: {teto[teto.alpha == a].teto.mean():.4f}" for a in ALPHAS))
    lines += ["", res[["alpha", "attack_type", "melhor_metodo", "melhor_acc", "teto_fedavg_sem_ataque",
                       "gap_pp", "gap_ic95_lo_pp", "gap_ic95_hi_pp", "melhor_regra_fixa_exp9",
                       "acc_melhor_regra_fixa_exp9"]].round(4).to_string(index=False), ""]
    for thr in (1.0, 2.0, 5.0):
        lines.append(f"células com gap > {thr:.0f} p.p.: {(res.gap_pp > thr).sum()}/21")
    lines.append(f"células com IC95 do gap acima de 0 (espaço estatisticamente > 0): {(res.gap_ic95_lo_pp > 0).sum()}/21")
    text = "\n".join(lines)
    open(os.path.join(OUT, "analise.txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
