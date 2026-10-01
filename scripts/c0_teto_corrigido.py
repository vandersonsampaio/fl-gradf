"""
C.0 com teto corrigido (complementa scripts/c0_espaco_restante.py; não altera
os arquivos dele).

Motivo: no C.0 original, o teto (FedAvg sem ataque, 15 rodadas) ficou abaixo de
métodos sob ataque em alpha <= 0,1 (Krum ~0,895 contra teto ~0,854), indicando
que FedAvg não converge em 15 rodadas com heterogeneidade extrema. O teto passa
a ser a MELHOR das 5 regras sem ataque (FedAvg, FLTrust, Krum, Median,
Trimmed-Mean), no mesmo regime e com a mesma construção do exp9
(byzantine_ids=[0,1], 15 rodadas, root 100, sementes 42-51).

Dois gaps por célula (pareados por semente, média e IC95):
  gap_esqueleto = teto - melhor entre {fixed, sr_only, td3_ref (AdaAggRL)}
  gap_global    = teto - melhor entre esses E as 4 regras fixas do exp9 sob ataque
O gap_global responde à pergunta do Portão C0: há espaço além do que já existe?

Uso (CPU, baixa prioridade, sem GPU):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.c0_teto_corrigido
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/c0_espaco_restante"
SEEDS = list(range(42, 52))
ALPHAS = [0.5, 0.1, 0.05]
RULES = ["fltrust", "krum", "median", "trimmed_mean"]
ABL = "results/frente1_ablacao_adaaggrl/grade_combined_raw.csv"
EXP9 = "results/tables/exp9_dominance_grid_10seeds_ALL_root100_raw.csv"
FEDAVG_NONE = os.path.join(OUT, "teto_fedavg_sem_ataque_raw.csv")
SKELETON = ["fixed", "sr_only", "td3_ref"]


def rules_no_attack() -> pd.DataFrame:
    path = os.path.join(OUT, "teto_regras_sem_ataque_raw.csv")
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
            for rule in RULES:
                learner = AttackedFederatedLearner(
                    n_rounds=15, aggregation=rule, n_classes=10, attack_type="none",
                    byzantine_ids=[0, 1], seed=seed,
                )
                acc = learner.train(participants, root_data=root, verbose=False)[-1].global_accuracy
                rows.append({"alpha": alpha, "seed": seed, "rule": rule, "accuracy": acc})
                print(f"sem ataque alpha={alpha} seed={seed} {rule} acc={acc:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def ci95(x):
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def main():
    fa = pd.read_csv(FEDAVG_NONE).assign(rule="fedavg")
    na = pd.concat([fa, rules_no_attack()])
    ceil_tab = na.groupby(["alpha", "rule"])["accuracy"].mean().unstack()
    ceil_rule = ceil_tab.idxmax(axis=1)  # regra-teto por alpha

    abl = pd.read_csv(ABL)
    ex9 = pd.read_csv(EXP9)
    rows = []
    for (alpha, attack), g in abl.groupby(["alpha", "attack_type"]):
        wide = g.pivot_table(index="seed", columns="mode", values="accuracy")[SKELETON]
        f9 = ex9[(ex9.alpha == alpha) & (ex9.attack_type == attack)].pivot_table(
            index="seed", columns="strategy", values="accuracy")
        allm = wide.join(f9, how="inner")
        best_sk = wide.mean().idxmax()
        best_all = allm.mean().idxmax()
        rule = ceil_rule[alpha]
        t = na[(na.alpha == alpha) & (na.rule == rule)].set_index("seed")["accuracy"]
        g_sk = (t - wide[best_sk]).dropna().to_numpy()
        g_all = (t - allm[best_all]).dropna().to_numpy()
        msk, lsk, hsk = ci95(g_sk)
        mal, lal, hal = ci95(g_all)
        rows.append({
            "alpha": alpha, "attack_type": attack, "regra_teto": rule, "teto": t.mean(),
            "melhor_esqueleto": best_sk, "acc_melhor_esqueleto": wide[best_sk].mean(),
            "gap_esqueleto_pp": 100 * msk, "gap_esq_ic95_lo": 100 * lsk, "gap_esq_ic95_hi": 100 * hsk,
            "melhor_global": best_all, "acc_melhor_global": allm[best_all].mean(),
            "gap_global_pp": 100 * mal, "gap_glob_ic95_lo": 100 * lal, "gap_glob_ic95_hi": 100 * hal,
        })
    res = pd.DataFrame(rows).sort_values("gap_global_pp", ascending=False)
    res.to_csv(os.path.join(OUT, "espaco_restante_teto_corrigido.csv"), index=False)

    lines = ["C.0 — teto corrigido = melhor das 5 regras SEM ataque (mesmo regime do exp9)", "",
             "Acurácia sem ataque por alpha × regra (média de 10 sementes):",
             ceil_tab.round(4).to_string(), "",
             "Regra-teto por alpha: " + ", ".join(f"α={a}: {r}" for a, r in ceil_rule.items()), "",
             res[["alpha", "attack_type", "teto", "melhor_esqueleto", "gap_esqueleto_pp",
                  "melhor_global", "acc_melhor_global", "gap_global_pp",
                  "gap_glob_ic95_lo", "gap_glob_ic95_hi"]].round(3).to_string(index=False), ""]
    for col, lab in [("gap_esqueleto_pp", "esqueleto (fixed/sr_only/AdaAggRL)"),
                     ("gap_global_pp", "global (inclui regras fixas)")]:
        lines.append(f"[{lab}] gap > 1 p.p.: {(res[col] > 1).sum()}/21 | > 2 p.p.: {(res[col] > 2).sum()}/21 | "
                     f"> 5 p.p.: {(res[col] > 5).sum()}/21")
    lines.append(f"[global] IC95 do gap acima de 0: {(res.gap_glob_ic95_lo > 0).sum()}/21 | "
                 f"gap negativo (algum método sob ataque supera o teto): {(res.gap_global_pp < 0).sum()}/21")
    text = "\n".join(lines)
    open(os.path.join(OUT, "analise_teto_corrigido.txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
