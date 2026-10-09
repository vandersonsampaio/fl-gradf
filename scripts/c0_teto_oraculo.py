"""
C.0 with an ORACLE ceiling. Complements
scripts/c0_espaco_restante.py and scripts/c0_teto_corrigido.py without changing them.

Oracle ceiling: FedAvg without attack aggregating only the 8 honest clients (indices
2-9 of the participant list; the Byzantine ones are indices 0 and 1 in
byzantine_ids), with the SAME partitions as C.0 (nothing is repartitioned).
Metric identical to C.0's: accuracy of the final global model averaged over the
10 clients' X_test.

Criterion (results/c0_espaco_restante/NOTAS.md §5, frozen with a hash before this script ran):
  cell with headroom = global gap > 2 p.p. with the whole CI95 above 0,
  over the 19 valid cells (fltrust_aligned at alpha 0.05 and 0.1 excluded).

Usage (CPU, low priority, no GPU):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.c0_teto_oraculo
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/c0_espaco_restante"
SEEDS = list(range(42, 52))
ALPHAS = [0.5, 0.1, 0.05]
BYZ = [0, 1]
ABL = "results/frente1_ablacao_adaaggrl/grade_combined_raw.csv"
EXP9 = "results/tables/exp9_dominance_grid_10seeds_ALL_root100_raw.csv"
SKELETON = ["fixed", "sr_only", "td3_ref"]
EXCLUDED = {(0.05, "fltrust_aligned"), (0.1, "fltrust_aligned")}
THRESH_PP = 2.0


def oracle_ceiling() -> pd.DataFrame:
    path = os.path.join(OUT, "teto_oraculo_raw.csv")
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
            honest = [p for i, p in enumerate(participants) if i not in BYZ]
            learner = AttackedFederatedLearner(
                n_rounds=15, n_classes=10, attack_type="none", byzantine_ids=[],
                seed=seed, aggregation="fedavg",
            )
            acc8 = learner.train(honest, root_data=root, verbose=False)[-1].global_accuracy
            model = learner._make_model(participants[0].n_features)
            model.set_params(learner._global_params)
            acc10 = float(np.mean([model.accuracy(p.X_test, p.y_test) for p in participants]))
            rows.append({"alpha": alpha, "seed": seed, "accuracy": acc10, "accuracy_8_honestos": acc8})
            print(f"oracle alpha={alpha} seed={seed} acc(10 test sets)={acc10:.4f} acc(8)={acc8:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def ci95(x):
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def main():
    ora = oracle_ceiling()
    fa = pd.read_csv(os.path.join(OUT, "teto_fedavg_sem_ataque_raw.csv"))
    abl = pd.read_csv(ABL)
    ex9 = pd.read_csv(EXP9)

    rows = []
    for (alpha, attack), g in abl.groupby(["alpha", "attack_type"]):
        if (alpha, attack) in EXCLUDED:
            continue
        wide = g.pivot_table(index="seed", columns="mode", values="accuracy")[SKELETON]
        f9 = ex9[(ex9.alpha == alpha) & (ex9.attack_type == attack)].pivot_table(
            index="seed", columns="strategy", values="accuracy")
        allm = wide.join(f9, how="inner")
        best_sk, best_all = wide.mean().idxmax(), allm.mean().idxmax()
        t = ora[ora.alpha == alpha].set_index("seed")["accuracy"]
        msk, lsk, hsk = ci95((t - wide[best_sk]).dropna().to_numpy())
        mal, lal, hal = ci95((t - allm[best_all]).dropna().to_numpy())
        rows.append({
            "alpha": alpha, "attack_type": attack, "teto_oraculo": t.mean(),
            "melhor_esqueleto": best_sk, "acc_melhor_esqueleto": wide[best_sk].mean(),
            "gap_esqueleto_pp": 100 * msk, "gap_esq_ic95_lo": 100 * lsk, "gap_esq_ic95_hi": 100 * hsk,
            "melhor_global": best_all, "acc_melhor_global": allm[best_all].mean(),
            "gap_global_pp": 100 * mal, "gap_glob_ic95_lo": 100 * lal, "gap_glob_ic95_hi": 100 * hal,
            "com_espaco": bool(100 * mal > THRESH_PP and 100 * lal > 0),
            "esqueleto_perde_p_regra_fixa": bool(best_all not in SKELETON),
        })
    res = pd.DataFrame(rows).sort_values("gap_global_pp", ascending=False)
    res.to_csv(os.path.join(OUT, "espaco_restante_teto_oraculo.csv"), index=False)

    cmp_ = ora.groupby("alpha")[["accuracy", "accuracy_8_honestos"]].mean().join(
        fa.groupby("alpha")["accuracy"].mean().rename("fedavg_10_sem_ataque"))
    lines = ["C.0 — ORACLE ceiling = FedAvg without attack with the 8 honest clients only (same partitions; metric over the 10 test sets)", "",
             "Ceiling per alpha (mean of 10 seeds):", cmp_.round(4).to_string(), "",
             res[["alpha", "attack_type", "teto_oraculo", "melhor_esqueleto", "gap_esqueleto_pp",
                  "melhor_global", "acc_melhor_global", "gap_global_pp", "gap_glob_ic95_lo",
                  "gap_glob_ic95_hi", "com_espaco"]].round(3).to_string(index=False), "",
             f"Criterion (NOTAS.md §5): global gap > {THRESH_PP:.0f} p.p. with CI95 > 0, 19 valid cells",
             f"  cells WITH headroom: {int(res.com_espaco.sum())}/{len(res)}",
             "    " + "; ".join(f"{r.attack_type} α={r.alpha}" for r in res[res.com_espaco].itertuples()),
             f"  cells WITHOUT headroom: {int((~res.com_espaco).sum())}/{len(res)}",
             "    " + "; ".join(f"{r.attack_type} α={r.alpha}" for r in res[~res.com_espaco].itertuples()),
             f"  skeleton loses to a fixed rule in: " + "; ".join(
                 f"{r.attack_type} α={r.alpha} ({r.melhor_global})" for r in res[res.esqueleto_perde_p_regra_fixa].itertuples())]
    text = "\n".join(lines)
    open(os.path.join(OUT, "analise_teto_oraculo.txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
