"""
Experiment 2: GRADF's robustness under different attack types and byzantine
client fractions, measuring detection rate and accuracy retention. Aggregates
multiple seeds with mean/std/CI95 (src.utils.stats) into a table that holds
up under conference review.

Decision A5: `accuracy_retention` originally only compared GRADF against
FedAvg WITH NO DEFENSE — `retention > 1.0` only said "better than doing
nothing," not "competitive with the state of the art." We now report BOTH
metrics side by side: `accuracy_retention` (vs. FedAvg, kept for
compatibility with already-published tables) and
`accuracy_retention_vs_best_fixed` (vs. the best of FLTrust/Median/
Trimmed-Mean in THAT attack_type x fraction x seed combination) — the
comparison that actually matters for judging whether GRADF competes with
classical defenses, not just whether it beats no defense at all.

Each (attack_type, byzantine_fraction, seed) combination uses an
`AuditTrail` with its own path, cleared before training: `XAIExplainer`/
`AuditTrail` default to the same file (`results/audit_trail.jsonl`) across
every instance — without a per-run path, `.load()` would also read records
from earlier combinations (same `hospital_id` values like
`client_0..client_9`, different attack types) and contaminate the
`detection_rate` calculation. Per-run files live in `results/audit_trail/`
and are reused by `exp8_xai_examples.py`.
"""

import argparse
import os
import time
from typing import List, Optional

import pandas as pd

from src.experiments.exp1_baseline import train_gradf_models
from src.fl.attacked_learner import AttackedFederatedLearner
from src.fl.gradf_learner import GRADFFederatedLearner
from src.utils.data_loader import load_dataset_participants
from src.utils.logger import get_logger
from src.utils.metrics import accuracy_retention, detection_rate
from src.utils.stats import run_over_seeds, summarize
from src.utils.visualization import plot_bar_comparison
from src.xai.audit_trail import AuditTrail
from src.xai.explanation_generator import XAIExplainer

logger = get_logger(__name__)

AUDIT_TRAIL_DIR = "results/audit_trail"
FIXED_DEFENSE_STRATEGIES = ["fltrust", "median", "trimmed_mean"]


def run_robustness_experiment(
    dataset: str = "mnist",
    split: str = "non_iid",
    n_clients: int = 10,
    n_rounds: int = 15,
    attack_types: Optional[List[str]] = None,
    byzantine_fractions: Optional[List[float]] = None,
    n_classes: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    attack_types = attack_types or ["sign_flipping", "gaussian_noise", "label_flipping", "poisoning"]
    byzantine_fractions = byzantine_fractions or [0.1, 0.2, 0.3]

    participants, root_data = load_dataset_participants(dataset, split, n_clients)

    rows = []
    for attack_type in attack_types:
        logger.info("Pretraining classifier/selector for '%s' (seed=%d)...", attack_type, seed)
        pretrain_start = time.perf_counter()
        classifier, selector, attack_labels, _oracle_map = train_gradf_models(
            participants, attack_type, byzantine_fraction=0.2, n_classes=n_classes, seed=seed
        )
        pretrain_seconds = time.perf_counter() - pretrain_start

        for frac in byzantine_fractions:
            byzantine_ids = list(range(max(1, int(n_clients * frac))))

            baseline = AttackedFederatedLearner(
                n_rounds=n_rounds, aggregation="fedavg", n_classes=n_classes,
                attack_type=attack_type, byzantine_ids=byzantine_ids, seed=seed,
            )
            baseline_start = time.perf_counter()
            baseline_acc = baseline.train(participants, root_data=root_data, verbose=False)[-1].global_accuracy
            baseline_wall_clock_seconds = time.perf_counter() - baseline_start

            # Best fixed defense in THIS combination — a stricter comparison
            # baseline for accuracy_retention than "no defense."
            fixed_accuracies = {}
            for strategy in FIXED_DEFENSE_STRATEGIES:
                fixed_learner = AttackedFederatedLearner(
                    n_rounds=n_rounds, aggregation=strategy, n_classes=n_classes,
                    attack_type=attack_type, byzantine_ids=byzantine_ids, seed=seed,
                )
                fixed_accuracies[strategy] = fixed_learner.train(
                    participants, root_data=root_data, verbose=False
                )[-1].global_accuracy
            best_fixed_strategy = max(fixed_accuracies, key=fixed_accuracies.get)
            best_fixed_acc = fixed_accuracies[best_fixed_strategy]

            audit_path = os.path.join(AUDIT_TRAIL_DIR, f"exp2_{attack_type}_{frac}_{seed}.jsonl")
            audit_trail = AuditTrail(path=audit_path)
            audit_trail.clear()

            gradf = GRADFFederatedLearner(
                n_rounds=n_rounds, aggregation="fedavg", n_classes=n_classes,
                attack_type=attack_type, byzantine_ids=byzantine_ids, seed=seed,
                classifier=classifier, selector=selector, attack_labels=attack_labels,
                explainer=XAIExplainer(audit_trail=audit_trail),
            )
            gradf_start = time.perf_counter()
            gradf_acc = gradf.train(participants, root_data=root_data, verbose=False)[-1].global_accuracy
            gradf_wall_clock_seconds = time.perf_counter() - gradf_start

            records = audit_trail.load()
            byz_ids_set = {f"client_{i}" for i in byzantine_ids}
            y_true = [r["hospital_id"] in byz_ids_set for r in records]
            y_pred = [r["predicted_attack_type"] != "none" for r in records]

            row = {
                "attack_type": attack_type,
                "byzantine_fraction": frac,
                "fedavg_accuracy": baseline_acc,
                "best_fixed_strategy": best_fixed_strategy,
                "best_fixed_accuracy": best_fixed_acc,
                "gradf_accuracy": gradf_acc,
                "accuracy_retention": accuracy_retention(baseline_acc, gradf_acc),
                "accuracy_retention_vs_best_fixed": accuracy_retention(best_fixed_acc, gradf_acc),
                "detection_rate": detection_rate(y_true, y_pred),
                "pretrain_seconds": pretrain_seconds,
                "baseline_wall_clock_seconds": baseline_wall_clock_seconds,
                "gradf_wall_clock_seconds": gradf_wall_clock_seconds,
            }
            rows.append(row)
            logger.info(
                "%s frac=%.1f: fedavg=%.3f gradf=%.3f det_rate=%.2f wall_clock(fedavg/gradf)=%.1fs/%.1fs",
                attack_type, frac, baseline_acc, gradf_acc, row["detection_rate"],
                baseline_wall_clock_seconds, gradf_wall_clock_seconds,
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--split", default="non_iid", choices=["iid", "non_iid"])
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--n_rounds", type=int, default=15)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--attack_types", nargs="+", default=None)
    parser.add_argument("--byzantine_fractions", type=float, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    raw = run_over_seeds(
        run_robustness_experiment,
        seeds=args.seeds,
        dataset=args.dataset, split=args.split, n_clients=args.n_clients,
        n_rounds=args.n_rounds, n_classes=args.n_classes,
        attack_types=args.attack_types, byzantine_fractions=args.byzantine_fractions,
    )

    os.makedirs("results/tables", exist_ok=True)
    raw.to_csv("results/tables/exp2_robustness_raw.csv", index=False)
    print("=== raw (per seed) ===")
    print(raw.to_string(index=False))

    group_cols = ["attack_type", "byzantine_fraction"]
    summaries = []
    for value_col in [
        "gradf_accuracy", "best_fixed_accuracy", "accuracy_retention",
        "accuracy_retention_vs_best_fixed", "detection_rate",
        "pretrain_seconds", "baseline_wall_clock_seconds", "gradf_wall_clock_seconds",
    ]:
        summary = summarize(raw, group_cols, value_col)
        summary = summary.rename(columns={c: f"{value_col}_{c}" for c in ["mean", "std", "ci95_low", "ci95_high", "n"]})
        summaries.append(summary.set_index(group_cols))
    summary_df = pd.concat(summaries, axis=1).reset_index()
    summary_df.to_csv("results/tables/exp2_robustness_summary.csv", index=False)
    print("\n=== summary (mean/std/CI95 over seeds) ===")
    print(summary_df.to_string(index=False))

    final_per_attack = summary_df.groupby("attack_type")["detection_rate_mean"].mean().to_dict()
    plot_bar_comparison(
        final_per_attack, title=f"Detection rate by attack type ({args.dataset})",
        ylabel="Detection rate", save_path="results/figures/exp2_detection_rate.png",
    )
