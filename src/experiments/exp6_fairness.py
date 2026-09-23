"""
Experiment 6: fairness — accuracy variance across non-IID clients, with and
without GRADF, under attack.
"""

import argparse
import os

import pandas as pd

from src.experiments.exp1_baseline import train_gradf_models
from src.fl.attacked_learner import AttackedFederatedLearner
from src.fl.gradf_learner import GRADFFederatedLearner
from src.utils.data_loader import load_dataset_participants
from src.utils.logger import get_logger
from src.utils.metrics import fairness_std
from src.utils.visualization import plot_bar_comparison

logger = get_logger(__name__)


def run_fairness_experiment(
    dataset: str = "mnist",
    split: str = "non_iid",
    n_clients: int = 10,
    n_rounds: int = 15,
    attack_type: str = "sign_flipping",
    byzantine_fraction: float = 0.2,
    n_classes: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    participants, root_data = load_dataset_participants(dataset, split, n_clients)
    byzantine_ids = list(range(max(1, int(n_clients * byzantine_fraction))))

    classifier, selector, attack_labels, _oracle_map = train_gradf_models(
        participants, attack_type, byzantine_fraction, n_classes, seed=seed
    )

    baseline = AttackedFederatedLearner(
        n_rounds=n_rounds, aggregation="fedavg", n_classes=n_classes,
        attack_type=attack_type, byzantine_ids=byzantine_ids, seed=seed,
    )
    baseline_result = baseline.train(participants, root_data=root_data, verbose=False)[-1]

    gradf = GRADFFederatedLearner(
        n_rounds=n_rounds, aggregation="fedavg", n_classes=n_classes,
        attack_type=attack_type, byzantine_ids=byzantine_ids, seed=seed,
        classifier=classifier, selector=selector, attack_labels=attack_labels,
    )
    gradf_result = gradf.train(participants, root_data=root_data, verbose=False)[-1]

    rows = [
        {
            "system": "FedAvg",
            "mean_accuracy": baseline_result.global_accuracy,
            "fairness_std": fairness_std(baseline_result.per_participant_accuracy),
        },
        {
            "system": "GRADF",
            "mean_accuracy": gradf_result.global_accuracy,
            "fairness_std": fairness_std(gradf_result.per_participant_accuracy),
        },
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--split", default="non_iid", choices=["iid", "non_iid"])
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--n_rounds", type=int, default=15)
    parser.add_argument("--attack_type", default="sign_flipping")
    parser.add_argument("--byzantine_fraction", type=float, default=0.2)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = run_fairness_experiment(
        dataset=args.dataset, split=args.split, n_clients=args.n_clients,
        n_rounds=args.n_rounds, attack_type=args.attack_type,
        byzantine_fraction=args.byzantine_fraction, n_classes=args.n_classes, seed=args.seed,
    )

    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/exp6_fairness.csv", index=False)
    print(df.to_string(index=False))

    plot_bar_comparison(
        dict(zip(df["system"], df["fairness_std"])),
        title=f"Fairness (accuracy std across clients) — {args.dataset}",
        ylabel="std(per-client accuracy)", save_path="results/figures/exp6_fairness.png",
    )
