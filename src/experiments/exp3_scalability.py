"""
Experiment 3: scalability — GRADF's final accuracy and per-round latency as
the number of clients grows (capped at 10 by the non-IID shards available
under data/processed/{dataset}/non_iid/).
"""

import argparse
import os
import time
from typing import List, Optional

import pandas as pd

from src.experiments.exp1_baseline import train_gradf_models
from src.fl.gradf_learner import GRADFFederatedLearner
from src.utils.data_loader import load_dataset_participants
from src.utils.logger import get_logger
from src.utils.visualization import plot_bar_comparison

logger = get_logger(__name__)


def run_scalability_experiment(
    dataset: str = "mnist",
    split: str = "non_iid",
    n_clients_list: Optional[List[int]] = None,
    n_rounds: int = 10,
    attack_type: str = "sign_flipping",
    byzantine_fraction: float = 0.2,
    n_classes: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    n_clients_list = n_clients_list or [3, 5, 10]

    rows = []
    for n_clients in n_clients_list:
        participants, root_data = load_dataset_participants(dataset, split, n_clients)
        byzantine_ids = list(range(max(1, int(n_clients * byzantine_fraction))))

        classifier, selector, attack_labels, _oracle_map = train_gradf_models(
            participants, attack_type, byzantine_fraction, n_classes, seed=seed
        )

        learner = GRADFFederatedLearner(
            n_rounds=n_rounds, aggregation="fedavg", n_classes=n_classes,
            attack_type=attack_type, byzantine_ids=byzantine_ids, seed=seed,
            classifier=classifier, selector=selector, attack_labels=attack_labels,
        )
        start = time.time()
        history = learner.train(participants, root_data=root_data, verbose=False)
        elapsed = time.time() - start

        row = {
            "n_clients": n_clients,
            "final_accuracy": history[-1].global_accuracy,
            "total_time_s": elapsed,
            "avg_time_per_round_s": elapsed / n_rounds,
        }
        rows.append(row)
        logger.info("n_clients=%d: acc=%.3f total=%.1fs (%.2fs/round)",
                    n_clients, row["final_accuracy"], elapsed, row["avg_time_per_round_s"])

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--split", default="non_iid", choices=["iid", "non_iid"])
    parser.add_argument("--n_clients_list", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--n_rounds", type=int, default=10)
    parser.add_argument("--attack_type", default="sign_flipping")
    parser.add_argument("--byzantine_fraction", type=float, default=0.2)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = run_scalability_experiment(
        dataset=args.dataset, split=args.split, n_clients_list=args.n_clients_list,
        n_rounds=args.n_rounds, attack_type=args.attack_type,
        byzantine_fraction=args.byzantine_fraction, n_classes=args.n_classes, seed=args.seed,
    )

    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/exp3_scalability.csv", index=False)
    print(df.to_string(index=False))

    plot_bar_comparison(
        dict(zip(df["n_clients"].astype(str), df["avg_time_per_round_s"])),
        title=f"Avg time per round vs n_clients ({args.dataset})",
        ylabel="seconds/round", save_path="results/figures/exp3_scalability.png",
    )
