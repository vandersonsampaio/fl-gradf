"""
Experiment 7: quantifies multi-modal detection's contribution — trains
`RLAttackClassifier` with all 4 modalities vs. with a single modality at a
time (the others masked/zeroed out), and compares classification accuracy,
detection rate and false-positive rate on a held-out test set.

Numerically justifies the claim "multi-modal detection beats single
modality" without needing any new component: `RLAttackClassifier` has a
fixed 4-feature input (see `_build_network`), so "single modality" is
simulated by zeroing the unused columns — equivalent to a classifier that
only sees that one signal, keeping the same architecture for a fair
comparison.

Column order of `generate_detection_dataset` (see
`src.detection.modality_recorder.ModalityRecorder.score`): 0=gradient,
1=accuracy, 2=similarity/clustering, 3=temporal.
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.classification.rl_classifier import RLAttackClassifier
from src.utils.data_loader import generate_detection_dataset, load_dataset_participants
from src.utils.logger import get_logger
from src.utils.metrics import detection_rate, false_positive_rate
from src.utils.stats import run_over_seeds, summarize
from src.utils.visualization import plot_bar_comparison

logger = get_logger(__name__)

MODALITY_SUBSETS = {
    "all_4_modalities": [0, 1, 2, 3],
    "gradient_only": [0],
    "accuracy_only": [1],
}


def run_modality_ablation(
    seed: int,
    dataset: str = "mnist",
    split: str = "non_iid",
    n_clients: int = 10,
    n_rounds: int = 10,
    attack_types: Optional[List[str]] = None,
    byzantine_fraction: float = 0.2,
    n_classes: int = 10,
    test_size: float = 0.3,
) -> pd.DataFrame:
    attack_types = attack_types or ["sign_flipping", "gaussian_noise", "label_flipping"]
    attack_labels = ["none"] + attack_types
    label_to_idx = {label: i for i, label in enumerate(attack_labels)}
    none_idx = label_to_idx["none"]

    participants, _root_data = load_dataset_participants(dataset, split, n_clients)

    logger.info("Generating detection dataset (seed=%d)...", seed)
    X, y_binary, y_attack = generate_detection_dataset(
        participants, n_rounds=n_rounds, attack_types=attack_types,
        byzantine_fraction=byzantine_fraction, aggregation="fedavg",
        n_classes=n_classes, seed=seed,
    )
    y_idx = np.array([label_to_idx[t] for t in y_attack])

    X_train, X_test, y_idx_train, y_idx_test, y_bin_train, y_bin_test = train_test_split(
        X, y_idx, y_binary, test_size=test_size, random_state=seed, stratify=y_idx,
    )

    rows = []
    for name, cols in MODALITY_SUBSETS.items():
        mask = np.zeros(4, dtype=np.float32)
        mask[cols] = 1.0
        X_train_masked = (X_train * mask).astype(np.float32)
        X_test_masked = (X_test * mask).astype(np.float32)

        classifier = RLAttackClassifier(n_attacks=len(attack_labels))
        classifier.train(X_train_masked, y_idx_train, epochs=30, batch_size=32, verbose=False)

        probs = classifier.model(X_test_masked).numpy()
        y_pred_idx = probs.argmax(axis=1)

        accuracy = float(np.mean(y_pred_idx == y_idx_test))
        y_pred_bin = (y_pred_idx != none_idx).tolist()
        y_true_bin = y_bin_test.astype(bool).tolist()

        rows.append({
            "modality_subset": name,
            "accuracy": accuracy,
            "detection_rate": detection_rate(y_true_bin, y_pred_bin),
            "false_positive_rate": false_positive_rate(y_true_bin, y_pred_bin),
        })
        logger.info("%s: accuracy=%.3f det_rate=%.2f", name, accuracy, rows[-1]["detection_rate"])

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--split", default="non_iid", choices=["iid", "non_iid"])
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--n_rounds", type=int, default=10)
    parser.add_argument("--attack_types", nargs="+", default=None)
    parser.add_argument("--byzantine_fraction", type=float, default=0.2)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    raw = run_over_seeds(
        run_modality_ablation,
        seeds=args.seeds,
        dataset=args.dataset, split=args.split, n_clients=args.n_clients,
        n_rounds=args.n_rounds, attack_types=args.attack_types,
        byzantine_fraction=args.byzantine_fraction, n_classes=args.n_classes,
    )

    os.makedirs("results/tables", exist_ok=True)
    raw.to_csv("results/tables/exp7_modality_ablation_raw.csv", index=False)
    print("=== raw (per seed) ===")
    print(raw.to_string(index=False))

    summaries = []
    for value_col in ["accuracy", "detection_rate", "false_positive_rate"]:
        s = summarize(raw, ["modality_subset"], value_col)
        s = s.rename(columns={c: f"{value_col}_{c}" for c in ["mean", "std", "ci95_low", "ci95_high", "n"]})
        summaries.append(s.set_index("modality_subset"))
    summary_df = pd.concat(summaries, axis=1).reset_index()
    summary_df.to_csv("results/tables/exp7_modality_ablation_summary.csv", index=False)
    print("\n=== summary (mean/std/CI95 over seeds) ===")
    print(summary_df.to_string(index=False))

    mean_accuracy = raw.groupby("modality_subset")["accuracy"].mean().to_dict()
    plot_bar_comparison(
        mean_accuracy,
        title=f"Classification accuracy by modality subset ({args.dataset})",
        ylabel="Accuracy",
        save_path="results/figures/exp7_modality_ablation.png",
    )
