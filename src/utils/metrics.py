"""Metrics shared by the experiments (src/experiments/) and notebooks."""

from typing import Dict, List

import numpy as np


def detection_rate(y_true_attacked: List[bool], y_pred_attacked: List[bool]) -> float:
    """Fraction of truly attacked updates that were correctly flagged
    (recall of the "attacked" class). NaN if there are no attacked updates."""
    y_true = np.asarray(y_true_attacked, dtype=bool)
    y_pred = np.asarray(y_pred_attacked, dtype=bool)
    if y_true.sum() == 0:
        return float("nan")
    return float(np.mean(y_pred[y_true]))


def false_positive_rate(y_true_attacked: List[bool], y_pred_attacked: List[bool]) -> float:
    """Fraction of honest updates that were incorrectly flagged/rejected."""
    y_true = np.asarray(y_true_attacked, dtype=bool)
    y_pred = np.asarray(y_pred_attacked, dtype=bool)
    honest = ~y_true
    if honest.sum() == 0:
        return float("nan")
    return float(np.mean(y_pred[honest]))


def fairness_std(per_client_accuracy: Dict[str, float]) -> float:
    """Standard deviation of accuracy across clients — lower is more equitable."""
    return float(np.std(list(per_client_accuracy.values())))


def accuracy_retention(baseline_accuracy: float, current_accuracy: float) -> float:
    """Fraction of baseline accuracy retained (1.0 = no degradation, <1.0 = degraded)."""
    if baseline_accuracy == 0:
        return float("nan")
    return current_accuracy / baseline_accuracy


def online_selector_reward(
    attack_detected: bool, old_accuracy: float, new_accuracy: float, tol: float = 0.02,
) -> float:
    """Reward for the `RLDefenseSelector`'s ONLINE learning — same shape as
    `src.utils.data_loader._selector_reward` (accuracy retention as the
    dominant term, bonus for neutralizing a real attack, penalty for a false
    positive), but computed from the before/after of the ACTUAL real FL round
    (`old_accuracy`/`new_accuracy` for this round), not from an offline
    pretraining episode against a separate baseline. Used by
    `GRADFFederatedLearner._run_round` to train the selector on every real
    round (see `src/defense/rl_selector.py`, "real online learning")."""
    retained = new_accuracy >= old_accuracy - tol
    reward = 1.0 if retained else -0.5
    if attack_detected and retained:
        reward += 0.5
    if not attack_detected and not retained:
        reward -= 0.1
    return reward


if __name__ == "__main__":
    y_true = [True, True, False, False, True]
    y_pred = [True, False, False, True, True]
    print("detection_rate:", detection_rate(y_true, y_pred))
    print("false_positive_rate:", false_positive_rate(y_true, y_pred))
    print("fairness_std:", fairness_std({"a": 0.9, "b": 0.85, "c": 0.5}))
    print("accuracy_retention:", accuracy_retention(0.9, 0.88))
