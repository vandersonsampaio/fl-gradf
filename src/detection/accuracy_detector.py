"""Detection modality 2: accuracy degradation.

Δ_2^{t,i} = |Acc(θ^{t-1}; V) - Acc(θ_updated; V)| / Acc(θ^{t-1}; V)
Z_2^{t,i} = (Δ_2^{t,i} - E[Δ_2^{t,i}]) / σ(Δ_2^{t,i})

This detector is deliberately decoupled from `src.fl` — whoever calls `.analyze()` has
already evaluated `old_accuracy`/`new_accuracy` on the validation set (e.g. by applying
`_LogisticModel.set_params()` + `.accuracy()` before and after a candidate update).
This avoids circular coupling between `src.detection` and `src.fl`.
"""

from collections import deque

import numpy as np


class AccuracyDetector:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.degradation_history = {}

    def analyze(self, hospital_id, old_accuracy, new_accuracy):
        """Analyze the impact of a candidate update on validation accuracy.

        Args:
            hospital_id: participant identifier.
            old_accuracy: current global model's accuracy on the validation set.
            new_accuracy: global model accuracy after (hypothetically) applying
                the participant's update.

        Returns:
            dict with `accuracy_drop`, `degradation`, `z_score`, `anomaly_score` (0-1).
        """
        accuracy_drop = old_accuracy - new_accuracy
        degradation = abs(accuracy_drop) / max(old_accuracy, 1e-10)

        history = self.degradation_history.setdefault(
            hospital_id, deque(maxlen=self.window_size)
        )

        if len(history) >= 2:
            mean = np.mean(history)
            std = np.std(history)
            z_score = (degradation - mean) / std if std > 0 else 0.0
        else:
            z_score = 0.0

        history.append(degradation)

        anomaly_score = min(1.0, abs(z_score) / 10.0 + max(0.0, degradation))

        return {
            "accuracy_drop": accuracy_drop,
            "degradation": degradation,
            "z_score": z_score,
            "anomaly_score": anomaly_score,
        }


if __name__ == "__main__":
    detector = AccuracyDetector()
    for old_acc, new_acc in [(0.90, 0.89), (0.90, 0.88), (0.90, 0.55)]:
        result = detector.analyze("hospital_a", old_acc, new_acc)
        print(f"old={old_acc} new={new_acc} -> {result}")
