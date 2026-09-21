"""Detection modality 4: temporal consistency (see FORMALISMO_MATEMATICO_E_INEDITISMO.md, Definition 4).

τ_4^{t,i} = ||g_i^t - g_i^{t-1}||_2 / (||g_i^{t-1}||_2 + ε)
Z_4^{t,i} = |τ_4^{t,i} - E[τ_4^{t,i}]| / σ(τ_4^{t,i})
"""

from collections import deque

import numpy as np


class TemporalDetector:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.previous_update = {}
        self.consistency_history = {}

    def analyze(self, hospital_id, update):
        """Analyze a hospital's update consistency relative to the previous round.

        Args:
            hospital_id: participant identifier.
            update: the current round's parameter (delta) vector.

        Returns:
            dict with `consistency`, `z_score`, `anomaly_score` (0-1).
        """
        prev = self.previous_update.get(hospital_id)

        if prev is None:
            consistency = 0.0
        else:
            consistency = float(np.linalg.norm(update - prev) / (np.linalg.norm(prev) + 1e-10))

        self.previous_update[hospital_id] = np.array(update, copy=True)

        history = self.consistency_history.setdefault(
            hospital_id, deque(maxlen=self.window_size)
        )

        if len(history) >= 2:
            mean = np.mean(history)
            std = np.std(history)
            z_score = abs(consistency - mean) / std if std > 0 else 0.0
        else:
            z_score = 0.0

        history.append(consistency)

        anomaly_score = min(1.0, abs(z_score) / 10.0)

        return {
            "consistency": consistency,
            "z_score": z_score,
            "anomaly_score": anomaly_score,
        }


if __name__ == "__main__":
    detector = TemporalDetector()
    rng = np.random.default_rng(0)
    base = rng.standard_normal(50)
    for round_num in range(5):
        update = base + rng.standard_normal(50) * 0.05
        result = detector.analyze("hospital_a", update)
        print(f"round {round_num}: {result}")

    print("--- attack: completely different update ---")
    result = detector.analyze("hospital_a", rng.standard_normal(50) * 20)
    print(result)
