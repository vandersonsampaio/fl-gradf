"""Detection modality 3: clustering distance.

S_3^{t,i,j} = cos(g_i^t, g_j^t)  for all j != i
Z_3^{t,i} = (1/N) * sum_{j!=i} [1 - S_3^{t,i,j}]
"""

import numpy as np


def _cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


class ClusteringDetector:
    def analyze(self, hospital_id, update, peer_updates):
        """Analyze the distance (1 - mean cosine similarity) of an update to the
        other participants' updates in the same round.

        Args:
            hospital_id: participant identifier (used only for the return value).
            update: the participant's parameter (delta) vector.
            peer_updates: list of parameter vectors from the other participants in
                the same round (not including `update`).

        Returns:
            dict with `mean_cosine_similarity`, `distance`, `anomaly_score` (0-1).
        """
        if not peer_updates:
            return {
                "hospital_id": hospital_id,
                "mean_cosine_similarity": 1.0,
                "distance": 0.0,
                "anomaly_score": 0.0,
            }

        similarities = [_cosine_similarity(update, peer) for peer in peer_updates]
        mean_similarity = float(np.mean(similarities))
        distance = 1.0 - mean_similarity

        anomaly_score = min(1.0, max(0.0, distance) / 2.0)

        return {
            "hospital_id": hospital_id,
            "mean_cosine_similarity": mean_similarity,
            "distance": distance,
            "anomaly_score": anomaly_score,
        }


if __name__ == "__main__":
    detector = ClusteringDetector()
    rng = np.random.default_rng(0)
    base = rng.standard_normal(50)
    honest_updates = [base + rng.standard_normal(50) * 0.05 for _ in range(4)]

    print("Honest update:", detector.analyze("hospital_a", honest_updates[0], honest_updates[1:]))
    print("Attacked update:", detector.analyze("hospital_x", -base * 10, honest_updates))
