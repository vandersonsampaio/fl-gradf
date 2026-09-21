import numpy as np
from scipy.spatial.distance import jensenshannon


class GradientAnalyzer:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.gradient_history = {}
        # Holds the last raw gradient vector per hospital, separate from
        # `gradient_history` (which only holds scalar magnitudes, used by the
        # z-score). The JS-divergence term below must diff against this raw
        # vector, not a magnitude scalar — `np.histogram()` of a scalar is a
        # degenerate distribution (all mass in one bin) and does not represent
        # the actual previous gradient.
        self.last_gradient = {}

    def analyze(self, hospital_id, gradient):
        """Analyze gradient magnitude and distribution."""

        # 1. Magnitude (Euclidean norm)
        magnitude = np.linalg.norm(gradient)

        # 2. Z-score against history
        if hospital_id not in self.gradient_history:
            self.gradient_history[hospital_id] = []

        self.gradient_history[hospital_id].append(magnitude)

        if len(self.gradient_history[hospital_id]) >= self.window_size:
            recent = self.gradient_history[hospital_id][-self.window_size:]
            mean = np.mean(recent)
            std = np.std(recent)

            if std > 0:
                z_score = (magnitude - mean) / std
            else:
                z_score = 0
        else:
            z_score = 0

        # 3. Jensen-Shannon divergence vs. the previous round's raw gradient
        # (not vs. the magnitude scalar — see note in __init__).
        previous_gradient = self.last_gradient.get(hospital_id)
        self.last_gradient[hospital_id] = np.asarray(gradient, dtype=float).copy()

        if previous_gradient is not None:
            current_hist = np.histogram(gradient, bins=10)[0].astype(float)
            previous_hist = np.histogram(previous_gradient, bins=10)[0].astype(float)

            if current_hist.sum() > 0 and previous_hist.sum() > 0:
                current_hist /= current_hist.sum()
                previous_hist /= previous_hist.sum()
                js_div = float(jensenshannon(current_hist, previous_hist))
                if np.isnan(js_div):
                    js_div = 0.0
            else:
                js_div = 0.0
        else:
            js_div = 0

        # Score: 0-1 (1 = anomalous)
        anomaly_score = min(1.0, abs(z_score) / 10.0 + js_div)

        return {
            'magnitude': magnitude,
            'z_score': z_score,
            'js_divergence': js_div,
            'anomaly_score': anomaly_score
        }


if __name__ == '__main__':
    analyzer = GradientAnalyzer()
    test_gradient = np.random.randn(100)
    result = analyzer.analyze('hospital_a', test_gradient)
    print(f"Anomaly Score: {result['anomaly_score']:.4f}")
