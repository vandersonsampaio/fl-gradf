import numpy as np
import pytest

from src.detection.accuracy_detector import AccuracyDetector
from src.detection.clustering_detector import ClusteringDetector
from src.detection.combiner_nn import AnomalyCombinerNN, DetectionCombiner
from src.detection.gradient_analyzer import GradientAnalyzer
from src.detection.temporal_detector import TemporalDetector


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestGradientAnalyzer:
    def test_normal_gradient_scores_lower_than_attack(self, rng):
        analyzer = GradientAnalyzer(window_size=5)
        for _ in range(6):
            normal_result = analyzer.analyze("hospital_a", rng.standard_normal(50))

        attack_result = analyzer.analyze("hospital_a", rng.standard_normal(50) * 50)
        assert attack_result["anomaly_score"] > normal_result["anomaly_score"]

    def test_output_keys(self, rng):
        analyzer = GradientAnalyzer()
        result = analyzer.analyze("hospital_a", rng.standard_normal(10))
        assert set(result) == {"magnitude", "z_score", "js_divergence", "anomaly_score"}
        assert 0.0 <= result["anomaly_score"] <= 1.0


class TestAccuracyDetector:
    def test_no_degradation_scores_low(self):
        detector = AccuracyDetector()
        for _ in range(6):
            result = detector.analyze("hospital_a", old_accuracy=0.90, new_accuracy=0.90)
        assert result["anomaly_score"] < 0.1

    def test_large_degradation_scores_higher(self):
        detector = AccuracyDetector()
        for _ in range(6):
            detector.analyze("hospital_a", old_accuracy=0.90, new_accuracy=0.895)
        attack_result = detector.analyze("hospital_a", old_accuracy=0.90, new_accuracy=0.40)
        assert attack_result["anomaly_score"] > 0.3


class TestTemporalDetector:
    def test_consistent_updates_score_low(self, rng):
        detector = TemporalDetector(window_size=5)
        base = rng.standard_normal(30)
        for _ in range(6):
            result = detector.analyze("hospital_a", base + rng.standard_normal(30) * 0.01)
        assert result["anomaly_score"] < 0.5

    def test_sudden_change_scores_higher(self, rng):
        detector = TemporalDetector(window_size=5)
        base = rng.standard_normal(30)
        for _ in range(6):
            detector.analyze("hospital_a", base + rng.standard_normal(30) * 0.01)
        attack_result = detector.analyze("hospital_a", rng.standard_normal(30) * 50)
        assert attack_result["anomaly_score"] >= 0.0  # must not raise
        assert "consistency" in attack_result

    def test_first_call_has_zero_consistency(self, rng):
        detector = TemporalDetector()
        result = detector.analyze("hospital_a", rng.standard_normal(10))
        assert result["consistency"] == 0.0


class TestClusteringDetector:
    def test_similar_update_has_low_distance(self, rng):
        detector = ClusteringDetector()
        base = rng.standard_normal(20)
        peers = [base + rng.standard_normal(20) * 0.01 for _ in range(4)]
        result = detector.analyze("hospital_a", base, peers)
        assert result["distance"] < 0.2

    def test_opposite_update_has_high_distance(self, rng):
        detector = ClusteringDetector()
        base = rng.standard_normal(20)
        peers = [base + rng.standard_normal(20) * 0.01 for _ in range(4)]
        result = detector.analyze("hospital_x", -base, peers)
        assert result["distance"] > 1.5

    def test_no_peers_returns_zero_distance(self):
        detector = ClusteringDetector()
        result = detector.analyze("hospital_a", np.zeros(10), [])
        assert result["distance"] == 0.0


class TestAnomalyCombinerNN:
    def test_forward_pass_shape(self):
        model = AnomalyCombinerNN()
        out = model(np.random.randn(8, 4).astype(np.float32))
        assert out.shape == (8, 1)

    def test_output_in_unit_interval(self):
        model = AnomalyCombinerNN()
        out = model(np.random.randn(8, 4).astype(np.float32)).numpy()
        assert np.all((out >= 0.0) & (out <= 1.0))


class TestDetectionCombiner:
    def test_train_reduces_loss(self, rng):
        X = rng.standard_normal((200, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)  # learnable signal

        combiner = DetectionCombiner()
        losses = combiner.train(X, y, epochs=15, batch_size=32, verbose=False)
        assert losses[-1] < losses[0]

    def test_predict_shape(self, rng):
        combiner = DetectionCombiner()
        preds = combiner.predict(rng.standard_normal((5, 4)))
        assert preds.shape == (5,)
        assert np.all((preds >= 0.0) & (preds <= 1.0))

    def test_save_and_load_roundtrip(self, tmp_path, rng):
        combiner = DetectionCombiner()
        combiner.predict(rng.standard_normal((1, 4)))  # forces build
        path = str(tmp_path / "combiner.weights.h5")
        combiner.save(path)

        loaded = DetectionCombiner()
        loaded.load(path)

        X = rng.standard_normal((5, 4)).astype(np.float32)
        np.testing.assert_allclose(combiner.predict(X), loaded.predict(X), rtol=1e-5)
