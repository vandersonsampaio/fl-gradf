"""
Infrastructure tests for `model_type='cnn'` (Decision A4,
`references/plano_gradf_iclr2027.md`) — cover `_CNNModel` in isolation and
`FederatedLearner` end to end. Do not test real learning quality (that is
validated separately on real MNIST/CIFAR-10, see
`src/experiments/exp1_cnn_scale.py`) — only that the flat-vector
infrastructure (fit/set_params/params/accuracy) and the `_make_model`
dispatch work. Small synthetic data (`input_shape=(4,4,1)`, 16 features) so
the suite runs in milliseconds.
"""

import numpy as np
import pytest

from src.fl.federated_learner import FederatedLearner, _CNNModel
from tests.fixtures.mock_data import make_synthetic_participants


@pytest.fixture
def synthetic_image_participants():
    # n_features=16 == prod((4,4,1)) — synthetic 4x4x1 "image".
    return make_synthetic_participants(n_clients=3, n_features=16, n_train=32, n_test=16, seed=0)


class TestCNNModelInterface:
    def test_fit_returns_delta_with_correct_shape(self, synthetic_image_participants):
        p = synthetic_image_participants[0]
        model = _CNNModel(n_features=16, input_shape=(4, 4, 1), n_classes=2)
        n_params = len(model.params)
        delta = model.fit(p.X_train, p.y_train)
        assert delta.shape == (n_params,)
        assert np.all(np.isfinite(delta))

    def test_set_params_round_trip(self):
        model = _CNNModel(n_features=16, input_shape=(4, 4, 1), n_classes=2)
        original = model.params
        tweaked = original + 1.0
        model.set_params(tweaked)
        np.testing.assert_allclose(model.params, tweaked, rtol=1e-5)

    def test_accuracy_in_valid_range(self, synthetic_image_participants):
        p = synthetic_image_participants[0]
        model = _CNNModel(n_features=16, input_shape=(4, 4, 1), n_classes=2)
        acc = model.accuracy(p.X_test, p.y_test)
        assert 0.0 <= acc <= 1.0

    def test_mismatched_input_shape_raises(self):
        with pytest.raises(ValueError):
            _CNNModel(n_features=16, input_shape=(5, 5, 1), n_classes=2)  # 25 != 16


class TestFederatedLearnerModelDispatch:
    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError):
            FederatedLearner(model_type="not_a_real_model")

    def test_cnn_without_input_shape_raises(self):
        with pytest.raises(ValueError):
            FederatedLearner(model_type="cnn")

    def test_logistic_default_unaffected(self, synthetic_image_participants):
        """model_type='logistic' (the default) keeps working exactly as
        before — should not import `_CNNModel`/TensorFlow beyond what the
        suite already imports."""
        learner = FederatedLearner(n_rounds=2, aggregation="fedavg", n_classes=2)
        history = learner.train(synthetic_image_participants, verbose=False)
        assert len(history) == 2
        assert 0.0 <= history[-1].global_accuracy <= 1.0

    def test_cnn_end_to_end_training_runs(self, synthetic_image_participants):
        learner = FederatedLearner(
            n_rounds=2, aggregation="fedavg", n_classes=2,
            model_type="cnn", input_shape=(4, 4, 1), local_epochs=1,
        )
        history = learner.train(synthetic_image_participants, verbose=False)
        assert len(history) == 2
        for r in history:
            assert 0.0 <= r.global_accuracy <= 1.0
        # Glorot initialization (non-zero) — global_params should not be all-zero.
        assert not np.allclose(learner.global_params, 0.0)

    def test_cnn_works_with_robust_aggregation(self, synthetic_image_participants):
        """Ensures `_CNNModel`'s flat vector is agnostic to the aggregation
        strategy — median/trimmed_mean operate coordinate-wise and shouldn't
        care about the vector's provenance (CNN vs. logistic)."""
        learner = FederatedLearner(
            n_rounds=2, aggregation="median", n_classes=2,
            model_type="cnn", input_shape=(4, 4, 1), local_epochs=1,
        )
        history = learner.train(synthetic_image_participants, verbose=False)
        assert len(history) == 2
