"""
Seeded CNN runs (P1 CIFAR-10 campaign): with the global NumPy/TensorFlow
seeds set and a separate server RNG (`server_rng_seed`), a run is
reproducible bit for bit, and a rule that does not use the root (trimmed
mean) gives the same result whatever the root size, also under an informed
attack whose server fit consumes random draws. Small synthetic data, as in
test_federated_learner_cnn.py.
"""

import numpy as np
import pytest

import src.defense.aggregation_methods  # noqa: F401  registers trimmed_mean
from src.fl.attacked_learner import InformedAttackedFederatedLearner
from tests.fixtures.mock_data import make_synthetic_participants

tf = pytest.importorskip("tensorflow")


@pytest.fixture(scope="module", autouse=True)
def _deterministic_ops():
    tf.config.experimental.enable_op_determinism()


@pytest.fixture
def participants():
    return make_synthetic_participants(n_clients=4, n_features=16, n_train=32, n_test=16, seed=0)


def _run(participants, aggregation, root_n, seed=7):
    tf.keras.utils.set_random_seed(seed)  # Python, NumPy and TF/Keras seeds
    learner = InformedAttackedFederatedLearner(
        n_rounds=2, aggregation=aggregation, n_classes=2, model_type="cnn",
        input_shape=(4, 4, 1), local_epochs=1, attack_type="trim_attack",
        byzantine_ids=[0], seed=seed, server_rng_seed=seed + 1,
    )
    src = participants[1]
    root = {"X": src.X_train[:root_n], "y": src.y_train[:root_n]}
    learner.train(participants, root_data=root, verbose=False)
    return learner.global_params


def test_seeded_cnn_run_is_reproducible(participants):
    a = _run(participants, "trimmed_mean", 20)
    b = _run(participants, "trimmed_mean", 20)
    assert np.array_equal(a, b)


def test_trimmed_mean_is_root_invariant_under_informed_attack(participants):
    a = _run(participants, "trimmed_mean", 20)
    b = _run(participants, "trimmed_mean", 8)
    assert np.array_equal(a, b)


def test_fltrust_depends_on_root(participants):
    a = _run(participants, "fltrust", 20)
    b = _run(participants, "fltrust", 8)
    assert not np.array_equal(a, b)
