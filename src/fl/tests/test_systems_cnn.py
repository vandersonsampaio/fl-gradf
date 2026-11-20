"""
The exp10 adaptive systems run on the CNN (P1 CIFAR-10 campaign): GRADF's
pretraining, GRADF itself, the Random-selector and FedStrategist, with the
legacy Layer-2 epsilon rescaled to the CNN's size. Infrastructure only, on
small synthetic 4x4x1 images, as in test_federated_learner_cnn.py.
"""

import numpy as np
import pytest

from src.defense.hardening import legacy_dp_epsilon_for
from tests.fixtures.mock_data import make_synthetic_participants

pytest.importorskip("tensorflow")

MODEL = {"model_type": "cnn", "input_shape": (4, 4, 1), "local_epochs": 1}


@pytest.fixture
def participants():
    return make_synthetic_participants(n_clients=3, n_features=16, n_train=32, n_test=16, seed=0)


def test_legacy_dp_epsilon_keeps_mnist_calibration():
    assert legacy_dp_epsilon_for(7850) == pytest.approx(50.0)
    assert legacy_dp_epsilon_for(34522) == pytest.approx(50.0 * np.sqrt(34522 / 7850))


def test_gradf_pretraining_and_run_on_cnn(participants):
    from src.experiments.exp1_baseline import train_gradf_models
    from src.fl.gradf_learner import GRADFFederatedLearner

    classifier, selector, labels, _ = train_gradf_models(
        participants, "sign_flipping", 0.34, 2, n_rounds=2, seed=0, model_kwargs=MODEL)
    learner = GRADFFederatedLearner(
        aggregation="fedavg", classifier=classifier, selector=selector, attack_labels=labels,
        n_rounds=2, n_classes=2, attack_type="sign_flipping", byzantine_ids=[0], seed=0,
        dp_epsilon=legacy_dp_epsilon_for(1858), **MODEL)
    history = learner.train(participants, verbose=False)
    assert len(history) == 2 and all(0.0 <= r.global_accuracy <= 1.0 for r in history)


def test_random_selector_and_fedstrategist_on_cnn(participants):
    from src.defense.rl_selector import RLDefenseSelector
    from src.experiments.exp1_baseline import GroundTruthClassifierStub, RandomSelector
    from src.experiments.exp10_selector_comparison import FedStrategistGridLearner
    from src.fl.gradf_learner import GRADFFederatedLearner

    rnd = GRADFFederatedLearner(
        aggregation="fedavg", classifier=GroundTruthClassifierStub(),
        selector=RandomSelector(RLDefenseSelector().actions, seed=0),
        attack_labels=["none", "sign_flipping"], n_rounds=2, n_classes=2,
        attack_type="sign_flipping", byzantine_ids=[0], seed=0, **MODEL)
    fs = FedStrategistGridLearner(
        n_rounds=2, n_classes=2, attack_type="sign_flipping", byzantine_ids=[0], seed=0,
        variant="a", **MODEL)
    for learner in (rnd, fs):
        history = learner.train(participants, verbose=False)
        assert len(history) == 2
