"""
AdaAggRL on the CNN (P1 CIFAR-10 campaign): the gradient inversion runs on
the CNN's own parameters, and the grid learner trains end to end with
model_type='cnn'. Small synthetic 4x4x1 images, as in
src/fl/tests/test_federated_learner_cnn.py.
"""

import numpy as np
import pytest

from src.defense.adaaggrl_agent import reconstruct_client_distribution_cnn
from src.fl.federated_learner import _CNNModel
from tests.fixtures.mock_data import make_synthetic_participants

pytest.importorskip("tensorflow")

SHAPE = (4, 4, 1)


@pytest.fixture
def participants():
    return make_synthetic_participants(n_clients=3, n_features=16, n_train=32, n_test=16, seed=0)


def test_cnn_inversion_aligns_dummy_gradient(participants):
    model = _CNNModel(16, input_shape=SHAPE, n_classes=2, lr=0.05, n_epochs=1, batch_size=32)
    theta = model.params
    p = participants[0]
    update = model.fit(p.X_train, p.y_train)  # one full-batch step: update = -lr * grad
    model.set_params(theta)
    _, s_short = reconstruct_client_distribution_cnn(
        model._model, -update, 0.05, SHAPE, 2, num_images=8, max_iters=1, seed=0)
    D_rec, s_long = reconstruct_client_distribution_cnn(
        model._model, -update, 0.05, SHAPE, 2, num_images=8, max_iters=60, seed=0)
    assert D_rec.shape == (8, 16)
    assert np.isfinite(s_long) and s_long > s_short


def test_adaaggrl_grid_learner_runs_on_cnn(participants):
    from src.experiments.exp10_selector_comparison import AdaAggRLGridLearner

    learner = AdaAggRLGridLearner(
        n_rounds=2, n_classes=2, attack_type="sign_flipping", byzantine_ids=[0], seed=0,
        input_shape=SHAPE, model_type="cnn", local_epochs=1, num_images=4, max_iters=3,
    )
    history = learner.train(participants, verbose=False)
    assert len(history) == 2
    assert all(0.0 <= r.global_accuracy <= 1.0 for r in history)
