"""
Infrastructure tests for the 4 "informed" attacks and for
`InformedAttackedFederatedLearner`. Small synthetic data, only covering that
the code runs and produces outputs with valid shape/values — not measuring
attack effectiveness (that's the actual purpose of `exp9_dominance_grid.py`,
which runs on real data).
"""

import numpy as np
import pytest

import src.defense.aggregation_methods
from src.classification.attack_simulator import AttackSimulator
from src.fl.attacked_learner import AttackedFederatedLearner, InformedAttackedFederatedLearner
from tests.fixtures.mock_data import make_synthetic_participants


@pytest.fixture
def participants():
    return make_synthetic_participants(n_clients=6, n_features=12, n_train=40, n_test=16, seed=1)


class TestInformedAttackImplementations:
    """Direct unit tests on AttackSimulator.poison_update for the 4 new types."""

    def setup_method(self):
        self.sim = AttackSimulator(seed=0)
        self.update = np.random.default_rng(0).standard_normal(12)
        self.reference = np.random.default_rng(1).standard_normal(12)
        self.peers = [np.random.default_rng(i).standard_normal(12) for i in range(2, 6)]

    @pytest.mark.parametrize("attack_type", AttackSimulator.INFORMED_ATTACK_TYPES)
    def test_shape_and_finiteness_with_context(self, attack_type):
        out = self.sim.poison_update(
            self.update, attack_type, reference=self.reference, peer_updates=self.peers,
        )
        assert out.shape == self.update.shape
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("attack_type", AttackSimulator.INFORMED_ATTACK_TYPES)
    def test_falls_back_gracefully_without_context(self, attack_type):
        """Without reference/peer_updates (blind attacker), each type falls
        back to its documented behavior instead of breaking."""
        out = self.sim.poison_update(self.update, attack_type)
        assert out.shape == self.update.shape
        assert np.all(np.isfinite(out))

    def test_fltrust_aligned_clones_reference(self):
        out = self.sim.poison_update(self.update, "fltrust_aligned", reference=self.reference)
        np.testing.assert_allclose(out, self.reference)

    def test_informed_attack_types_are_registered(self):
        assert set(AttackSimulator.INFORMED_ATTACK_TYPES) <= set(AttackSimulator.ALL_ATTACK_TYPES)


class TestInformedAttackedFederatedLearner:
    """Light integration tests: the learner trains end to end, with and
    without FLTrust as the top-level aggregation, for each informed attack."""

    @pytest.mark.parametrize("attack_type", AttackSimulator.INFORMED_ATTACK_TYPES)
    @pytest.mark.parametrize("aggregation", ["fltrust", "median", "trimmed_mean", "krum"])
    def test_trains_without_error(self, participants, attack_type, aggregation):
        learner = InformedAttackedFederatedLearner(
            n_rounds=2, aggregation=aggregation, n_classes=2,
            attack_type=attack_type, byzantine_ids=[0, 1], seed=0,
        )
        history = learner.train(participants, verbose=False)
        assert len(history) == 2
        assert 0.0 <= history[-1].global_accuracy <= 1.0

    def test_label_flipping_still_handled_at_data_level(self, participants):
        """label_flipping is not an 'informed' attack but should keep
        working through InformedAttackedFederatedLearner (same data path as
        AttackedFederatedLearner)."""
        learner = InformedAttackedFederatedLearner(
            n_rounds=2, aggregation="fedavg", n_classes=2,
            attack_type="label_flipping", byzantine_ids=[0], seed=0,
        )
        history = learner.train(participants, verbose=False)
        assert len(history) == 2

    def test_matches_blind_learner_interface(self, participants):
        """Both return the same kind of history for the same honest scenario
        (attack_type='none') — equivalent infrastructure when no attack is
        active."""
        blind = AttackedFederatedLearner(
            n_rounds=2, aggregation="fedavg", n_classes=2,
            attack_type="none", byzantine_ids=[], seed=0,
        )
        informed = InformedAttackedFederatedLearner(
            n_rounds=2, aggregation="fedavg", n_classes=2,
            attack_type="none", byzantine_ids=[], seed=0,
        )
        h_blind = blind.train(participants, verbose=False)
        h_informed = informed.train(participants, verbose=False)
        assert len(h_blind) == len(h_informed) == 2
