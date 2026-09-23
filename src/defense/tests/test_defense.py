import numpy as np
import pytest

from src.defense.aggregation_methods import (
    ClusteringStrategy,
    FedProxStrategy,
    KrumStrategy,
    MedianStrategy,
    TrimmedMeanStrategy,
)
from src.defense.rl_selector import RLDefenseSelector
from src.fl.federated_learner import FederatedLearner, FedAvgStrategy, _STRATEGIES
from src.utils.data_loader import generate_selector_experiences
from tests.fixtures.mock_data import make_synthetic_participants as _make_synthetic_participants


def _clustered_updates(n_honest=7, n_outliers=2, dim=10, outlier_scale=50.0, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(dim)
    honest = [base + rng.standard_normal(dim) * 0.05 for _ in range(n_honest)]
    outliers = [-base * outlier_scale + rng.standard_normal(dim) * 0.05 for _ in range(n_outliers)]
    return honest, outliers, base


class TestMedianStrategy:
    def test_robust_to_minority_outliers(self):
        honest, outliers, base = _clustered_updates()
        updates = honest + outliers
        agg, meta = MedianStrategy().aggregate(updates)
        assert meta is None
        assert np.linalg.norm(agg - base) < np.linalg.norm(agg - outliers[0])


class TestTrimmedMeanStrategy:
    def test_trims_outliers_when_ratio_covers_them(self):
        honest, outliers, base = _clustered_updates(n_honest=7, n_outliers=2)
        updates = honest + outliers
        agg, meta = TrimmedMeanStrategy(trim_ratio=0.3).aggregate(updates)
        assert meta is None
        assert np.linalg.norm(agg - base) < 1.0

    def test_no_trim_equals_mean(self):
        honest, _outliers, _base = _clustered_updates(n_outliers=0)
        agg, _ = TrimmedMeanStrategy(trim_ratio=0.0).aggregate(honest)
        np.testing.assert_allclose(agg, np.mean(honest, axis=0), rtol=1e-6)


class TestFedProxStrategy:
    def test_mu_zero_equals_weighted_mean(self):
        honest, _outliers, _base = _clustered_updates(n_outliers=0)
        sample_sizes = [10] * len(honest)
        fedavg_agg, _ = FedAvgStrategy().aggregate(honest, sample_sizes=sample_sizes)
        fedprox_agg, _ = FedProxStrategy(mu=0.0).aggregate(honest, sample_sizes=sample_sizes)
        np.testing.assert_allclose(fedavg_agg, fedprox_agg, rtol=1e-6)

    def test_positive_mu_shrinks_magnitude(self):
        honest, _outliers, _base = _clustered_updates(n_outliers=0)
        fedavg_agg, _ = FedAvgStrategy().aggregate(honest)
        fedprox_agg, _ = FedProxStrategy(mu=1.0).aggregate(honest)
        assert np.linalg.norm(fedprox_agg) < np.linalg.norm(fedavg_agg)


class TestKrumStrategy:
    def test_does_not_select_the_outlier(self):
        honest, outliers, _base = _clustered_updates(n_honest=6, n_outliers=1)
        updates = honest + outliers
        outlier_index = len(honest)
        agg, meta = KrumStrategy(n_byzantine=1).aggregate(updates)
        assert meta["selected_index"] != outlier_index
        np.testing.assert_array_equal(agg, updates[meta["selected_index"]])


class TestClusteringStrategy:
    def test_aggregates_majority_cluster(self):
        honest, outliers, base = _clustered_updates(n_honest=7, n_outliers=2)
        updates = honest + outliers
        agg, meta = ClusteringStrategy().aggregate(updates)
        assert meta["cluster_sizes"][meta["majority_label"]] == 7
        assert np.linalg.norm(agg - base) < np.linalg.norm(agg - outliers[0])


class TestAggregationStrategiesRegistered:
    def test_all_seven_actions_are_registered_strategies(self):
        selector = RLDefenseSelector()
        assert len(selector.actions) == 7
        for action in selector.actions:
            assert action in _STRATEGIES
            # must be possible to instantiate a FederatedLearner with each strategy
            FederatedLearner(n_rounds=1, aggregation=action, n_classes=2)


class TestRLDefenseSelector:
    def test_select_action_returns_valid_action(self):
        selector = RLDefenseSelector()
        action, q_value = selector.select_action(attack_type=3, confidence=0.9)
        assert action in selector.actions
        assert isinstance(q_value, float)

    def test_select_action_accepts_rich_context(self):
        """5-dimensional state: attack_idx, confidence, and the previous
        round's context (acc/fairness_std/latency) — the last 3 default to
        0.0 for callers that don't track round history (e.g. standalone
        pretraining episodes)."""
        selector = RLDefenseSelector()
        action, q_value = selector.select_action(
            attack_type=3, confidence=0.9,
            prev_accuracy=0.85, prev_fairness_std=0.05, prev_latency=1.2,
        )
        assert action in selector.actions
        assert isinstance(q_value, float)

    def test_build_state_has_five_dims(self):
        state = RLDefenseSelector.build_state(2, 0.9, 0.8, 0.1, 1.5)
        assert state.shape == (RLDefenseSelector.N_STATE_DIMS,)
        np.testing.assert_allclose(state, [2.0, 0.9, 0.8, 0.1, 1.5])

    def test_select_action_without_round_num_is_always_greedy(self):
        """Without `round_num`, select_action must be 100% greedy (no
        exploration) — the behavior used by callers that don't track round
        (e.g. standalone pretraining episodes)."""
        selector = RLDefenseSelector(seed=0)
        actions_seen = {selector.select_action(2, 0.9)[0] for _ in range(30)}
        assert len(actions_seen) == 1  # always the same action (deterministic greedy)

    def test_select_action_explores_with_round_num_early_on(self):
        """With a low `round_num` (epsilon high, close to epsilon_start),
        select_action should vary its action across several calls — proof
        that ε-greedy exploration is actually active."""
        selector = RLDefenseSelector(seed=0, epsilon_start=1.0, epsilon_end=1.0)
        actions_seen = {selector.select_action(2, 0.9, round_num=1)[0] for _ in range(30)}
        assert len(actions_seen) > 1  # epsilon=1.0 -> always random -> should vary

    def test_epsilon_decays_with_round_num(self):
        selector = RLDefenseSelector(epsilon_start=0.3, epsilon_end=0.02, epsilon_decay_rounds=20)
        assert selector._epsilon(1) > selector._epsilon(10) > selector._epsilon(20)
        assert selector._epsilon(20) == pytest.approx(0.02)
        assert selector._epsilon(1000) == pytest.approx(0.02)  # saturates at epsilon_end

    def test_train_step_converges_on_fixed_experience(self):
        """Classic DQN sanity check: repeating the SAME experience should make
        the loss (Bellman error) decrease, as Q(s,a) converges to the fixed
        target."""
        selector = RLDefenseSelector(target_update_every=10)
        state = RLDefenseSelector.build_state(2.0, 0.9, 0.8, 0.05, 1.0)
        next_state = RLDefenseSelector.build_state(2.0, 0.9, 0.82, 0.04, 0.9)

        losses = [
            selector.train_step(state, action_idx=0, reward=1.0, next_state=next_state)
            for _ in range(100)
        ]
        assert np.mean(losses[-10:]) < np.mean(losses[:10])

    def test_train_step_uses_replay_buffer_and_target_network(self):
        """`train_step` must populate the replay buffer and periodically
        synchronize the target network."""
        selector = RLDefenseSelector(target_update_every=5, batch_size=4)
        state = RLDefenseSelector.build_state(1, 0.5)
        next_state = RLDefenseSelector.build_state(1, 0.5, 0.5)

        assert len(selector.replay_buffer) == 0
        for _ in range(5):
            selector.train_step(state, action_idx=0, reward=0.5, next_state=next_state)
        assert len(selector.replay_buffer) == 5
        target_weights = selector.target_model.get_weights()
        model_weights = selector.model.get_weights()
        for tw, mw in zip(target_weights, model_weights):
            np.testing.assert_allclose(tw, mw)

    def test_save_and_load_roundtrip(self, tmp_path):
        selector = RLDefenseSelector()
        selector.select_action(1, 0.5)  # force build
        path = str(tmp_path / "selector.weights.h5")
        selector.save(path)

        loaded = RLDefenseSelector()
        loaded.load(path)

        action1, q1 = selector.select_action(4, 0.7)
        action2, q2 = loaded.select_action(4, 0.7)
        assert action1 == action2
        assert q1 == pytest.approx(q2, rel=1e-5)


class TestGenerateSelectorExperiences:
    """Integration test: generates real experiences by running actual FL
    under each (attack type, strategy) combination, and trains the selector
    on them."""

    def test_experience_count_and_reward_range(self):
        participants = _make_synthetic_participants()
        selector = RLDefenseSelector()
        attack_types = ["none", "sign_flipping", "poisoning"]

        experiences = generate_selector_experiences(
            participants, selector.actions, attack_types=attack_types,
            n_rounds=5, byzantine_fraction=0.3, n_classes=2, seed=1,
        )

        assert len(experiences) == len(attack_types) * len(selector.actions)
        for state, action_idx, reward, next_state in experiences:
            assert state.shape == (RLDefenseSelector.N_STATE_DIMS,)
            assert not np.array_equal(state, next_state)  # next_state must reflect real post-episode outcome
            assert 0 <= action_idx < len(selector.actions)
            assert -1.0 <= reward <= 2.0

    def test_selector_trains_on_real_experiences(self):
        participants = _make_synthetic_participants()
        selector = RLDefenseSelector()
        experiences = generate_selector_experiences(
            participants, selector.actions,
            attack_types=["none", "sign_flipping", "poisoning"],
            n_rounds=5, byzantine_fraction=0.3, n_classes=2, seed=1,
        )
        losses = selector.train_on_experiences(experiences, verbose=False)
        assert len(losses) == len(experiences)
        assert all(np.isfinite(losses))
