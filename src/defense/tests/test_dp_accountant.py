import numpy as np
import pytest

from src.defense.dp_accountant import (
    GaussianDPMechanism,
    calibrate_noise_multiplier,
    sensitivity_for_rule,
)
from src.defense.hardening import HardeningPipeline


class _StubClassifier:
    def classify(self, scores):
        return 0, 1.0  # always "none", maximum confidence


class _StubSelector:
    def __init__(self, action="fedavg"):
        self.action = action

    def select_action(self, attack_idx, confidence):
        return self.action, 1.0


class TestCalibration:
    def test_noise_multiplier_increases_epsilon_when_decreased(self):
        """Less noise (smaller noise_multiplier) should cost more privacy
        (larger epsilon) for the same (delta, n_rounds) — a direct sanity
        check of the monotonicity the RDP accountant must respect."""
        sigma_loose = calibrate_noise_multiplier(target_epsilon=20.0, target_delta=1e-5, n_rounds=20)
        sigma_tight = calibrate_noise_multiplier(target_epsilon=2.0, target_delta=1e-5, n_rounds=20)
        assert sigma_tight > sigma_loose  # tighter budget requires more noise

    def test_calibration_round_trips_through_accountant(self):
        """The calibrated noise_multiplier, when actually composed n_rounds
        times in the accountant, should reproduce (approximately) the target
        epsilon — not an arbitrary value the calibrator "made up"."""
        target_epsilon, target_delta, n_rounds = 8.0, 1e-5, 15
        sigma = calibrate_noise_multiplier(target_epsilon, target_delta, n_rounds)

        mech = GaussianDPMechanism(
            l2_clip_norm=2.0, target_epsilon=target_epsilon,
            target_delta=target_delta, n_rounds=n_rounds, seed=0,
        )
        assert mech.noise_multiplier == pytest.approx(sigma, rel=1e-6)

        agg = np.zeros(10)
        for _ in range(n_rounds):
            mech.privatize_aggregate(agg, "fedavg", weights=[1, 1, 1])
        assert mech.current_epsilon() == pytest.approx(target_epsilon, abs=1e-3)

    def test_epsilon_grows_with_rounds_composed(self):
        mech = GaussianDPMechanism(l2_clip_norm=2.0, target_epsilon=8.0, target_delta=1e-5, n_rounds=20, seed=0)
        eps_after = []
        agg = np.zeros(5)
        for _ in range(5):
            mech.privatize_aggregate(agg, "fedavg", weights=[1, 1])
            eps_after.append(mech.current_epsilon())
        assert eps_after == sorted(eps_after)  # monotonically non-decreasing
        assert eps_after[-1] < 8.0  # hasn't spent the full budget yet (only 5 of 20 rounds)


class TestSensitivityByRule:
    def test_linear_rule_sensitivity_shrinks_with_more_clients(self):
        """fedavg/fedprox/fltrust: sensitivity decreases with more clients
        (smaller uniform weights) — O(1/N) bound, not the conservative O(1)."""
        few = sensitivity_for_rule("fedavg", l2_clip_norm=2.0, weights=[1, 1])
        many = sensitivity_for_rule("fedavg", l2_clip_norm=2.0, weights=[1] * 20)
        assert many < few

    def test_nonlinear_rule_uses_conservative_bound(self):
        """median/trimmed_mean/krum/clustering: fixed conservative bound
        (2*l2_clip_norm), regardless of the number of clients."""
        few = sensitivity_for_rule("median", l2_clip_norm=2.0, weights=[1, 1])
        many = sensitivity_for_rule("median", l2_clip_norm=2.0, weights=[1] * 20)
        assert few == many == pytest.approx(4.0)

    def test_conservative_bound_never_smaller_than_linear_bound(self):
        """The conservative bound must never underestimate the sensitivity of
        the linear rules — underestimating would be the dangerous error
        (overestimated privacy)."""
        weights = [1] * 10
        linear = sensitivity_for_rule("fedavg", l2_clip_norm=2.0, weights=weights)
        conservative = sensitivity_for_rule("krum", l2_clip_norm=2.0, weights=weights)
        assert conservative >= linear


class TestHardeningPipelineIntegration:
    def test_default_pipeline_unaffected_by_new_mechanism(self):
        """Without dp_mechanism=, the pipeline uses the legacy Layer 2
        (Laplace) — default behavior preserved, no existing experiment
        changes."""
        pipeline = HardeningPipeline(_StubClassifier(), _StubSelector())
        assert pipeline.dp_mechanism is None

    def test_gaussian_mechanism_clips_without_per_client_noise(self):
        """With dp_mechanism=, Layer 2 only clips (does not add per-client
        noise) — the noise is added once, to the aggregated output."""
        mech = GaussianDPMechanism(l2_clip_norm=1.0, target_epsilon=8.0, target_delta=1e-5, n_rounds=10, seed=0)
        pipeline = HardeningPipeline(_StubClassifier(), _StubSelector(), dp_mechanism=mech)

        big_update = np.ones(5) * 10.0  # L2 norm >> clip_norm=1.0
        clipped = pipeline.dp_mechanism.clip(big_update)
        assert np.linalg.norm(clipped) == pytest.approx(1.0)
        # Deterministic: pure clip, no noise.
        expected_direction = big_update / np.linalg.norm(big_update)
        np.testing.assert_allclose(clipped, expected_direction, atol=1e-10)

    def test_full_pipeline_with_gaussian_mechanism_runs_and_tracks_budget(self):
        mech = GaussianDPMechanism(l2_clip_norm=5.0, target_epsilon=50.0, target_delta=1e-5, n_rounds=5, seed=0)
        pipeline = HardeningPipeline(
            _StubClassifier(), _StubSelector("fedavg"),
            magnitude_threshold=1e6, dp_mechanism=mech,
        )
        rng = np.random.default_rng(0)
        updates = [rng.standard_normal(8) * 0.1 for _ in range(4)]
        modality_scores = [np.zeros(4) for _ in range(4)]

        def evaluate_fn(delta):
            return 0.5  # constant: we're only testing that the pipeline doesn't break/reject

        agg_delta, results = pipeline.full_pipeline(
            ["h0", "h1", "h2", "h3"], updates, modality_scores, evaluate_fn,
            sample_sizes=[10, 10, 10, 10],
        )
        assert results["round_accepted"] is True
        assert agg_delta is not None
        assert "dp_budget" in results
        assert results["dp_budget"]["rounds_composed"] == 1
