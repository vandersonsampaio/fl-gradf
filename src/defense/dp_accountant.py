"""
Gaussian mechanism with RDP accounting for Layer 2 of HardeningPipeline.

Replaces the legacy per-coordinate Laplace mechanism used by
``HardeningPipeline.layer2_gradient_sanitization`` with a standard
Gaussian mechanism. The legacy mechanism remains the default for
backward compatibility.

The mechanism uses the ``dp_accounting`` library for RDP composition
across rounds. Privacy accounting is delegated to the library rather
than implemented manually.

Privacy model:

- Privacy unit: CLIENT-LEVEL. Each cross-silo hospital is treated as
  one privacy unit. Record-level privacy is out of scope because it
  would require a per-sample mechanism inside local training.

- Trust model: MODEL-RELEASE DP. Noise protects observers of the global
  model released at the end of each round, such as other hospitals,
  external attackers, or model repositories. It does not protect
  against the server itself.

The server processes client updates in plaintext before Layer 2, since
Layers 1, 3, and 4 require access to the updates for anomaly detection
and aggregation. Therefore, this mechanism does not provide
server-side privacy.

Sensitivity:

All aggregation strategies in ``src.fl.federated_learner._STRATEGIES``
operate on updates clipped to ``l2_clip_norm``. The L2 sensitivity of
the aggregated output depends on the aggregation rule.

- Weighted mean rules (``fedavg``, ``fedprox``, ``fltrust``):
  sensitivity is bounded by

      2 * w_max * l2_clip_norm

  where ``w_max`` is the largest normalized weight among accepted
  clients. With uniform weights, this approaches
  ``2 * l2_clip_norm / N``.

- Non-linear robust rules (``median``, ``trimmed_mean``, ``krum``,
  ``clustering``):
  no tight closed-form L2 sensitivity bound is assumed. The
  implementation uses the conservative bound

      2 * l2_clip_norm

  which is valid but generally not tight. This avoids underestimating
  sensitivity and therefore avoids overstating the resulting privacy
  guarantee.

The conservative bound is intentional and should be treated as a
documented implementation assumption rather than an exact sensitivity
characterization of the robust aggregation rules.

The resulting Gaussian noise is composed across rounds using RDP and
converted to an ``(epsilon, delta)`` guarantee according to the
configured privacy parameters.
"""

from typing import Dict, Optional, Sequence

import dp_accounting
import numpy as np

_LINEAR_STRATEGIES = frozenset({"fedavg", "fedprox", "fltrust"})


def calibrate_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    n_rounds: int,
    bracket: "tuple[float, float]" = (1e-3, 1e3),
) -> float:
    """Finds the `noise_multiplier` (noise standard deviation in units of L2
    sensitivity) such that composing a `GaussianDpEvent(noise_multiplier)`
    `n_rounds` times (one FL round = one application of the mechanism) spends
    exactly `target_epsilon` under `target_delta`, via an RDP accountant
    (`dp_accounting.rdp.RdpAccountant`, the same kind of accounting used in
    reference implementations such as TensorFlow Privacy)."""

    def make_fresh_accountant():
        return dp_accounting.rdp.RdpAccountant()

    def make_event_from_param(noise_multiplier: float):
        return dp_accounting.SelfComposedDpEvent(
            dp_accounting.GaussianDpEvent(noise_multiplier), n_rounds
        )

    return float(dp_accounting.calibrate_dp_mechanism(
        make_fresh_accountant, make_event_from_param,
        target_epsilon=target_epsilon, target_delta=target_delta,
        bracket_interval=dp_accounting.ExplicitBracketInterval(*bracket),
    ))


def sensitivity_for_rule(strategy_name: str, l2_clip_norm: float, weights: Optional[Sequence[float]]) -> float:
    """L2 sensitivity (under substitution of one accepted client) of the
    `strategy_name` aggregation rule, given that every input update has
    already been clipped to `l2_clip_norm` — see the module docstring for
    the per-rule justification."""
    if strategy_name in _LINEAR_STRATEGIES and weights:
        total = float(sum(weights))
        w_max = (max(weights) / total) if total > 0 else 1.0
        return 2.0 * w_max * l2_clip_norm
    # Robust/non-linear rules (median, trimmed_mean, krum, clustering) or
    # unavailable weights: conservative bound — see the module docstring.
    return 2.0 * l2_clip_norm


class GaussianDPMechanism:
    """Client-level DP Gaussian mechanism for the aggregated output of an FL
    round, with real (ε,δ) accounting via RDP composed over `n_rounds`.
    Intended use: `HardeningPipeline(dp_mechanism=...)` — see the
    `HardeningPipeline.__init__` docstring. Opt-in: the pipeline's default
    behavior (without `dp_mechanism`) remains the legacy per-coordinate
    Laplace noise, so already-reported results (exp1/exp2/exp4/exp7) are
    not silently changed."""

    def __init__(
        self,
        l2_clip_norm: float,
        target_epsilon: float,
        target_delta: float,
        n_rounds: int,
        seed: Optional[int] = None,
    ) -> None:
        self.l2_clip_norm = l2_clip_norm
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.n_rounds = n_rounds
        self.noise_multiplier = calibrate_noise_multiplier(target_epsilon, target_delta, n_rounds)
        self._accountant = dp_accounting.rdp.RdpAccountant()
        self._rounds_composed = 0
        self._rng = np.random.default_rng(seed)

    def clip(self, update: np.ndarray) -> np.ndarray:
        """L2 clip (does not add noise — the key difference from the legacy
        Layer 2, which clipped AND added noise per client; here the noise is
        added ONCE, to the aggregated output — see the module docstring)."""
        norm = float(np.linalg.norm(update))
        if norm <= self.l2_clip_norm or norm == 0.0:
            return update
        return update * (self.l2_clip_norm / norm)

    def privatize_aggregate(
        self, aggregate: np.ndarray, strategy_name: str, weights: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Adds calibrated Gaussian noise to the ALREADY AGGREGATED output
        (Layer 3) and records one round of privacy spend in the accountant.
        Call once per FL round — calling it more than once per round spends
        privacy budget not accounted for by the calibration's `n_rounds`."""
        sensitivity = sensitivity_for_rule(strategy_name, self.l2_clip_norm, weights)
        std = self.noise_multiplier * sensitivity
        noise = self._rng.normal(0.0, std, size=aggregate.shape)
        self._accountant.compose(dp_accounting.GaussianDpEvent(self.noise_multiplier))
        self._rounds_composed += 1
        return aggregate + noise

    def current_epsilon(self, delta: Optional[float] = None) -> float:
        """(ε,δ) actually spent so far (can be queried at any point during
        training, not just at the end)."""
        return float(self._accountant.get_epsilon(delta if delta is not None else self.target_delta))

    def budget_status(self) -> Dict[str, float]:
        return {
            "rounds_composed": self._rounds_composed,
            "n_rounds_budgeted": self.n_rounds,
            "noise_multiplier": self.noise_multiplier,
            "target_epsilon": self.target_epsilon,
            "current_epsilon": self.current_epsilon(),
        }
