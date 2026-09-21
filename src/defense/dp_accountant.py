"""
Gaussian mechanism + RDP accounting for Layer 2 of the HardeningPipeline —
Decision B1 (Option A) of `references/plano_gradf_iclr2027.md`.

Replaces the legacy per-coordinate Laplace noise of `HardeningPipeline.
layer2_gradient_sanitization` (kept as the default, for backward
compatibility — see `hardening.py`) with a standard Gaussian mechanism,
calibrated to the correct L2 sensitivity and composed over T rounds via a
real RDP accountant (the `dp_accounting` library, Google — the privacy
composition is not hand-rolled, since a silent error there would produce
an incorrect (ε,δ)-DP claim).

Trust model and privacy unit (declared explicitly, per the plan's
requirement):

  - Privacy unit: CLIENT-LEVEL (each cross-silo hospital is one unit, not
    an individual patient/record within the hospital — record-level DP
    would require a per-sample mechanism inside local training, out of
    scope for this submission).
  - Trust model: MODEL-RELEASE DP. The noise protects whoever observes the
    global model published at the end of the round (a competing hospital,
    an external attacker, a repository where the model is distributed) —
    NOT an honest-but-curious server. The server already inspects updates
    in the clear in Layers 1/(detection)/3/4 before the noise is added
    (anomaly detection NEEDS this — Byzantine behavior cannot be detected
    over encrypted/noised data without destroying the signal), so this DP
    is not a defense against the server itself. This is consistent with
    the tension discussed for HE in `hardening.py`/the paper (Layer 5): DP
    and detection compete for the same plaintext data.

Sensitivity per aggregation rule (the missing piece needed to make the
mechanism correct, not just "call Gaussian instead of Laplace"): every
aggregation rule in `src.fl.federated_learner._STRATEGIES` operates on
updates ALREADY CLIPPED to `l2_clip_norm`, but the L2 sensitivity of the
aggregated OUTPUT, under substitution of one client, differs by rule:

  - (Weighted) mean rules — `fedavg`, `fedprox`, `fltrust`: the output is a
    linear combination of the updates; sensitivity under substitution of
    one client is `2 * w_max * l2_clip_norm`, where `w_max` is the largest
    normalized weight among the accepted ones (tends to `2*l2_clip_norm/N`
    with uniform weights) — a tight bound, standard in the DP-FedAvg
    literature.
  - Non-linear/robust rules — `median`, `trimmed_mean`, `krum`,
    `clustering`: these have no known tight sensitivity with a simple
    closed form (coordinate-wise median/trimmed-mean, and Krum's
    combinatorial selection, make the substitution-sensitivity argument
    much harder — an open research problem in its own right). We use the
    CONSERVATIVE bound `2 * l2_clip_norm` (in the worst case, the output
    can be entirely the clipped update of a single swapped client) — valid
    (it does not underestimate sensitivity, which would be the dangerous
    error), but not tight; documented explicitly as such, here and in the
    paper (§7).
"""

from typing import Dict, Optional, Sequence

import dp_accounting
import numpy as np

# Rules whose output is a linear (weighted) combination of the accepted
# updates — tight sensitivity via w_max. All others use the conservative bound.
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
