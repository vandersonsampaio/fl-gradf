"""
AdaBFLAggregator: reproduction of AdaBFL, proposed in:

    Tang, Liu & Huang,
    "Multi-Layer Defensive Adaptive Aggregation for Byzantine-Robust
    Federated Learning", 2026.

Used as a competing baseline in Gate 1
(``src/experiments/exp4_adaptive.py``), alongside TARS
(``src/defense/tars_selector.py``).

AdaBFL is not part of the GRADF pipeline and is not imported by
``src/fl/gradf_learner.py``.

This implementation follows the paper's Algorithm 1 (AdaBFL Algorithm)
and Algorithm 2 (Weight Update Algorithm), using the AdaBFL-1
configuration reported for the main Table 1 results.

Main pipeline:

1. Malicious-client filtering (Eq. 2)
2. Parameter clipping using Trimmed-Mean (Eq. 3-4)
3. Derivative model computation (Eq. 5-8)
4. Adaptive fusion of the three components using beta1, beta2 and beta3
   (Eq. 9-12)

The paper does not fully specify several parameters and implementation
details. The following choices are therefore explicit instantiations:

- ``lambda(t) = 1/t`` is used for the filter decay, with
  ``gamma = 1.5`` and ``kappa_filter = 1.0``.

- ``p2_t`` is implemented as the normalized distance between the mean
  benign-client model and the fused model:

      p2_t = (1/d) * ||mean(theta_i) - theta_bar||

  The paper's rendered equation contains an apparent typesetting error.

- The beta3 update follows Algorithm 2's pseudocode, which uses the
  condition ``p2_t < rho2``. This is preferred over the contradictory
  condition stated in the surrounding text.

- Thresholds and weight constraints in Algorithm 2
  (``rho1``, ``rho2``, ``delta_high``, ``delta_low``,
  ``beta*_min``, ``beta2_max``, ``beta1_base`` and ``kappa_weight``)
  are instantiated with values within the ranges explored in the
  paper's parameter study.

- ``use_momentum=True`` enables the momentum variant described in
  Algorithm 4, applying EMA smoothing to ``rho1`` and ``rho2`` before
  the weight-update procedure. It is disabled by default because the
  main results in Table 1 use Algorithms 1 and 2 without momentum.

The implementation is therefore a good-faith reproduction of the
paper's specified AdaBFL mechanism, with the unspecified or internally
inconsistent details made explicit rather than left implicit.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class AdaBFLAggregator:
    """Stateful, per-round adaptive Byzantine-robust aggregator. Operates on
    FULL client parameter vectors (theta_t^i = global_params + update_i),
    reconstructed from the flat deltas this codebase's clients actually
    send — see `AttackedFederatedLearner._compute_param_updates` — then
    returns a DELTA (theta_t - global_params) so it composes with the same
    `self._global_params = self._global_params + agg_delta` pattern used
    everywhere else in `src/fl/`."""

    def __init__(
        self,
        gamma: float = 1.5,
        kappa_filter: float = 1.0,
        trim_ratio: float = 0.2,
        n_synthetic: int = 1,
        rho1: float = 0.02,
        rho2: float = 0.5,
        delta_high: float = 0.05,
        delta_low: float = 0.05,
        beta1_min: float = 0.1,
        beta2_max: float = 0.6,
        beta3_min: float = 0.05,
        beta1_base: float = 0.3,
        kappa_weight: float = 0.3,
        use_momentum: bool = False,
        momentum_alpha: float = 0.9,
    ) -> None:
        self.gamma = gamma
        self.kappa_filter = kappa_filter
        self.trim_ratio = trim_ratio
        self.n_synthetic = n_synthetic
        self.rho1 = rho1
        self.rho2 = rho2
        self.delta_high = delta_high
        self.delta_low = delta_low
        self.beta1_min = beta1_min
        self.beta2_max = beta2_max
        self.beta3_min = beta3_min
        self.beta1_base = beta1_base
        self.kappa_weight = kappa_weight
        self.use_momentum = use_momentum
        self.momentum_alpha = momentum_alpha

        self.beta1, self.beta2, self.beta3 = 1.0 / 3, 1.0 / 3, 1.0 / 3

    # -- Filter malicious clients (Algorithm 1, lines 15-16; Eq. 2) --------

    def _filter_malicious(self, client_params: List[np.ndarray], round_num: int) -> List[int]:
        n = len(client_params)
        lam_t = 1.0 / max(round_num, 1)
        threshold_factor = (self.gamma * np.exp(-self.kappa_filter * lam_t)) / 2.0

        benign: List[int] = []
        for i in range(n):
            others = [client_params[j] for j in range(n) if j != i]
            mean_others = np.mean(others, axis=0)
            lhs = np.linalg.norm(client_params[i] - mean_others)
            rhs = threshold_factor * np.linalg.norm(mean_others + client_params[i])
            if lhs <= rhs:
                benign.append(i)
        return benign

    # -- Parameter clipping / Trimmed-mean (Algorithm 1, lines 17-18; Eq. 3-4)

    def _trimmed_mean(self, params: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        stacked = np.sort(np.stack(params, axis=0), axis=0)
        n, d = stacked.shape
        k = int(np.floor(n * self.trim_ratio))
        if 2 * k >= n:
            k = max(0, (n - 1) // 2)
        trimmed = stacked[k: n - k] if k > 0 else stacked
        theta_tilde = trimmed.mean(axis=0)

        mean_theta = np.stack(params, axis=0).mean(axis=0)
        p = float(np.linalg.norm(theta_tilde - mean_theta) / d)
        return theta_tilde, p

    # -- Derivative model (Algorithm 1, lines 19-20; Eq. 5-8) ---------------

    def _derivative_model(self, benign_params: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        stacked = np.stack(benign_params, axis=0)
        d = stacked.shape[1]
        theta_max = stacked.max(axis=0)
        theta_min = stacked.min(axis=0)

        scores = [
            min(float(np.linalg.norm(p - theta_max)), float(np.linalg.norm(p - theta_min)))
            for p in benign_params
        ]
        i_star = int(np.argmax(scores))
        synthetic = [benign_params[i_star]] * self.n_synthetic

        fused = benign_params + synthetic
        theta_bar, _ = self._trimmed_mean(fused)

        mean_benign = stacked.mean(axis=0)
        p2 = float(np.linalg.norm(mean_benign - theta_bar) / d)
        return theta_bar, p2

    # -- Weight update (Algorithm 2 / Algorithm 4) ---------------------------

    def _update_weights(self, p1: float, p2: float) -> None:
        if self.use_momentum:
            self.rho1 = self.momentum_alpha * self.rho1 + (1 - self.momentum_alpha) * p1
            self.rho2 = self.momentum_alpha * self.rho2 + (1 - self.momentum_alpha) * p2

        if p1 >= self.rho1:
            self.beta2 = min(self.beta2 + self.delta_high, self.beta2_max)
            self.beta1 = max(self.beta1 - self.delta_high, self.beta1_min)
        elif p2 < self.rho2:
            self.beta3 = max(self.beta3 - self.delta_low, self.beta3_min)
            self.beta2 = min(self.beta2 + self.delta_high, self.beta2_max)
        else:
            self.beta1 = self.beta1_base + self.kappa_weight * (1 - p1)

        total = self.beta1 + self.beta2 + self.beta3
        self.beta1, self.beta2, self.beta3 = self.beta1 / total, self.beta2 / total, self.beta3 / total

    # -- Public entry point ---------------------------------------------------

    def aggregate(
        self,
        global_params: np.ndarray,
        updates: List[np.ndarray],
        round_num: int,
        sample_sizes: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """`sample_sizes` accepted for interface parity with the other
        strategies' `.aggregate()` but unused — AdaBFL's own weighting
        (beta1/beta2/beta3) replaces sample-size weighting entirely."""
        del sample_sizes
        client_params = [global_params + u for u in updates]

        benign_idx = self._filter_malicious(client_params, round_num)
        if not benign_idx:
            benign_idx = list(range(len(client_params)))
        benign_params = [client_params[i] for i in benign_idx]

        theta_tilde, p1 = self._trimmed_mean(benign_params)
        theta_bar, p2 = self._derivative_model(benign_params)
        self._update_weights(p1, p2)

        mean_benign = np.mean(benign_params, axis=0)
        theta_t = self.beta1 * mean_benign + self.beta2 * theta_tilde + self.beta3 * theta_bar

        agg_delta = theta_t - global_params
        metadata = {
            "benign_indices": benign_idx,
            "beta1": self.beta1, "beta2": self.beta2, "beta3": self.beta3,
            "p1": p1, "p2": p2,
        }
        return agg_delta, metadata
