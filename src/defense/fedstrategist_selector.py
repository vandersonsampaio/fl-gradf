"""
FedStrategist: reproduction of the meta-learning framework proposed in:

    Haque, Kamal & Hossain,
    "FedStrategist: A Meta-Learning Framework for Adaptive and Robust
    Aggregation in Federated Learning", 2025.

Used as a competing baseline in Gate 1
(``src/experiments/exp4_adaptive.py``), alongside TARS
(``src/defense/tars_selector.py``) and AdaBFL
(``src/defense/adabfl_aggregator.py``).

FedStrategist is not part of the GRADF pipeline and is not imported by
``src/fl/gradf_learner.py``.

The implementation reproduces the paper's Section 4 formulation using
the architecture of this repository rather than the authors' external
simulation framework. This adaptation is necessary because the external
implementation uses a PyTorch/CIFAR-10 stack, whereas this repository
represents models as flat NumPy parameter vectors.

Main components:

1. Diagnostic state vector ``S_t ∈ R^3``:

   - variance of client-update L2 norms;
   - average pairwise cosine similarity;
   - L2 norm of the mean client update.

2. Disjoint LinUCB contextual bandit over the Defense Arsenal:

   - FedAvg → ``fedavg``;
   - Coordinate-wise Median → ``median``;
   - Krum → ``krum``.

3. Reward:

      R_t = (Acc_t - Acc_{t-1}) - lambda_cost * C_j

   with the paper's heuristic defense costs:

      C_fedavg = 0.1
      C_median = 0.4
      C_krum  = 0.8

   ``lambda_cost`` defaults to ``0.5``, corresponding to the paper's
   balanced configuration.

The paper specifies the LinUCB framework but leaves the numerical
implementation as a standard NumPy/Scikit-learn formulation. This
implementation uses the standard disjoint-LinUCB updates:

      A_a <- A_a + x x^T
      b_a <- b_a + r x
      theta_a <- A_a^{-1} b_a

This is algebraically consistent with the corresponding regularized
linear regression formulation and preserves the method described in
the paper without introducing a different adaptation mechanism.

The implementation therefore constitutes a from-spec reproduction of
FedStrategist within this repository's federated-learning architecture,
with the model representation and execution framework adapted to the
existing ``_LogisticModel``/``_CNNModel`` and ``_STRATEGIES`` interfaces.
"""

from typing import Dict, List, Tuple

import numpy as np


class DiagnosticStateVector:
    """Computes S_t in R^3 from the round's client updates."""

    @staticmethod
    def compute(updates: List[np.ndarray]) -> Tuple[float, float, float]:
        stacked = np.stack(updates, axis=0)
        norms = np.linalg.norm(stacked, axis=1)

        variance_of_norms = float(np.var(norms))

        n = len(updates)
        sims: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                denom = norms[i] * norms[j] + 1e-12
                sims.append(float(np.dot(stacked[i], stacked[j]) / denom))
        avg_pairwise_cosine = float(np.mean(sims)) if sims else 0.0

        mean_update_norm = float(np.linalg.norm(stacked.mean(axis=0)))

        return (variance_of_norms, avg_pairwise_cosine, mean_update_norm)


class LinUCBAgent:
    """Disjoint LinUCB: per-action linear reward model,
    action = argmax_a (x^T theta_a + alpha*sqrt(x^T A_a^-1 x)). Context x is
    the 3-dim state vector plus a bias term (d=4)."""

    def __init__(self, actions: List[str], alpha: float = 1.5, context_dim: int = 3, seed: int = 0):
        self.actions = actions
        self.alpha = alpha
        self.d = context_dim + 1  # + bias
        self.A: Dict[str, np.ndarray] = {a: np.eye(self.d) for a in actions}
        self.b: Dict[str, np.ndarray] = {a: np.zeros(self.d) for a in actions}
        self._rng = np.random.RandomState(seed)

    @staticmethod
    def _featurize(state: Tuple[float, float, float]) -> np.ndarray:
        return np.concatenate([[1.0], np.asarray(state, dtype=np.float64)])

    def select_action(self, state: Tuple[float, float, float]) -> str:
        x = self._featurize(state)
        best_score = -np.inf
        best_action = self.actions[0]
        for a in self.actions:
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            ucb = self.alpha * float(np.sqrt(max(x @ A_inv @ x, 0.0)))
            score = float(x @ theta_hat) + ucb
            if score > best_score:
                best_score, best_action = score, a
        return best_action

    def update(self, state: Tuple[float, float, float], action: str, reward: float) -> None:
        x = self._featurize(state)
        self.A[action] = self.A[action] + np.outer(x, x)
        self.b[action] = self.b[action] + reward * x
