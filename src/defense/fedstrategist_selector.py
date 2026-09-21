"""
FedStrategist: reproduction of the meta-learning framework from "FedStrategist:
A Meta-Learning Framework for Adaptive and Robust Aggregation in Federated
Learning" (Haque, Kamal & Hossain, 2025 — see
`references/citations/FedStrategist: A Meta-Learning Framework for Adaptive
and Robust Aggregation in Federated Learning.pdf`), used as a COMPETING
BASELINE in Gate 1 (`src/experiments/exp4_adaptive.py`), alongside TARS
(`src/defense/tars_selector.py`) and AdaBFL
(`src/defense/adabfl_aggregator.py`). Not part of the GRADF pipeline, and
not imported by `src/fl/gradf_learner.py`.

Unlike AdaBFL, this paper's Supporting Information links a public repo
(`https://github.com/rafidhaque/FedStrategist`, archived on Zenodo,
DOI 10.5281/zenodo.16068113) — however it targets a standalone PyTorch/
CIFAR-10 simulation harness (its own `fl_core.py`/`aggregation.py`/
`bandit.py`) built around a different model representation than this
repo's flat-numpy-vector `_LogisticModel`/`_CNNModel` (see
`src/fl/federated_learner.py`), so porting it wholesale would not compose
with `AttackedFederatedLearner`/`_STRATEGIES`. The paper's algorithm
(Section 4, "Materials and Methods") is fully specified in closed form —
LinUCB with a documented state vector, reward, and hyperparameters (S1
Appendix) — so this is a from-spec reproduction inside this repo's own
architecture, not a port of that external code.

Reproduces the paper's 3 components (Section 4.4/4.5):
  1. Diagnostic state vector S_t in R^3: variance of update L2 norms,
     average pairwise cosine similarity, L2 norm of the mean update.
  2. LinUCB contextual bandit over the Defense Arsenal (paper Section 4.2:
     FedAvg, Coordinate-wise Median, Krum — mapped onto this repo's
     'fedavg'/'median'/'krum' entries in `src.fl.federated_learner._STRATEGIES`
     / `src.defense.aggregation_methods`).
  3. Reward R_t = (Acc_t - Acc_{t-1}) - lambda_cost * C_j, with the exact
     heuristic costs from the paper's S1 Appendix: C_fedavg=0.1,
     C_median=0.4, C_krum=0.8. lambda_cost defaults to the paper's
     "balanced" value (0.5) from Table S2/S4.

One detail the paper leaves as an implementation choice ("a standard
implementation of the LinUCB algorithm using NumPy and Scikit-learn's Ridge
regression for stable linear modeling", Section 5.3) rather than a closed
formula: we implement the standard disjoint-LinUCB update directly
(A_a += x x^T, b_a += r*x, theta_a = A_a^-1 b_a) — algebraically the same
fixed point Ridge regression with alpha=1 regularization converges to, so
this is a faithful reproduction of the described method, not a
reinterpretation of it.
"""

from typing import Dict, List, Tuple

import numpy as np


class DiagnosticStateVector:
    """Computes S_t in R^3 from the round's client updates (paper Section 4.4)."""

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
    """Disjoint LinUCB (paper Section 4.5): per-action linear reward model,
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
