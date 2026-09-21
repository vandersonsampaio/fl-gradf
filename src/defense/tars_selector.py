"""
TARSSelector: reproduction of the "Trust-Aware Reinforcement Selection"
framework (Ahmed et al., 2025 — see
`references/Trust-Aware_Reinforcement_Selection_for_Robust_Federated_Learning_under_Adaptive_Adversaries.pdf`),
used as a COMPETING BASELINE in Gate 1 (`src/experiments/exp4_adaptive.py`).
Not part of the GRADF pipeline, and not imported by `src/fl/gradf_learner.py`.

Reproduces the 4 components described in the paper (Section V): (i) per-client
trust inference, (ii) state encoding, (iii) tabular ε-greedy Q-learning over
the set of candidate aggregation rules, (iv) execution of the chosen
aggregation (Algorithm 1).

Two pieces are NOT specified as a closed-form formula by the paper and
required our own instantiation, documented explicitly here — this is OUR
good-faith reproduction of the described architecture, not the authors' code:

  1. The scoring function φ(·) that combines the three trust signals (loss
     divergence, cosine similarity, magnitude deviation) into τ ∈ [0,1] — the
     paper describes the three criteria (Section V-B) but leaves φ generic
     ("bounded scoring function"). We use a product of three factors, each
     already normalized to [0,1] by construction (see `TrustScorer.score`).
  2. The discretization of the continuous state s_t = [Acc, Loss, mean trust]
     into Q-table cells — the paper uses "Q-table" (Algorithm 1) but does not
     publish the bin boundaries. We use 3 fixed bins per dimension (see
     `TARSSelector._discretize`).
"""

from typing import Dict, List, Tuple

import numpy as np


def cross_entropy_loss(model, X: np.ndarray, y: np.ndarray) -> float:
    """Log-loss of the model on (X, y) — used as L(·, D_val) both in the trust
    score's loss divergence and in TARS's reward. `_LogisticModel`
    (src/fl/federated_learner.py) does not expose loss directly, only
    predict_proba/accuracy, so it is computed here."""
    proba = model.predict_proba(X)
    eps = 1e-12
    if proba.ndim == 1:  # binary (sigmoid)
        p = np.clip(proba, eps, 1 - eps)
        y = y.astype(np.float64)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    p = np.clip(proba[np.arange(len(y)), y], eps, 1 - eps)  # softmax
    return float(-np.mean(np.log(p)))


class TrustScorer:
    """Per-client trust score (paper Section V-B): loss divergence, cosine
    similarity with the global model, and update magnitude deviation —
    combined into τ_i^(t) ∈ [0,1] and smoothed over time:

        τ̂_i^(t) = β·τ̂_i^(t-1) + (1-β)·τ_i^(t)
    """

    def __init__(self, beta: float = 0.7, alpha_loss: float = 1.0, alpha_mag: float = 0.05):
        self.beta = beta
        self.alpha_loss = alpha_loss
        self.alpha_mag = alpha_mag
        self._trust_hat: Dict[str, float] = {}

    def score(
        self,
        hospital_id: str,
        global_params: np.ndarray,
        candidate_params: np.ndarray,
        delta: np.ndarray,
        loss_divergence: float,
    ) -> float:
        cos_sim = float(
            np.dot(global_params, candidate_params)
            / (np.linalg.norm(global_params) * np.linalg.norm(candidate_params) + 1e-12)
        )
        f_loss = float(np.exp(-self.alpha_loss * max(loss_divergence, 0.0)))  # worse loss -> less trust
        f_cos = (cos_sim + 1.0) / 2.0                                          # misaligned -> less trust
        f_mag = float(np.exp(-self.alpha_mag * np.linalg.norm(delta)))        # huge update -> less trust
        tau_instant = f_loss * f_cos * f_mag

        prev = self._trust_hat.get(hospital_id, 1.0)
        tau_hat = self.beta * prev + (1.0 - self.beta) * tau_instant
        self._trust_hat[hospital_id] = tau_hat
        return tau_hat


class TARSSelector:
    """Tabular ε-greedy Q-learning over the set of candidate rules (paper
    Section V-C/V-E, Algorithm 1). State discretized into 3 bins per
    dimension (see caveat 2 in the module docstring)."""

    def __init__(
        self,
        actions: List[str],
        lr: float = 0.1,
        gamma: float = 0.9,
        eps_start: float = 0.3,
        eps_end: float = 0.02,
        n_rounds: int = 15,
        seed: int = 0,
    ):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.n_rounds = max(n_rounds, 1)
        self.q: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._rng = np.random.RandomState(seed)

    @staticmethod
    def _bin(value: float, edges: List[float]) -> int:
        for i, e in enumerate(edges):
            if value < e:
                return i
        return len(edges)

    def _discretize(self, state: Tuple[float, float, float]) -> Tuple[int, int, int]:
        acc, loss, mean_trust = state
        return (
            self._bin(acc, [0.5, 0.8]),
            self._bin(loss, [1.0, 2.0]),
            self._bin(mean_trust, [0.33, 0.66]),
        )

    def _epsilon(self, round_num: int) -> float:
        """Linear decay from eps_start to eps_end over n_rounds (paper
        Section V-E: "decaying ε-greedy exploration policy")."""
        frac = min(round_num / self.n_rounds, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: Tuple[float, float, float], round_num: int) -> str:
        key = self._discretize(state)
        q_values = self.q.setdefault(key, np.zeros(len(self.actions)))
        if self._rng.rand() < self._epsilon(round_num):
            idx = self._rng.randint(len(self.actions))
        else:
            idx = int(np.argmax(q_values))
        return self.actions[idx]

    def update(
        self,
        state: Tuple[float, float, float],
        action: str,
        reward: float,
        next_state: Tuple[float, float, float],
    ) -> None:
        """Standard Bellman update (paper Section V-C):
        Q(s,a) <- Q(s,a) + η[R + γ·max_a' Q(s',a') - Q(s,a)]"""
        key = self._discretize(state)
        next_key = self._discretize(next_state)
        q_values = self.q.setdefault(key, np.zeros(len(self.actions)))
        next_q = self.q.setdefault(next_key, np.zeros(len(self.actions)))
        idx = self.actions.index(action)
        td_target = reward + self.gamma * next_q.max()
        q_values[idx] += self.lr * (td_target - q_values[idx])
