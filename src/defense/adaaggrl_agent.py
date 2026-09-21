"""
AdaAggRLAgent: a continuous-weighting adaptive-aggregation agent, the
SECOND competing baseline family (alongside discrete-selection agents like
TARS/FedStrategist/GRADF's DQN) required by
`references/experimento_seletores_adaptativos.md` and driven by
`src/experiments/exp10_selector_comparison.py`.

Unlike every other competing baseline in this codebase (TARS, AdaBFL,
FedStrategist), AdaAggRL has NO external paper to reproduce. The experiment
plan describes it only as (line 5): "AdaAggRL (RL/TD3 que aprende pesos
contínuos de agregação a partir de ~4 métricas)" — an RL/TD3 agent that
learns CONTINUOUS aggregation weights from ~4 diagnostic metrics, covering
the "continuous weighting" family as a contrast to GRADF/TARS/FedStrategist's
"discrete rule selection" family. This module is therefore a from-scratch,
good-faith INSTANTIATION of that one-line family description, not a
reproduction of a specific published method — flagged explicitly, per
condition 5 of the experiment plan ("Reproduções documentadas... distinguir
o que é deles do que é reprodução"), so nobody mistakes this for a citation.

Algorithm family: TD3 (Fujimoto, Hoof & Meger, "Addressing Function
Approximation Error in Actor-Critic Methods", ICML 2018), applied to a
single-step-per-round contextual-bandit setting (an FL round's aggregation
decision has no meaningful trajectory beyond "observe state, pick weights,
observe reward" — the same single-step-episode treatment already used
elsewhere in this codebase, e.g. `RLDefenseSelector`'s online updates and
`TARSSelector`).

Deliberate simplification, documented (mirrors an established precedent in
this codebase: `TARSSelector` uses TABULAR Q-learning instead of a DQN
because its state space is small — see `src/defense/tars_selector.py`):
actor and both critics are LINEAR function approximators (not deep nets).
With a ~4-dim state, a handful of training signals per FL round, and no
GPU-scale training budget, a deep TD3 network would be both unnecessary and
harder to train reliably than a linear one in this regime — the three
mechanisms that define TD3 (twin critics taking the min to fight Q
overestimation, delayed/less-frequent policy updates, and clipped noise
added to the target action for policy smoothing) are all still present,
just applied to linear Q(s,a)=[s;a]·w and actor(s)=softmax(s·W+b) instead
of MLPs. Gradients are closed-form (linear regression / softmax Jacobian),
computed directly with NumPy — no TensorFlow dependency for this module.
"""

from typing import List, Optional, Tuple

import numpy as np


class _ReplayBuffer:
    def __init__(self, capacity: int = 2000, seed: int = 0):
        self.capacity = capacity
        self._buf: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
        self._rng = np.random.RandomState(seed)

    def add(self, s: np.ndarray, w: np.ndarray, r: float, s2: np.ndarray) -> None:
        self._buf.append((s, w, r, s2))
        if len(self._buf) > self.capacity:
            self._buf.pop(0)

    def sample(self, batch_size: int):
        n = min(batch_size, len(self._buf))
        idx = self._rng.choice(len(self._buf), size=n, replace=False)
        return [self._buf[i] for i in idx]

    def __len__(self) -> int:
        return len(self._buf)


class AdaAggRLAgent:
    """Continuous-weighting TD3-family agent over a fixed arsenal of
    aggregation rules. `select_weights` returns a softmax weight vector
    (one weight per action, summing to 1) instead of a single discrete
    action — the caller blends the candidate strategies' own aggregated
    deltas by these weights (see `exp10_selector_comparison.py`)."""

    def __init__(
        self,
        actions: List[str],
        state_dim: int = 4,
        actor_lr: float = 0.05,
        critic_lr: float = 0.05,
        gamma: float = 0.9,
        tau: float = 0.1,
        policy_noise: float = 0.05,
        noise_clip: float = 0.1,
        policy_delay: int = 2,
        exploration_sigma: float = 0.15,
        batch_size: int = 8,
        buffer_capacity: int = 500,
        seed: int = 0,
    ) -> None:
        self.actions = actions
        self.n_actions = len(actions)
        self.state_dim = state_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploration_sigma = exploration_sigma
        self.batch_size = batch_size

        self._rng = np.random.RandomState(seed)
        feat_dim = state_dim + self.n_actions  # [state; weights] concatenated

        # Actor: logits = state @ W_actor + b_actor ; weights = softmax(logits)
        self.W_actor = self._rng.normal(0, 0.1, size=(state_dim, self.n_actions))
        self.b_actor = np.zeros(self.n_actions)
        self.W_actor_target = self.W_actor.copy()
        self.b_actor_target = self.b_actor.copy()

        # Twin critics: Q(s,w) = [s;w] @ Wc + bc
        self.Wc1 = self._rng.normal(0, 0.1, size=feat_dim)
        self.bc1 = 0.0
        self.Wc2 = self._rng.normal(0, 0.1, size=feat_dim)
        self.bc2 = 0.0
        self.Wc1_target, self.bc1_target = self.Wc1.copy(), self.bc1
        self.Wc2_target, self.bc2_target = self.Wc2.copy(), self.bc2

        self.buffer = _ReplayBuffer(capacity=buffer_capacity, seed=seed)
        self._n_updates = 0

    # -- forward passes ------------------------------------------------------

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max())
        return e / e.sum()

    def _actor(self, state: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._softmax(state @ W + b)

    def _critic(self, state: np.ndarray, weights: np.ndarray, Wc: np.ndarray, bc: float) -> float:
        feat = np.concatenate([state, weights])
        return float(feat @ Wc + bc)

    def select_weights(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        weights = self._actor(state, self.W_actor, self.b_actor)
        if explore:
            noise = self._rng.normal(0, self.exploration_sigma, size=self.n_actions)
            weights = self._softmax(np.log(weights + 1e-8) + noise)  # perturb in logit space, renormalize
        return weights

    # -- training --------------------------------------------------------------

    def train_step(self, state: np.ndarray, weights: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        self.buffer.add(np.asarray(state, dtype=np.float64), np.asarray(weights, dtype=np.float64),
                         float(reward), np.asarray(next_state, dtype=np.float64))
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        self._n_updates += 1

        # -- critic update (every step) --
        grad_Wc1 = np.zeros_like(self.Wc1)
        grad_bc1 = 0.0
        grad_Wc2 = np.zeros_like(self.Wc2)
        grad_bc2 = 0.0
        for s, w, r, s2 in batch:
            w2 = self._actor(s2, self.W_actor_target, self.b_actor_target)
            noise = np.clip(self._rng.normal(0, self.policy_noise, size=self.n_actions),
                             -self.noise_clip, self.noise_clip)
            w2 = self._softmax(np.log(w2 + 1e-8) + noise)  # target policy smoothing

            q1_target = self._critic(s2, w2, self.Wc1_target, self.bc1_target)
            q2_target = self._critic(s2, w2, self.Wc2_target, self.bc2_target)
            td_target = r + self.gamma * min(q1_target, q2_target)

            feat = np.concatenate([s, w])
            err1 = self._critic(s, w, self.Wc1, self.bc1) - td_target
            err2 = self._critic(s, w, self.Wc2, self.bc2) - td_target
            grad_Wc1 += err1 * feat
            grad_bc1 += err1
            grad_Wc2 += err2 * feat
            grad_bc2 += err2

        n = len(batch)
        self.Wc1 -= self.critic_lr * grad_Wc1 / n
        self.bc1 -= self.critic_lr * grad_bc1 / n
        self.Wc2 -= self.critic_lr * grad_Wc2 / n
        self.bc2 -= self.critic_lr * grad_bc2 / n

        # -- delayed actor update + target soft-update --
        if self._n_updates % self.policy_delay == 0:
            grad_W_actor = np.zeros_like(self.W_actor)
            grad_b_actor = np.zeros_like(self.b_actor)
            for s, _w, _r, _s2 in batch:
                weights_pred = self._actor(s, self.W_actor, self.b_actor)
                # Critic 1 is linear in weights, so dQ1/dweights is exactly
                # the "weights block" of Wc1 (constant w.r.t. weights).
                dQ_dw = self.Wc1[self.state_dim:]
                # Softmax Jacobian, applied to dQ/dweights (standard
                # deterministic-policy-gradient chain rule through softmax):
                # dQ/dlogits_j = weights_j * (dQ_dw_j - sum_i weights_i*dQ_dw_i)
                dQ_dlogits = weights_pred * (dQ_dw - float(np.dot(weights_pred, dQ_dw)))
                # Ascend on Q -> gradient ASCENT, so subtract the negative
                # gradient (equivalently: += , since we want to maximize Q).
                grad_W_actor += np.outer(s, dQ_dlogits)
                grad_b_actor += dQ_dlogits

            self.W_actor += self.actor_lr * grad_W_actor / n
            self.b_actor += self.actor_lr * grad_b_actor / n

            self.W_actor_target = self.tau * self.W_actor + (1 - self.tau) * self.W_actor_target
            self.b_actor_target = self.tau * self.b_actor + (1 - self.tau) * self.b_actor_target
            self.Wc1_target = self.tau * self.Wc1 + (1 - self.tau) * self.Wc1_target
            self.bc1_target = self.tau * self.bc1 + (1 - self.tau) * self.bc1_target
            self.Wc2_target = self.tau * self.Wc2 + (1 - self.tau) * self.Wc2_target
            self.bc2_target = self.tau * self.bc2 + (1 - self.tau) * self.bc2_target
