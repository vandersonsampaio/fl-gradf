"""
AdaAggRLAgent: reproduction of AdaAggRL, proposed in:

    Wang, Zhang, Wen, Qiu & Guo,
    "Defending Against Sophisticated Poisoning Attacks with RL-based
    Aggregation in Federated Learning", AAAI 2025.

Used as a competing baseline representing the continuous-weighting family
in ``src/experiments/exp10_selector_comparison.py``.

This implementation follows the paper's Algorithm 1/2 and equations and
is implemented independently in this codebase.

Main components:

1. Distribution learning
   Recovers an approximate client gradient from the uploaded parameter
   update and reconstructs dummy inputs through gradient inversion.
   The reconstruction similarity is used as the cue ``S_{k,R}``.

2. Environmental cues
   Extracts features from reconstructed images and computes distribution
   similarities using RBF-kernel MMD. Each client is represented by four
   cues:

       (S_{k,R}, S_{k,cl}, S_{k,cg}, S_{k,lg})

3. Actions learning
   A TD3 policy produces:

       A^t = (a^t, b^t)

   where ``a^t`` contains four cue weights and ``b^t`` is the threshold
   fraction. Client scores are computed from the weighted cues, normalized,
   thresholded, and combined with the persistent malicious-behavior
   counter ``h_k`` to obtain the final aggregation weights.

4. Reward
   The reward follows the paper's definition:

       r = f(theta^t) - f(theta^{t+1})

   and is implemented as the decrease in held-out evaluation loss.

Implementation choices and documented deviations:

- ``RandomCNNFeatureExtractor`` is used instead of the paper's unspecified
  pretrained CNN. It is frozen and randomly initialized, providing a fixed
  feature mapping without requiring an external checkpoint.

- The TD3 implementation uses lightweight linear function approximators
  rather than MLPs. It retains the defining TD3 mechanisms: twin critics,
  delayed policy updates, and clipped target-policy-smoothing noise.

- Because the number of participating clients varies by round, the policy
  receives the mean of the clients' four-dimensional cue vectors as a
  fixed-size, permutation-invariant state representation.

- Client aggregation operates on full client parameter vectors:

      theta_k^{t+1} = global_params + update_k

  rather than directly blending client deltas from other aggregation rules.

- ``compute_weights_and_penalty`` returns the coefficients defined by
  Equation 3 without normalization. The caller normalizes them before
  aggregation to prevent scale drift when client parameter vectors are
  combined.

- ``V_k^current`` and ``V_k^history`` retain all reconstructed-image feature
  vectors. ``V_g`` is represented as the pooled set of feature vectors from
  all participating clients, keeping the MMD computations as
  distribution-to-distribution comparisons.

The implementation therefore reproduces the paper's continuous-weighting
mechanism while making the necessary architectural choices explicit where
the paper does not fully specify an implementation detail.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# MMD-based environmental cues (paper's Eq. 1)
# ---------------------------------------------------------------------------

def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
    """Squared MMD between feature sets X, Y under an RBF kernel (Gretton et
    al., the standard estimator cited by the paper via Arbel et al. 2019 /
    Wang et al. 2021). `sigma`: kernel bandwidth; defaults to the median
    pairwise distance across X and Y (the common median heuristic) when not
    given."""
    XY = np.concatenate([X, Y], axis=0)
    if sigma is None:
        d = np.linalg.norm(XY[:, None, :] - XY[None, :, :], axis=-1)
        nonzero = d[d > 0]
        sigma = float(np.median(nonzero)) if nonzero.size else 1.0
        sigma = max(sigma, 1e-6)

    def _kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-d2 / (2.0 * sigma ** 2))

    Kxx = _kernel(X, X)
    Kyy = _kernel(Y, Y)
    Kxy = _kernel(X, Y)
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def cue_similarity(mmd_value: float) -> float:
    """Paper's Eq. 1: S = 2*cos(tanh(MMD/2)) - 1, squashing MMD in
    [0, +inf) into a similarity in (roughly) (-1, 1]. Implemented literally."""
    return float(2.0 * np.cos(np.tanh(mmd_value / 2.0)) - 1.0)


# ---------------------------------------------------------------------------
# Feature extractor (documented substitute for the paper's pre-trained CNN)
# ---------------------------------------------------------------------------

class RandomCNNFeatureExtractor:
    """Frozen, randomly-initialized small CNN used in place of the paper's
    unspecified PRE-TRAINED CNN (see module docstring, point 2b) — a real,
    documented deviation, not a silent substitution."""

    def __init__(self, input_shape: Tuple[int, int, int], feature_dim: int = 16, seed: int = 0):
        from tensorflow import keras

        keras.utils.set_random_seed(seed)
        self.input_shape = input_shape
        self._model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(8, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(feature_dim, activation="relu"),
        ])
        for layer in self._model.layers:
            layer.trainable = False

    def extract(self, images_flat: np.ndarray) -> np.ndarray:
        """`images_flat`: (n_images, prod(input_shape)) -> (n_images, feature_dim)."""
        imgs = images_flat.reshape((-1,) + self.input_shape).astype("float32")
        return self._model.predict(imgs, verbose=0)


# ---------------------------------------------------------------------------
# Gradient inversion (paper's "Distribution Learning" section)
# ---------------------------------------------------------------------------

def reconstruct_client_distribution(
    W: np.ndarray,
    b: np.ndarray,
    update: np.ndarray,
    local_lr: float,
    n_features: int,
    n_classes: int,
    num_images: int = 16,
    max_iters: int = 30,
    lr: float = 0.05,
    tv_weight: float = 1e-4,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """Reconstructs `num_images` dummy samples whose gradient (on the
    client's own model weights W,b) aligns with the client's uploaded
    update, via `max_iters` steps of Adam over the dummy inputs (nested
    `tf.GradientTape`, see module docstring). Returns
    (D_rec (num_images, n_features), S_R (final cosine similarity)).
    """
    import tensorflow as tf

    rng = np.random.RandomState(seed)
    target_grad = (update.astype(np.float64) / local_lr).astype(np.float32)

    W_c = tf.constant(W, dtype=tf.float32)
    b_c = tf.constant(b, dtype=tf.float32)
    target = tf.constant(target_grad, dtype=tf.float32)

    y_idx = rng.randint(0, n_classes, size=num_images)
    y_onehot = tf.one_hot(y_idx, n_classes)

    X_dummy = tf.Variable(tf.zeros((num_images, n_features), dtype=tf.float32))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    cos_sim_val = 0.0
    for _ in range(max_iters):
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                inner_tape.watch([W_c, b_c])
                logits = tf.matmul(X_dummy, W_c) + b_c
                ce = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits)
                )
            gW, gb = inner_tape.gradient(ce, [W_c, b_c])
            dummy_grad = tf.concat([tf.reshape(gW, [-1]), tf.reshape(gb, [-1])], axis=0)
            num = tf.reduce_sum(dummy_grad * target)
            denom = tf.norm(dummy_grad) * tf.norm(target) + 1e-8
            cos_sim = num / denom
            tv = tf.reduce_mean(tf.square(X_dummy[:, 1:] - X_dummy[:, :-1]))
            loss = (1.0 - cos_sim) + tv_weight * tv
        grads = outer_tape.gradient(loss, [X_dummy])
        optimizer.apply_gradients(zip(grads, [X_dummy]))
        cos_sim_val = float(cos_sim.numpy())

    return X_dummy.numpy(), cos_sim_val


# ---------------------------------------------------------------------------
# TD3 policy over the [0,1]^5 action space (paper's "Actions Learning")
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    def __init__(self, capacity: int = 500, seed: int = 0):
        self.capacity = capacity
        self._buf: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
        self._rng = np.random.RandomState(seed)

    def add(self, s: np.ndarray, a: np.ndarray, r: float, s2: np.ndarray) -> None:
        self._buf.append((s, a, r, s2))
        if len(self._buf) > self.capacity:
            self._buf.pop(0)

    def sample(self, batch_size: int):
        n = min(batch_size, len(self._buf))
        idx = self._rng.choice(len(self._buf), size=n, replace=False)
        return [self._buf[i] for i in idx]

    def __len__(self) -> int:
        return len(self._buf)


class AdaAggRLAgent:
    """TD3 policy over A=(a,b) in [0,1]^5: `a` (4-dim) weights the 4
    environmental cues, `b` (1-dim, action index 4) is the threshold
    fraction (paper's Eq. 2). See module docstring for the linear-function-
    approximator simplification and the mean-pooling state-aggregation
    choice."""

    ACTION_DIM = 5  # 4 cue weights + 1 threshold

    def __init__(
        self,
        state_dim: int = 4,
        actor_lr: float = 0.05,
        critic_lr: float = 0.05,
        gamma: float = 0.99,  # paper's own value (Details of Experimental Settings)
        tau: float = 0.1,
        policy_noise: float = 0.05,
        noise_clip: float = 0.1,
        policy_delay: int = 2,
        exploration_sigma: float = 0.15,
        batch_size: int = 8,
        buffer_capacity: int = 500,
        seed: int = 0,
    ) -> None:
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
        feat_dim = state_dim + self.ACTION_DIM

        self.W_actor = self._rng.normal(0, 0.1, size=(state_dim, self.ACTION_DIM))
        self.b_actor = np.zeros(self.ACTION_DIM)
        self.W_actor_target = self.W_actor.copy()
        self.b_actor_target = self.b_actor.copy()

        self.Wc1 = self._rng.normal(0, 0.1, size=feat_dim)
        self.bc1 = 0.0
        self.Wc2 = self._rng.normal(0, 0.1, size=feat_dim)
        self.bc2 = 0.0
        self.Wc1_target, self.bc1_target = self.Wc1.copy(), self.bc1
        self.Wc2_target, self.bc2_target = self.Wc2.copy(), self.bc2

        self.buffer = _ReplayBuffer(capacity=buffer_capacity, seed=seed)
        self._n_updates = 0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _actor(self, state: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._sigmoid(state @ W + b)

    def _critic(self, state: np.ndarray, action: np.ndarray, Wc: np.ndarray, bc: float) -> float:
        feat = np.concatenate([state, action])
        return float(feat @ Wc + bc)

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        action = self._actor(state, self.W_actor, self.b_actor)
        if explore:
            noise = self._rng.normal(0, self.exploration_sigma, size=self.ACTION_DIM)
            action = np.clip(action + noise, 0.0, 1.0)
        return action

    def train_step(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        self.buffer.add(np.asarray(state, dtype=np.float64), np.asarray(action, dtype=np.float64),
                         float(reward), np.asarray(next_state, dtype=np.float64))
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        self._n_updates += 1

        grad_Wc1 = np.zeros_like(self.Wc1); grad_bc1 = 0.0
        grad_Wc2 = np.zeros_like(self.Wc2); grad_bc2 = 0.0
        for s, a, r, s2 in batch:
            a2 = self._actor(s2, self.W_actor_target, self.b_actor_target)
            noise = np.clip(self._rng.normal(0, self.policy_noise, size=self.ACTION_DIM),
                             -self.noise_clip, self.noise_clip)
            a2 = np.clip(a2 + noise, 0.0, 1.0)  # target policy smoothing

            q1_target = self._critic(s2, a2, self.Wc1_target, self.bc1_target)
            q2_target = self._critic(s2, a2, self.Wc2_target, self.bc2_target)
            td_target = r + self.gamma * min(q1_target, q2_target)

            feat = np.concatenate([s, a])
            err1 = self._critic(s, a, self.Wc1, self.bc1) - td_target
            err2 = self._critic(s, a, self.Wc2, self.bc2) - td_target
            grad_Wc1 += err1 * feat; grad_bc1 += err1
            grad_Wc2 += err2 * feat; grad_bc2 += err2

        n = len(batch)
        self.Wc1 -= self.critic_lr * grad_Wc1 / n
        self.bc1 -= self.critic_lr * grad_bc1 / n
        self.Wc2 -= self.critic_lr * grad_Wc2 / n
        self.bc2 -= self.critic_lr * grad_bc2 / n

        if self._n_updates % self.policy_delay == 0:
            grad_W_actor = np.zeros_like(self.W_actor)
            grad_b_actor = np.zeros_like(self.b_actor)
            for s, _a, _r, _s2 in batch:
                a_pred = self._actor(s, self.W_actor, self.b_actor)
                dQ_da = self.Wc1[self.state_dim:]                    # critic 1 is linear in action
                dsig = a_pred * (1.0 - a_pred)                       # sigmoid derivative
                dQ_dlogits = dQ_da * dsig
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


# ---------------------------------------------------------------------------
# Weighting mechanism (paper's Eq. 2/3)
# ---------------------------------------------------------------------------

def compute_weights_and_penalty(
    state_matrix: np.ndarray,   # (n_clients, 4): per-client (S_R, S_cl, S_cg, S_lg)
    action: np.ndarray,         # (5,): [a (4,), b (scalar)]
    prev_h: np.ndarray,         # (n_clients,) malicious-behavior counters
    lam: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Paper's Eq. 2 (client filtering via an adaptive threshold) + the
    exponential-penalty counter h (unnumbered equation right after Eq. 2) +
    the per-client coefficient used in Eq. 3 (w_k / lambda^{h_k}).
    Returns (final_weights, new_h, delta)."""
    a, b = action[:4], action[4]
    w_hat = state_matrix @ a

    lo, hi = w_hat.min(), w_hat.max()
    w_tilde = (w_hat - lo) / (hi - lo + 1e-8)

    delta = float(w_tilde.max() * b)
    flagged = w_tilde <= delta
    w_filtered = np.where(flagged, 0.0, w_tilde)  # f_delta, Eq. 2

    new_h = np.where(flagged, prev_h + 1, np.maximum(prev_h - 1, 0))

    final_weights = w_filtered / (lam ** new_h)
    return final_weights, new_h, delta
