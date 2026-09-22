"""
AdaAggRLAgent: reproduction of AdaAggRL ("Defending Against Sophisticated
Poisoning Attacks with RL-based Aggregation in Federated Learning", Wang,
Zhang, Wen, Qiu & Guo, AAAI 2025 — see
`references/citations/Defending Against Sophisticated Poisoning Attacks
with RL-based Aggregation in Federated Learning.pdf`), used as a competing
baseline (the "continuous weighting" family) in
`src/experiments/exp10_selector_comparison.py`, per
`references/experimento_seletores_adaptativos.md`.

CORRECTION NOTE: an earlier version of this module was a from-scratch
instantiation of the plan doc's one-line family description ("RL/TD3 que
aprende pesos contínuos de agregação a partir de ~4 métricas"), written
before this paper had been identified as AdaAggRL's actual source. The
paper's PDF's Supplementary Information links a public repo
(`https://github.com/yjEugenia/AdaAggRL`), but — same treatment as
FedStrategist's repo (`src/defense/fedstrategist_selector.py`) — it was not
fetched/ported; this module is a from-spec reproduction of the paper's own
Algorithm 1/2 and equations, built independently in this codebase's
architecture.

The real algorithm (very different from the earlier guess) has THREE parts:

1. Distribution learning via GRADIENT INVERSION (paper's "Distribution
   Learning" section, adapting Geiping et al. 2020's IG method): from each
   client's uploaded parameter delta, the server recovers a target gradient
   g_bar_k = update_k / local_lr, then optimizes a small batch of DUMMY
   inputs (initialized to zero, Adam, lr=0.05, `max_iters` steps) so that
   the gradient the dummy batch would produce on this client's model
   matches g_bar_k in cosine similarity (plus a small total-variation
   smoothness penalty, weight `tv_weight=1e-4`, matching the paper's beta).
   The achieved cosine similarity IS `S_{k,R}` (`reconstruction similarity`,
   one of the 4 environmental cues below). Implemented here as a NESTED
   `tf.GradientTape` (the standard "double backprop through a gradient"
   pattern for gradient-inversion/DLG-style attacks): an inner tape
   computes the dummy batch's loss gradient w.r.t. the (frozen, constant)
   model weights, and an outer tape differentiates the cosine-similarity
   objective w.r.t. the dummy inputs through that inner gradient. Verified
   to converge (cosine similarity 0.20 -> 0.99 over 30 steps on a synthetic
   check) before being wired into the rest of this module.

2. Environmental cues (paper's "Environmental Cues" section, Eq. 1): the
   `num_images` reconstructed dummy images are passed through a feature
   extractor (`RandomCNNFeatureExtractor` below) to get this round's
   feature vector V_k^current; a per-client rolling V_k^history (last
   round's V_current) and this round's V_g (mean of V_current across all
   participating clients) are then compared via a Gaussian/RBF-kernel MMD
   estimator (`mmd_rbf`) and squashed into similarities in (0,1) by
   `2*cos(tanh(mmd/2)) - 1` (paper's Eq. 1, `cue_similarity` below,
   implemented literally even though the composition cos(tanh(.)) is an
   unusual squashing choice — not ours to second-guess). Per-client state:
   s_k = (S_{k,R}, S_{k,cl}, S_{k,cg}, S_{k,lg}) in (0,1)^4 — the paper's
   own "~4 metrics", now the REAL 4, replacing the earlier guess (variance
   of update norms / avg cosine / mean norm / outlier fraction).

2b. DOCUMENTED SUBSTITUTION (superseded by 2c below, kept for history): the
   paper's feature extractor is a "pre-trained CNN" (unspecified
   architecture/checkpoint). No suitable pretrained checkpoint for
   MNIST/CIFAR-shaped inputs exists in this repo, and downloading one was
   out of scope initially — `RandomCNNFeatureExtractor` uses a FROZEN,
   RANDOMLY-INITIALIZED small CNN instead (random-projection features are a
   known, if weaker, substitute for a pretrained encoder; still a
   deterministic, fixed, non-trivial feature map). Flagged explicitly as a
   real deviation from the paper, not silently substituted. Every number
   produced with this extractor is a documented FLOOR, not the AdaAggRL
   paper's real performance (see `references/1_roadmap_frentes_futuras.md`,
   "Passo Zero").

2c. PASSO ZERO (`references/1_roadmap_frentes_futuras.md`): `RandomCNNFeatureExtractor`
   is now complemented by `PretrainedCNNFeatureExtractor` below, a REAL
   trained-then-frozen CNN, closing the gap flagged in 2b. It reuses the
   exact architecture already used for CNN-scale FL in this repo
   (`src/fl/federated_learner.py::_CNNModel`: Conv2D(8)->MaxPool->Conv2D(16)
   ->MaxPool->Flatten->Dense(32,relu)->Dense(n_classes,softmax)), trained
   centrally as an ordinary image classifier on `data/raw/{dataset}/X_test.npy`
   /`y_test.npy` — the RAW MNIST/CIFAR-10 TEST split, which is disjoint from
   both the FL clients' data and the `server_val` root (both carved from the
   TRAIN pool by `data/download_datasets.py`), so pretraining introduces no
   leakage into anything `exp10_selector_comparison.py` measures. The frozen
   Dense(32, relu) activations (not the final softmax layer) are the feature
   vector, feature_dim=32 — a different, larger dimensionality than
   `RandomCNNFeatureExtractor`'s default 16; the two are independent
   configurations and were never required to match. Trained once via
   `scripts/pretrain_adaaggrl_extractor.py`, saved to
   `results/models/adaaggrl_pretrained_extractor_{dataset}.weights.h5`, and
   loaded (never retrained) by `PretrainedCNNFeatureExtractor`.
   `RandomCNNFeatureExtractor` is kept, unmodified, so the original floor
   result stays reproducible for the piso-vs-real comparison — callers pick
   one via `AdaAggRLGridLearner(feature_extractor=...)`.

3. Actions Learning (paper's "Actions Learning" section + Algorithm 1/2):
   a TD3 policy maps the round's environmental state to an action
   A^t=(a^t,b^t) in [0,1]^5 — a^t in [0,1]^4 weights the 4 cues, b^t in
   [0,1] is a threshold fraction. Per client: ŵ_k = s_k · a^t (weighted
   score); w̃ = g(ŵ) normalizes scores to [0,1] (min-max here — the paper
   only says "maps ŵ to [0,1] and normalizes it" without fixing g's exact
   form, our documented instantiation); δ = max(w̃)·b^t; clients with
   w̃_k <= δ get zero weight (`f_δ`, Eq. 2) AND have their persistent
   malicious-behavior counter h_k incremented (else decremented toward 0);
   final aggregation weight is w_k / lambda^{h_k^{t+1}} (Eq. 3) — repeatedly
   flagged clients are penalized exponentially harder each additional
   consecutive round they're flagged. Reward: r = f(theta^t) - f(theta^{t+1})
   (paper's own definition — the round's LOSS decrease; implemented here as
   `old_loss - new_loss` on the caller's held-out evaluation set).

   TD3 mechanics: same lightweight, linear-function-approximator
   instantiation already used in this codebase's prior AdaAggRL attempt and
   documented there as a deliberate simplification (mirrors
   `TARSSelector`'s tabular Q-learning instead of a DQN, for the same
   reason — tiny state/training-signal budget per FL round). Still
   implements TD3's three defining mechanisms (twin critics taking the min,
   delayed policy updates, clipped target-policy-smoothing noise); only the
   function class (linear vs. MLP) and the action's squashing (sigmoid,
   since the paper's action space is the BOX [0,1]^5, not a probability
   simplex — an actual, not just cosmetic, difference from the earlier
   version's softmax-over-7-strategies action) changed from the prior draft.

3b. State aggregation across a variable number of clients: the network
   needs a FIXED-size input, but |C^t| (participating clients this round)
   varies. The paper's Algorithm 1 writes `A^t = Actions(s^t)` with s^t
   stacking every participating client's 4-vector, without specifying how a
   variable-length stack becomes one policy input (an implementation detail
   presumably resolved in the paper's own repo, not fetched here — see the
   correction note above). This module uses the MEAN of the per-client
   4-vectors across the round as a permutation-invariant, fixed-size
   summary fed to the actor — our own documented instantiation, exactly the
   same kind of unavoidable interpretation Eq. 2/3's `g(·)` above already
   required.

Aggregation (Eq. 3) operates on clients' FULL PARAMETER VECTORS
(theta_k^{t+1} = global_params + update_k), not on deltas blended from
other aggregation rules like the earlier version did — a substantive
mechanism difference, not just a relabeling.

NORMALIZATION NOTE on Eq. 3: taken completely literally, `sum_k w_k /
lambda^{h_k}` is not guaranteed to sum to 1 (some clients are zeroed by the
threshold, and the rest are further shrunk by `lambda^{h_k} >= 1`), which
would make the aggregated theta's overall scale drift across rounds when
applied to CLIENT PARAMETERS instead of deltas. This is either a genuine
gap in the paper's equation or an implementation detail resolved in its own
repo. `compute_weights_and_penalty` below returns the coefficients exactly
as Eq. 3 defines them (unnormalized); the caller
(`src/experiments/exp10_selector_comparison.py::AdaAggRLGridLearner`)
renormalizes them to sum to 1 before the weighted combination, as a
documented stabilizing choice — kept as a caller-side decision rather than
baked into this function, so Eq. 3's literal output stays inspectable.

V_k^current / V_k^history / V_g (paper's "Environmental Cues" section):
kept as the FULL set of `num_images` reconstructed-image feature vectors
per client/round (matching the paper's own phrase "a collection of feature
vectors"), not collapsed to a single mean vector — this keeps MMD a
genuine distribution-vs-distribution comparison throughout. V_g ("obtained
by averaging feature vectors from all participating clients") is
instantiated here as the POOLED union of every participating client's
V_current set this round, our documented reading of "averaging... from all
participating clients" that keeps every one of the three MMD calls
(current-vs-history, current-vs-global, history-vs-global) a consistent
set-vs-set comparison instead of mixing set-vs-point cases.
"""

import os
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


class PretrainedCNNFeatureExtractor:
    """REAL trained-then-frozen CNN feature extractor (see module docstring,
    point 2c / Passo Zero) — replaces `RandomCNNFeatureExtractor`'s random
    projection with actual learned image features, closing the documented
    gap between this reproduction and the AdaAggRL paper's "pre-trained
    CNN". Weights must already exist on disk (produced once by
    `scripts/pretrain_adaaggrl_extractor.py`); this class only loads and
    freezes them, it never trains."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        n_classes: int = 10,
        feature_dim: int = 32,
        weights_path: Optional[str] = None,
        dataset: str = "mnist",
    ):
        from tensorflow import keras

        self.input_shape = input_shape
        self.feature_dim = feature_dim
        if weights_path is None:
            weights_path = f"results/models/adaaggrl_pretrained_extractor_{dataset}.weights.h5"
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Pretrained AdaAggRL feature extractor weights not found at '{weights_path}'. "
                "Run `python -m scripts.pretrain_adaaggrl_extractor "
                f"--dataset {dataset}` first (Passo Zero, "
                "references/1_roadmap_frentes_futuras.md)."
            )

        classifier = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(8, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(feature_dim, activation="relu", name="features"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ])
        classifier.load_weights(weights_path)
        for layer in classifier.layers:
            layer.trainable = False
        self._model = keras.Model(inputs=classifier.inputs, outputs=classifier.get_layer("features").output)

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

    `W`, `b`: this client's post-update weights (theta_k^{t+1}), packed the
    same way `_LogisticModel` does ((n_features, n_classes) / (n_classes,)).
    Assumes n_classes >= 3 (softmax) — every dataset used in this
    repo's experiments (MNIST/CIFAR-10) has n_classes=10.
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
    w_hat = state_matrix @ a  # (n_clients,)

    lo, hi = w_hat.min(), w_hat.max()
    w_tilde = (w_hat - lo) / (hi - lo + 1e-8)  # g(.): min-max normalization to [0,1] (our instantiation)

    delta = float(w_tilde.max() * b)
    flagged = w_tilde <= delta
    w_filtered = np.where(flagged, 0.0, w_tilde)  # f_delta, Eq. 2

    new_h = np.where(flagged, prev_h + 1, np.maximum(prev_h - 1, 0))

    final_weights = w_filtered / (lam ** new_h)
    return final_weights, new_h, delta
