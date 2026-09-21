import os
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


class RLDefenseSelector:
    """DQN that chooses the aggregation strategy best suited to the detected
    attack type. Unlike `RLAttackClassifier`, this component trains as
    genuine RL: there is a real environment (FL rounds) and an observable
    reward (accuracy retention, attack neutralization) — see
    `src.utils.data_loader.generate_selector_experiences`.

    Design notes (see `references/plano_gradf_iclr2027.md` and
    `FORMALISMO_MATEMATICO_E_INEDITISMO.md` for the underlying formalism):

    1. **5-dimensional state**, not 2: `[attack_idx, confidence,
       prev_accuracy, prev_fairness_std, prev_latency]` — the accuracy/
       fairness/latency of the previous round are part of the state so the
       network can differentiate rounds with the same detected attack but
       very different contexts (accuracy dropping, poor fairness, high
       latency).
    2. **Real online learning.** `train_step` is called both during offline
       pretraining (`train_on_experiences`) and on EVERY REAL FL ROUND by
       `HardeningPipeline.full_pipeline` (see its docstring) — the policy
       keeps updating during deployment, not just once before round 1.
    3. **Target network**, synchronized every `target_update_every` training
       steps, used for the Bellman bootstrap (`max_a' Q_target(s',a')`) —
       avoids the known instability of using the SAME network being updated
       as its own moving target.
    4. **Replay buffer with mini-batch updates.** Every experience (offline
       or online) enters the same buffer; each `train_step` samples a
       mini-batch from it for the gradient step, instead of training on a
       single sample at a time — reduces gradient variance and allows old
       experiences to be reused.
    5. **Decaying ε-greedy exploration during online learning**:
       `select_action` explores with probability `ε` (decaying from
       `epsilon_start` to `epsilon_end` over `epsilon_decay_rounds`) when
       `round_num` is supplied (as `GRADFFederatedLearner` does via
       `round_context`) — the same mechanism used by TARS
       (`src/defense/tars_selector.py`), so the Gate 1 comparison stays fair
       (neither selector gets more exploration than the other). Without
       `round_num` (callers that don't track round, e.g. standalone
       pretraining episodes), it stays fully greedy.
    """

    N_STATE_DIMS = 5

    def __init__(
        self,
        target_update_every: int = 20,
        replay_buffer_size: int = 2000,
        batch_size: int = 16,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.02,
        epsilon_decay_rounds: int = 20,
        seed: int = 0,
    ):
        self.actions = [
            'fedavg', 'fedprox', 'median', 'trimmed_mean',
            'fltrust', 'clustering', 'krum'
        ]
        self.n_actions = len(self.actions)
        self.model = self._build_policy_network()
        self.target_model = self._build_policy_network()
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = keras.optimizers.Adam(0.001)
        self.discount = 0.99

        self.target_update_every = target_update_every
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rounds = max(epsilon_decay_rounds, 1)
        self.replay_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray]] = deque(maxlen=replay_buffer_size)
        self._train_steps = 0
        self._rng = np.random.default_rng(seed)

    def _build_policy_network(self):
        """DQN: input 5-D state, output Q-values for 7 actions."""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.N_STATE_DIMS,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.n_actions)  # Q-value for each action
        ])
        return model

    @classmethod
    def build_state(
        cls,
        attack_type: float,
        confidence: float,
        prev_accuracy: float = 0.0,
        prev_fairness_std: float = 0.0,
        prev_latency: float = 0.0,
    ) -> np.ndarray:
        """Builds the 5-dimensional state vector — the single point that
        defines the state's order/composition, used both here and in
        `src.utils.data_loader.generate_selector_experiences` (pretraining)
        and `src.fl.gradf_learner.GRADFFederatedLearner` (online learning),
        so the three sources can never silently diverge."""
        return np.array(
            [attack_type, confidence, prev_accuracy, prev_fairness_std, prev_latency],
            dtype=np.float32,
        )

    def _epsilon(self, round_num: int) -> float:
        """Linear decay from `epsilon_start` to `epsilon_end` over
        `epsilon_decay_rounds` — same schedule as `TARSSelector._epsilon`
        (`src/defense/tars_selector.py`), so the Gate 1 comparison doesn't
        favor either one with more exploration than the other."""
        frac = min(round_num / self.epsilon_decay_rounds, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(
        self,
        attack_type,
        confidence,
        prev_accuracy: float = 0.0,
        prev_fairness_std: float = 0.0,
        prev_latency: float = 0.0,
        round_num: Optional[int] = None,
    ):
        """Selects the best defense for this attack type, in the context of
        the previous round's state (`prev_*`, all optional/0.0 by default —
        cold start on round 1, or callers that don't track round context,
        e.g. `exp1_baseline.py`'s standalone pretraining episodes).

        `attack_type` must be a numeric index (same encoding used during
        training, e.g. an index into `src.detection.modality_recorder.ATTACK_LABELS`).

        `round_num`: if supplied, enables decaying ε-greedy exploration (see
        `_epsilon`) — needed for real ONLINE learning to have a chance of
        discovering an action better than the one offline pretraining
        favored (without it, the policy would only reinforce its own initial
        choice, never questioning it). If `None` (default), stays fully
        greedy — used by callers that don't track round number."""
        state = self.build_state(attack_type, confidence, prev_accuracy, prev_fairness_std, prev_latency)
        q_values = self.model(state.reshape(1, -1))[0].numpy()

        if round_num is not None and self._rng.random() < self._epsilon(round_num):
            best_action_idx = int(self._rng.integers(self.n_actions))
        else:
            best_action_idx = int(np.argmax(q_values))

        best_action = self.actions[best_action_idx]
        q_value = float(q_values[best_action_idx])

        return best_action, q_value

    def _sync_target_network(self) -> None:
        self.target_model.set_weights(self.model.get_weights())

    def _sample_batch(self) -> List[Tuple[np.ndarray, int, float, np.ndarray]]:
        if len(self.replay_buffer) <= self.batch_size:
            return list(self.replay_buffer)
        idx = self._rng.choice(len(self.replay_buffer), size=self.batch_size, replace=False)
        return [self.replay_buffer[i] for i in idx]

    def train_step(self, state, action_idx, reward, next_state) -> float:
        """Records the experience in the replay buffer and takes a mini-batch
        gradient step (sampled from the buffer — not just the received
        experience), using the target network for the Bellman bootstrap.
        Called both by offline pretraining (`train_on_experiences`) and by
        online learning (`HardeningPipeline.full_pipeline`, on every real FL
        round)."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
        self.replay_buffer.append((state, int(action_idx), float(reward), next_state))

        batch = self._sample_batch()
        states = tf.convert_to_tensor(np.stack([e[0] for e in batch]), dtype=tf.float32)
        action_idxs = tf.convert_to_tensor([e[1] for e in batch], dtype=tf.int32)
        rewards = tf.convert_to_tensor([e[2] for e in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.stack([e[3] for e in batch]), dtype=tf.float32)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_selected = tf.gather(q_values, action_idxs, batch_dims=1)

            next_q_values = self.target_model(next_states)
            max_next_q = tf.reduce_max(next_q_values, axis=1)

            targets = rewards + self.discount * max_next_q
            loss = tf.reduce_mean(tf.square(q_selected - targets))

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        self._train_steps += 1
        if self._train_steps % self.target_update_every == 0:
            self._sync_target_network()

        return float(loss.numpy())

    def train_on_experiences(self, experiences, epochs: int = 1, shuffle: bool = True, seed: int = 0, verbose=True):
        """Trains over a list of (state, action_idx, reward, next_state),
        repeating `epochs` times over the buffer (reshuffled each epoch if
        `shuffle=True`). Every `train_step` call feeds the SAME replay buffer
        used by online learning — pretraining and production learning are
        not two disconnected mechanisms.

        `epochs=1` (the default, kept so callers that don't pass `epochs`
        keep their existing behavior — including
        `tests/test_defense.py::test_selector_trains_on_real_experiences`,
        which assumes `len(losses) == len(experiences)`) means a SINGLE
        training step per experience, which is not enough to reliably
        converge with the small experience buffers used in practice — see
        CLAUDE.md. Production callers (`exp1_baseline.py`, `exp4_adaptive.py`)
        pass `epochs=200` explicitly."""
        rng = np.random.default_rng(seed)
        order = list(range(len(experiences)))
        loss_history = []
        for _ in range(epochs):
            if shuffle:
                rng.shuffle(order)
            for i in order:
                state, action_idx, reward, next_state = experiences[i]
                loss_history.append(self.train_step(state, action_idx, reward, next_state))
        if verbose and loss_history:
            print(f"Training complete: {len(experiences)} experiences x {epochs} epochs, "
                  f"mean loss={np.mean(loss_history):.6f}")
        return loss_history

    def save(self, path="results/models/rl_selector.weights.h5"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save_weights(path)

    def load(self, path="results/models/rl_selector.weights.h5"):
        self.model(np.zeros((1, self.N_STATE_DIMS), dtype=np.float32))
        self.model.load_weights(path)
        self._sync_target_network()


if __name__ == '__main__':
    # Smoke test with synthetic experiences (the real training, with actual FL
    # rounds, lives in src.utils.data_loader.generate_selector_experiences and
    # is exercised in src/defense/tests/test_defense.py).
    rng = np.random.default_rng(0)
    selector = RLDefenseSelector()

    for episode in range(100):
        state = selector.build_state(
            rng.integers(0, 11), rng.random(), rng.random(), rng.random(), rng.random(),
        )
        action_idx = rng.integers(0, len(selector.actions))
        reward = float(rng.random())
        next_state = selector.build_state(
            rng.integers(0, 11), rng.random(), rng.random(), rng.random(), rng.random(),
        )

        loss = selector.train_step(state, action_idx, reward, next_state)

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}: Loss = {loss:.6f}")

    selector.save()
    print("Model saved to results/models/rl_selector.weights.h5")
