import os

import numpy as np
import tensorflow as tf
from tensorflow import keras


class RLAttackClassifier:
    """Classifies the attack type from the 4 detection modality scores.

    Design note: despite the name (kept for consistency with the papers'
    narrative in references/), this component is trained as a supervised
    classifier — the ground-truth attack label is known during training
    (comes from AttackSimulator), so there is no real "environment" to
    explore via DQN. `src.defense.rl_selector.RLDefenseSelector` is the
    component that trains as genuine RL, since it has a real environment
    (FL rounds) with an observable reward.
    """

    def __init__(self, n_attacks=11):
        self.n_attacks = n_attacks
        self.model = self._build_network()
        self.optimizer = keras.optimizers.Adam(0.001)

    def _build_network(self):
        """Network: input 4 modalities, output softmax over n_attacks attack types."""
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(4,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(self.n_attacks, activation='softmax')
        ])
        return model

    def classify(self, modalidades):
        """Classifies the attack type from a single 4-modality score vector."""
        prediction = self.model(np.array([modalidades], dtype=np.float32))
        probabilities = prediction[0].numpy()
        attack_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[attack_idx])
        return attack_idx, confidence

    def train_on_batch(self, X_batch, y_batch):
        """One training step (cross-entropy) over a real labeled batch."""
        X_batch = np.asarray(X_batch, dtype=np.float32)
        y_batch = np.asarray(y_batch, dtype=np.int64)

        with tf.GradientTape() as tape:
            predictions = self.model(X_batch)
            loss = keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        return float(loss.numpy())

    def train(self, X, y, epochs=20, batch_size=32, verbose=True):
        """Trains over several epochs on a full labeled dataset.

        X: (N, 4) modality scores. y: (N,) integer attack-type indices.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n = len(X)

        loss_history = []
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                epoch_loss += self.train_on_batch(X[idx], y[idx])
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

        return loss_history

    def save(self, path="results/models/rl_classifier.weights.h5"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save_weights(path)

    def load(self, path="results/models/rl_classifier.weights.h5", n_modalities=4):
        self.model(np.zeros((1, n_modalities), dtype=np.float32))
        self.model.load_weights(path)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n_attacks = 4
    X_train = rng.standard_normal((400, 4)).astype(np.float32)
    y_train = (X_train[:, 0] > 0).astype(np.int64) + 2 * (X_train[:, 1] > 0).astype(np.int64)

    classifier = RLAttackClassifier(n_attacks=n_attacks)
    classifier.train(X_train, y_train, epochs=20)

    attack_idx, confidence = classifier.classify([0.5, 0.8, 0.2, 0.9])
    print(f"\nPrediction: Attack type {attack_idx} (confidence: {confidence:.2%})")

    classifier.save()
    print("Model saved to results/models/rl_classifier.weights.h5")
