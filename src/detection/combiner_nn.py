import os

import numpy as np
import tensorflow as tf
from tensorflow import keras


class AnomalyCombinerNN(keras.Model):
    """Combines the 4 detection modalities' scores into an anomaly probability.

    See FORMALISMO_MATEMATICO_E_INEDITISMO.md, Definition 5:
    f_combine(z) = sigmoid(W_out . ReLU(W_2 . ReLU(W_1 . z))), z = [Z_1, Z_2, Z_3, Z_4].
    """

    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dense2 = keras.layers.Dense(32, activation='relu')
        self.output_layer = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # inputs: [batch_size, 4] (4 modalities)
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


class DetectionCombiner:
    """Train/inference/persistence wrapper around `AnomalyCombinerNN`."""

    def __init__(self):
        self.model = AnomalyCombinerNN()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = keras.losses.BinaryCrossentropy()

    def train(self, X_train, y_train, epochs=20, batch_size=32, verbose=True):
        """Train the combiner. X_train: [N, 4] modality scores; y_train: [N] binary label."""
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)

        loss_history = []
        for epoch in range(epochs):
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, len(X_train), batch_size):
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]

                with tf.GradientTape() as tape:
                    predictions = self.model(X_batch, training=True)
                    loss = self.loss_fn(y_batch, predictions)

                gradients = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

                epoch_loss += float(loss.numpy())
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

        return loss_history

    def predict(self, X):
        """Return the anomaly probability for each row of X ([N, 4] or [4])."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model(X, training=False).numpy().flatten()

    def save(self, path="results/models/combiner_nn.weights.h5"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save_weights(path)

    def load(self, path="results/models/combiner_nn.weights.h5", n_modalities=4):
        # A real forward pass (instead of `.build()`) ensures each Dense sub-layer
        # builds its weights with the correct shape before load_weights.
        self.model(np.zeros((1, n_modalities), dtype=np.float32))
        self.model.load_weights(path)


if __name__ == '__main__':
    # Synthetic smoke-test data; real training data comes from
    # src/utils/data_loader.py + src/fl/attacked_learner.py.
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((1000, 4))
    y_train = rng.integers(0, 2, 1000)

    combiner = DetectionCombiner()
    combiner.train(X_train, y_train, epochs=10)
    combiner.save()
    print("Model saved to results/models/combiner_nn.weights.h5")
