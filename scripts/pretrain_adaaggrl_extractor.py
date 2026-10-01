"""
Passo Zero (`references/1_roadmap_frentes_futuras.md`): pretrains the REAL
CNN feature extractor for AdaAggRL's environmental cues
(`src/defense/adaaggrl_agent.py::PretrainedCNNFeatureExtractor`), replacing
the frozen-random-weights `RandomCNNFeatureExtractor` documented as a floor,
not the paper's real performance.

Trains an ordinary image classifier (same architecture as
`src/fl/federated_learner.py::_CNNModel`: Conv2D(8)->MaxPool->Conv2D(16)
->MaxPool->Flatten->Dense(32,relu)->Dense(n_classes,softmax)) on the RAW
MNIST/CIFAR-10 TEST split (`data/raw/{dataset}/X_test.npy`/`y_test.npy`) —
deliberately NOT the train pool, since both the FL clients' partitions and
the FLTrust `server_val` root used throughout `src/experiments/exp10_*`
are carved from `X_train`/`y_train` by `data/download_datasets.py`; using
the disjoint test split for pretraining avoids leaking any information the
downstream AdaAggRL grid run measures. Only the frozen Dense(32, relu)
activations are used afterwards (the final softmax head is training
scaffolding, discarded at inference by `PretrainedCNNFeatureExtractor`).

Usage:
    python -m scripts.pretrain_adaaggrl_extractor --dataset mnist
"""

import argparse
import os

import numpy as np


def _build_classifier(input_shape, n_classes, feature_dim):
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(8, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(feature_dim, activation="relu", name="features"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def pretrain(
    dataset: str = "mnist",
    feature_dim: int = 32,
    epochs: int = 10,
    batch_size: int = 64,
    val_fraction: float = 0.1,
    seed: int = 0,
    raw_root: str = "data/raw",
    output_dir: str = "results/models",
) -> float:
    input_shapes = {"mnist": (28, 28, 1), "cifar10": (32, 32, 3)}
    if dataset not in input_shapes:
        raise ValueError(f"Unsupported dataset '{dataset}'; expected one of {list(input_shapes)}")
    input_shape = input_shapes[dataset]

    X = np.load(os.path.join(raw_root, dataset, "X_test.npy")).astype("float32")
    y = np.load(os.path.join(raw_root, dataset, "y_test.npy"))
    X = X.reshape((-1,) + input_shape)
    n_classes = int(y.max()) + 1

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    n_val = int(len(y) * val_fraction)
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]

    from tensorflow import keras
    keras.utils.set_random_seed(seed)

    model = _build_classifier(input_shape, n_classes, feature_dim)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size, verbose=2,
    )
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, f"adaaggrl_pretrained_extractor_{dataset}.weights.h5")
    model.save_weights(weights_path)
    print(f"Saved pretrained extractor weights to {weights_path} (held-out val accuracy: {val_acc:.4f})")
    return float(val_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pretrain(
        dataset=args.dataset, feature_dim=args.feature_dim, epochs=args.epochs,
        batch_size=args.batch_size, val_fraction=args.val_fraction, seed=args.seed,
    )
