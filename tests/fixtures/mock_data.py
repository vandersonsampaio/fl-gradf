"""Synthetic data generators shared across tests (unit and integration) —
avoids depending on MNIST/CIFAR-10 on disk for fast tests."""

import numpy as np

from src.fl.federated_learner import ParticipantData


def make_synthetic_participants(
    n_clients: int = 10,
    n_features: int = 20,
    n_train: int = 60,
    n_test: int = 20,
    seed: int = 0,
):
    """Generates `n_clients` ParticipantData with a linearly separable binary
    classification problem (same `true_w` for all clients — IID)."""
    rng = np.random.default_rng(seed)
    true_w = rng.standard_normal(n_features)
    participants = []
    for i in range(n_clients):
        X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
        y_train = (X_train @ true_w > 0).astype(np.int32)
        X_test = rng.standard_normal((n_test, n_features)).astype(np.float32)
        y_test = (X_test @ true_w > 0).astype(np.int32)
        participants.append(
            ParticipantData(
                id=f"client_{i}", X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
            )
        )
    return participants
