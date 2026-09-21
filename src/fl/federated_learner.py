"""
Federated Learning training loop with pluggable aggregation strategies.

Default aggregation is FLTrust (Byzantine-robust).  Pass aggregation='fedavg'
for standard weighted averaging.

Typical usage:

    from src.fl.federated_learner import FederatedLearner, ParticipantData

    # Load from partitioned .npy files (auto-flattens images)
    participants = [
        ParticipantData.from_npy(f'data/processed/mnist/non_iid/client_{i}')
        for i in range(10)
    ]
    root_data = ParticipantData.load_server_val('data/processed/mnist/non_iid/server_val')

    learner = FederatedLearner(n_rounds=30, aggregation='fltrust', n_classes=10)
    history = learner.train(participants, root_data=root_data)
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ParticipantData:
    """Data held by one FL participant.

    X arrays must be 2-D (N, features); use from_npy() or from_dict() which
    handle flattening automatically.
    """

    id: str
    X_train: np.ndarray   # shape (n, features), float32
    y_train: np.ndarray   # shape (n,), int
    X_test: np.ndarray    # shape (m, features), float32
    y_test: np.ndarray    # shape (m,), int

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_npy(cls, client_dir: str, participant_id: str = None) -> "ParticipantData":
        """Load from a directory produced by data/download_datasets.py.

        Files expected: X_train.npy, y_train.npy, X_test.npy, y_test.npy
        Multi-dimensional arrays (images) are flattened to (N, features).
        """
        pid = participant_id or os.path.basename(os.path.normpath(client_dir))

        def _load(fname, flat=True):
            arr = np.load(os.path.join(client_dir, fname))
            if flat:
                return arr.reshape(len(arr), -1).astype(np.float32)
            return arr.astype(np.int32)

        return cls(
            id=pid,
            X_train=_load("X_train.npy"),
            y_train=_load("y_train.npy", flat=False),
            X_test=_load("X_test.npy"),
            y_test=_load("y_test.npy", flat=False),
        )

    @staticmethod
    def load_server_val(val_dir: str) -> Dict[str, np.ndarray]:
        """Load the server validation set (X.npy, y.npy) used by FLTrust."""
        X = np.load(os.path.join(val_dir, "X.npy"))
        X = X.reshape(len(X), -1).astype(np.float32)
        y = np.load(os.path.join(val_dir, "y.npy")).astype(np.int32)
        return {"X": X, "y": y}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_features(self) -> int:
        # Works whether X_train is already 2-D or still multi-dimensional
        return int(np.prod(self.X_train.shape[1:]))


@dataclass
class RoundResult:
    """Outcome of one FL communication round."""

    round_num: int
    global_accuracy: float
    per_participant_accuracy: Dict[str, float]
    trust_scores: Optional[Dict[str, float]] = None
    n_accepted: Optional[int] = None  # clients with trust_score > 0 (FLTrust)


# ---------------------------------------------------------------------------
# Local model — mini-batch SGD logistic regression (binary + multiclass)
# ---------------------------------------------------------------------------

class _LogisticModel:
    """
    Logistic regression with mini-batch SGD supporting binary and K-class
    classification.

    Parameters are stored as a single flat vector so that FL aggregation
    (cosine similarity, weighted mean) operates on a consistent 1-D array
    regardless of the number of classes:

        params = [W_flat (n_features × K), b (K,)]   total length = (n_features+1)*K

    For binary classification K=1 (sigmoid); for K≥3 K=n_classes (softmax).
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        lr: float = 0.01,
        n_epochs: int = 5,
        batch_size: int = 32,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # K = 1 for binary (one weight vector + one bias)
        # K = n_classes for softmax
        self._K = 1 if n_classes == 2 else n_classes
        self._params = np.zeros(n_features * self._K + self._K, dtype=np.float64)

    # -- helpers -------------------------------------------------------------

    def _W(self) -> np.ndarray:
        """View of weight matrix (n_features, K) inside _params."""
        return self._params[: self.n_features * self._K].reshape(self.n_features, self._K)

    def _b(self) -> np.ndarray:
        """View of bias vector (K,) inside _params."""
        return self._params[self.n_features * self._K :]

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250.0, 250.0)))

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # -- inference -----------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(len(X), -1)
        if self.n_classes == 2:
            return self._sigmoid(X @ self._W()[:, 0] + self._b()[0])
        return self._softmax(X @ self._W() + self._b())

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        if self.n_classes == 2:
            return (proba >= 0.5).astype(int)
        return proba.argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    # -- training ------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Train on (X, y) and return the flat parameter *update* (delta).

        W and b are views into _params, so in-place updates automatically
        keep _params consistent — no need to pack at the end.
        """
        X = X.reshape(len(X), -1).astype(np.float64)
        y = y.astype(np.int32)
        p0 = self._params.copy()
        W, b = self._W(), self._b()   # live views into _params
        n = len(X)

        for _ in range(self.n_epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                Xb, yb = X[idx], y[idx]
                m = len(idx)

                if self.n_classes == 2:
                    prob = self._sigmoid(Xb @ W[:, 0] + b[0])
                    err = prob - yb.astype(np.float64)
                    W[:, 0] -= self.lr * (Xb.T @ err) / m
                    b[0] -= self.lr * err.mean()
                else:
                    prob = self._softmax(Xb @ W + b)
                    one_hot = np.zeros_like(prob)
                    one_hot[np.arange(m), yb] = 1.0
                    err = prob - one_hot          # (m, K)
                    W -= self.lr * (Xb.T @ err) / m
                    b -= self.lr * err.mean(axis=0)

        return self._params - p0

    # -- state management ----------------------------------------------------

    def set_params(self, params: np.ndarray) -> None:
        self._params = params.copy()

    @property
    def params(self) -> np.ndarray:
        return self._params.copy()


class _CNNModel:
    """
    CNN local model (Keras), used when `FederatedLearner(model_type='cnn',
    input_shape=...)` — Decision A4 (`references/plano_gradf_iclr2027.md`):
    stepping up from logistic regression to a small CNN.

    Deliberately implements the SAME public interface as `_LogisticModel`
    (`fit`/`set_params`/`params`/`accuracy`/`predict`/`predict_proba`,
    operating on a flat parameter vector): ALL of the aggregation
    infrastructure (`AggregationStrategy.aggregate`), attack
    (`AttackSimulator.poison_update`), and hardening (`HardeningPipeline`,
    L2-norm/skewness checks) operates blindly on 1-D `np.ndarray`s — it
    doesn't know, and doesn't need to know, that the parameters come from a
    CNN rather than logistic regression. `FederatedLearner._make_model` is
    the ONLY dispatch point between the two; no other file
    (`attacked_learner.py`, `gradf_learner.py`, `hardening.py`,
    `data_loader.py`, `exp*.py`) needed to change to gain CNN support.

    Keras's parameters (`model.get_weights()`, a list of per-layer arrays)
    are flattened into a single sequence to form the flat vector;
    `set_params` does the inverse using the `shapes`/`sizes` captured at
    construction time (the architecture — and therefore the shapes — doesn't
    change after creation).
    """

    def __init__(
        self,
        n_features: int,
        input_shape: Tuple[int, int, int],
        n_classes: int = 10,
        lr: float = 0.01,
        n_epochs: int = 5,
        batch_size: int = 32,
    ) -> None:
        expected_features = int(np.prod(input_shape))
        if expected_features != n_features:
            raise ValueError(
                f"input_shape={input_shape} implies {expected_features} features, "
                f"but the data has n_features={n_features}"
            )
        self.n_features = n_features
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self._model = self._build()
        weights = self._model.get_weights()
        self._shapes = [w.shape for w in weights]
        self._sizes = [int(w.size) for w in weights]

    def _build(self):
        # Local import: keeps the cost of importing TensorFlow (heavy) scoped
        # to callers that actually use model_type='cnn' — the rest of the
        # module (used by the whole test/experiment suite with
        # `_LogisticModel`) doesn't pay that cost.
        from tensorflow import keras

        model = keras.Sequential([
            keras.layers.Input(shape=self.input_shape),
            keras.layers.Conv2D(8, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(self.n_classes, activation="softmax"),
        ])
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=self.lr),
            loss="sparse_categorical_crossentropy",
        )
        return model

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        return X.reshape((-1,) + self.input_shape).astype("float32")

    # -- inference -------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._reshape(X), verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    # -- training ----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Train locally and return the parameter delta — same contract as
        `_LogisticModel.fit`."""
        p0 = self.params
        self._model.fit(
            self._reshape(X), y.astype("int32"),
            epochs=self.n_epochs, batch_size=self.batch_size, verbose=0,
        )
        return self.params - p0

    # -- state management ----------------------------------------------------

    def set_params(self, params: np.ndarray) -> None:
        weights = []
        offset = 0
        for shape, size in zip(self._shapes, self._sizes):
            weights.append(params[offset : offset + size].reshape(shape).astype("float32"))
            offset += size
        self._model.set_weights(weights)

    @property
    def params(self) -> np.ndarray:
        return np.concatenate([w.flatten() for w in self._model.get_weights()]).astype(np.float64)


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------

class AggregationStrategy(ABC):
    """
    Interface for FL aggregation methods.

    Receives a list of flat client parameter updates, returns:
        (aggregated_update, metadata_dict | None)
    """

    @abstractmethod
    def aggregate(
        self,
        updates: List[np.ndarray],
        **kwargs,
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        ...


class FedAvgStrategy(AggregationStrategy):
    """FedAvg: sample-size weighted mean of client updates.

    McMahan et al., "Communication-Efficient Learning of Deep Networks from
    Decentralized Data", AISTATS 2017.
    """

    def aggregate(
        self,
        updates: List[np.ndarray],
        sample_sizes: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, None]:
        if sample_sizes is None:
            w = np.ones(len(updates)) / len(updates)
        else:
            total = float(sum(sample_sizes))
            w = np.array(sample_sizes, dtype=float) / total
        return sum(wi * u for wi, u in zip(w, updates)), None  # type: ignore[return-value]


class FLTrustStrategy(AggregationStrategy):
    """FLTrust: weight each client update by ReLU(cosine_similarity(update, server_ref)),
    clip update norm to server norm before summing.

    Cao et al., "FLTrust: Byzantine-robust Federated Learning via Trust
    Bootstrapping", NDSS 2022.
    """

    def aggregate(
        self,
        updates: List[np.ndarray],
        server_update: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        if server_update is None:
            raise ValueError("FLTrustStrategy requires server_update kwarg")

        server_norm = float(np.linalg.norm(server_update))

        if server_norm < 1e-10:
            logger.warning("FLTrust: server gradient near-zero; falling back to mean")
            return np.mean(updates, axis=0), {"trust_scores": [0.0] * len(updates)}

        trust_scores: List[float] = []
        clipped: List[np.ndarray] = []

        for upd in updates:
            norm = float(np.linalg.norm(upd))
            if norm < 1e-10:
                trust_scores.append(0.0)
                clipped.append(np.zeros_like(upd))
                continue
            ts = max(0.0, float(np.dot(upd, server_update) / (norm * server_norm)))
            trust_scores.append(ts)
            clipped.append((server_norm / norm) * upd)  # clip to server norm

        total_trust = sum(trust_scores)
        if total_trust < 1e-10:
            logger.warning("FLTrust: all clients distrusted; returning server update")
            return server_update.copy(), {"trust_scores": trust_scores}

        agg = sum(ts * u for ts, u in zip(trust_scores, clipped)) / total_trust  # type: ignore[return-value]
        return agg, {"trust_scores": trust_scores}


_STRATEGIES: Dict[str, type] = {
    "fltrust": FLTrustStrategy,
    "fedavg": FedAvgStrategy,
}


def register_strategy(name: str, cls: type) -> None:
    """Register a custom AggregationStrategy under a new name."""
    _STRATEGIES[name] = cls


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FederatedLearner:
    """
    Federated learning training loop with pluggable aggregation.

    Parameters
    ----------
    n_rounds : int
        Number of communication rounds.
    aggregation : str
        'fltrust' (default) or 'fedavg'.  Custom strategies can be registered
        with register_strategy() and referenced by name.
    n_classes : int | None
        Number of output classes.  If None, auto-detected from participant
        labels on the first call to train().  Use 2 for binary, 10 for
        MNIST/CIFAR-10.
    local_lr : float
        Learning rate for local SGD.
    local_epochs : int
        Local epochs per round.
    local_batch_size : int
        Mini-batch size for local SGD.
    root_fraction : float
        Fraction of the first participant's training data to use as the
        FLTrust server root dataset when no root_data is supplied.
    model_type : str
        'logistic' (default, `_LogisticModel`) or 'cnn' (`_CNNModel` — see
        Decision A4, `references/plano_gradf_iclr2027.md`). With 'cnn',
        `input_shape` is required.
    input_shape : (int, int, int) | None
        Original image shape (e.g. `(28,28,1)` for MNIST, `(32,32,3)` for
        CIFAR-10) — only used when `model_type='cnn'`; `ParticipantData`
        still flattens images to 2-D, and `_CNNModel` reconstructs the shape
        internally from this parameter.

    Notes
    -----
    With `model_type='logistic'` (default), the task model is multinomial
    logistic regression (softmax cross-entropy for n_classes≥3, binary for
    n_classes=2). Feature vectors must be normalised before training. With
    `model_type='cnn'`, the model is a small CNN (Keras) — see `_CNNModel`.
    In both cases, `_make_model` is the only dispatch point: all of the
    aggregation/attack/hardening infrastructure operates on the flat
    parameter vector without knowing which of the two models produced it.
    """

    def __init__(
        self,
        n_rounds: int = 20,
        aggregation: str = "fltrust",
        n_classes: Optional[int] = None,
        local_lr: float = 0.01,
        local_epochs: int = 5,
        local_batch_size: int = 32,
        root_fraction: float = 0.05,
        model_type: str = "logistic",
        input_shape: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        if aggregation not in _STRATEGIES:
            raise ValueError(
                f"Unknown aggregation {aggregation!r}. "
                f"Available: {sorted(_STRATEGIES)}"
            )
        if model_type not in ("logistic", "cnn"):
            raise ValueError(f"Unknown model_type {model_type!r}. Available: 'logistic', 'cnn'")
        if model_type == "cnn" and input_shape is None:
            raise ValueError("model_type='cnn' requires input_shape=(H, W, C)")

        self.n_rounds = n_rounds
        self.aggregation = aggregation
        self.n_classes = n_classes          # resolved in train() if None
        self.local_lr = local_lr
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.root_fraction = root_fraction
        self.model_type = model_type
        self.input_shape = input_shape

        self._strategy: AggregationStrategy = _STRATEGIES[aggregation]()
        self._global_params: Optional[np.ndarray] = None  # flat param vector
        self.history: List[RoundResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        participants: List[ParticipantData],
        root_data: Optional[Dict[str, np.ndarray]] = None,
        verbose: bool = True,
    ) -> List[RoundResult]:
        """Run FL training.

        Parameters
        ----------
        participants : List[ParticipantData]
            One entry per FL client.  Use ParticipantData.from_npy() to load
            from the partitioned dataset directories.
        root_data : dict {'X': ndarray, 'y': ndarray} | None
            Clean server validation set for FLTrust.  Use
            ParticipantData.load_server_val() to load.  If None and
            aggregation='fltrust', a small slice is carved from
            participants[0].
        verbose : bool
            Print round-level progress.
        """
        # Resolve n_classes
        if self.n_classes is None:
            all_labels = np.concatenate([p.y_train for p in participants])
            self.n_classes = int(all_labels.max()) + 1
            logger.info("auto-detected n_classes=%d", self.n_classes)

        n_features = participants[0].n_features
        if self.model_type == "cnn":
            # Size/initial values are determined by the Keras architecture
            # (default Glorot initialization), not by a closed-form formula
            # like the logistic case — zero-init would break symmetry
            # between convolutional filters, a known CNN training issue, not
            # just a sizing convenience.
            self._global_params = self._instantiate_model(n_features).params
        else:
            _K = 1 if self.n_classes == 2 else self.n_classes
            self._global_params = np.zeros(n_features * _K + _K)
        self.history = []

        if self.aggregation == "fltrust" and root_data is None:
            root_data = self._carve_root(participants[0])
            logger.info(
                "FLTrust: auto-carved %d root samples from %s",
                len(root_data["y"]),
                participants[0].id,
            )

        for round_num in range(self.n_rounds):
            result = self._run_round(round_num + 1, participants, root_data)
            self.history.append(result)

            if verbose:
                suffix = ""
                if result.trust_scores is not None:
                    suffix = f" | trusted={result.n_accepted}/{len(participants)}"
                print(f"Round {result.round_num:3d}: acc={result.global_accuracy:.4f}{suffix}")

        return self.history

    @property
    def global_params(self) -> Optional[np.ndarray]:
        """Flat parameter vector [W_flat, b] of the current global model."""
        return None if self._global_params is None else self._global_params.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_round(
        self,
        round_num: int,
        participants: List[ParticipantData],
        root_data: Optional[Dict],
    ) -> RoundResult:
        param_updates: List[np.ndarray] = []

        for p in participants:
            model = self._make_model(p.n_features)
            delta = model.fit(p.X_train, p.y_train)
            param_updates.append(delta)

        agg_kwargs: Dict = {}
        if self.aggregation == "fltrust":
            server_model = self._make_model(participants[0].n_features)
            server_delta = server_model.fit(root_data["X"], root_data["y"])  # type: ignore[index]
            agg_kwargs["server_update"] = server_delta
        else:
            agg_kwargs["sample_sizes"] = [p.n_train for p in participants]

        agg_delta, metadata = self._strategy.aggregate(param_updates, **agg_kwargs)
        self._global_params = self._global_params + agg_delta  # type: ignore[operator]

        per_participant_acc = {
            p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test)
            for p in participants
        }
        global_acc = float(np.mean(list(per_participant_acc.values())))

        trust_scores: Optional[Dict[str, float]] = None
        n_accepted: Optional[int] = None
        if metadata and "trust_scores" in metadata:
            raw = metadata["trust_scores"]
            trust_scores = {p.id: ts for p, ts in zip(participants, raw)}
            n_accepted = sum(1 for ts in raw if ts > 0)

        return RoundResult(
            round_num=round_num,
            global_accuracy=global_acc,
            per_participant_accuracy=per_participant_acc,
            trust_scores=trust_scores,
            n_accepted=n_accepted,
        )

    def _instantiate_model(self, n_features: int):
        """Build a NEW model (default initial weights), without applying
        `self._global_params` — used by `_make_model` (which applies the
        params afterward) and by `train()` to determine the size and initial
        values of the global parameter vector before the first round (see
        `train()` — zeros for logistic, Keras init for CNN)."""
        if self.model_type == "cnn":
            return _CNNModel(
                n_features,
                input_shape=self.input_shape,  # type: ignore[arg-type]
                n_classes=self.n_classes or 2,
                lr=self.local_lr,
                n_epochs=self.local_epochs,
                batch_size=self.local_batch_size,
            )
        return _LogisticModel(
            n_features,
            n_classes=self.n_classes or 2,
            lr=self.local_lr,
            n_epochs=self.local_epochs,
            batch_size=self.local_batch_size,
        )

    def _make_model(self, n_features: int):
        """Instantiate a local model seeded with the current global params."""
        model = self._instantiate_model(n_features)
        model.set_params(self._global_params)  # type: ignore[arg-type]
        return model

    def _carve_root(self, participant: ParticipantData) -> Dict[str, np.ndarray]:
        n = max(10, int(participant.n_train * self.root_fraction))
        idx = np.random.choice(participant.n_train, n, replace=False)
        return {"X": participant.X_train[idx], "y": participant.y_train[idx]}
