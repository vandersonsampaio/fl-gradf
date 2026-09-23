"""
Generates labeled datasets (4-detection-modality scores -> attack label) from
real FL rounds with injected attacks, to train `AnomalyCombinerNN`/
`DetectionCombiner` (src/detection/combiner_nn.py) and `RLAttackClassifier`
(src/classification/rl_classifier.py) on real signals instead of random noise.
"""

import time
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.classification.attack_simulator import AttackSimulator
from src.defense.rl_selector import RLDefenseSelector
from src.detection.modality_recorder import ATTACK_LABELS, ModalityRecorder
from src.fl.attacked_learner import AttackedFederatedLearner, RotatingAttackedFederatedLearner
from src.fl.federated_learner import ParticipantData, RoundResult
from src.utils.metrics import fairness_std

# Ensures the 7 aggregation strategies are registered in
# src.fl.federated_learner before any code below tries to instantiate an
# AttackedFederatedLearner with aggregation='median'/'krum'/etc.
import src.defense.aggregation_methods  # noqa: F401

__all__ = [
    "ATTACK_LABELS",
    "ModalityRecorder",
    "RecordingFederatedLearner",
    "generate_detection_dataset",
    "generate_selector_experiences",
    "generate_rotating_selector_experiences",
    "load_dataset_participants",
]


class RecordingFederatedLearner(AttackedFederatedLearner):
    """AttackedFederatedLearner that records (modality scores, is_byzantine, attack
    type) per client/round in `self.records`, to build training datasets."""

    def __init__(self, *args, recorder: Optional[ModalityRecorder] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = recorder or ModalityRecorder()
        self.records: List[Tuple[np.ndarray, bool, str]] = []

    def _run_round(
        self,
        round_num: int,
        participants: List[ParticipantData],
        root_data: Optional[Dict],
    ) -> RoundResult:
        active = self._is_active(round_num)
        param_updates, is_byz_list = self._compute_param_updates(participants, active)

        eval_data = root_data or {"X": participants[0].X_test, "y": participants[0].y_test}
        old_acc = self._make_model(participants[0].n_features).accuracy(
            eval_data["X"], eval_data["y"]
        )

        for i, (p, update) in enumerate(zip(participants, param_updates)):
            candidate = self._make_model(p.n_features)
            candidate.set_params(self._global_params + update)  # type: ignore[operator]
            new_acc = candidate.accuracy(eval_data["X"], eval_data["y"])

            peers = [u for j, u in enumerate(param_updates) if j != i]
            scores = self.recorder.score(p.id, update, peers, old_acc, new_acc)
            label = self.attack_type if is_byz_list[i] else "none"
            self.records.append((scores, is_byz_list[i], label))

        agg_kwargs: Dict = {}
        if self.aggregation == "fltrust":
            srv = self._make_model(participants[0].n_features)
            srv_delta = srv.fit(root_data["X"], root_data["y"])  # type: ignore[index]
            agg_kwargs["server_update"] = srv_delta
        else:
            agg_kwargs["sample_sizes"] = [p.n_train for p in participants]

        agg_delta, metadata = self._strategy.aggregate(param_updates, **agg_kwargs)
        self._global_params = self._global_params + agg_delta  # type: ignore[operator]

        per_acc = {
            p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test)
            for p in participants
        }
        global_acc = float(np.mean(list(per_acc.values())))

        trust_scores: Optional[Dict[str, float]] = None
        n_accepted: Optional[int] = None
        if metadata and "trust_scores" in metadata:
            raw = metadata["trust_scores"]
            trust_scores = {p.id: ts for p, ts in zip(participants, raw)}
            n_accepted = sum(1 for ts in raw if ts > 0)

        return RoundResult(round_num, global_acc, per_acc, trust_scores, n_accepted)


def generate_detection_dataset(
    participants: List[ParticipantData],
    root_data: Optional[Dict] = None,
    n_rounds: int = 15,
    attack_types: Optional[List[str]] = None,
    byzantine_fraction: float = 0.3,
    aggregation: str = "fedavg",
    n_classes: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs FL rounds with and without each attack type and collects, per
    client/round, the 4 detection scores and the corresponding label.

    Always includes a clean pass (attack_type='none') as the control class,
    so the labels aren't biased toward attacks.

    Returns
    -------
    X : ndarray (N, 4)          scores of the 4 modalities
    y_binary : ndarray (N,)     1 = attacked, 0 = honest
    y_attack_type : ndarray (N,) textual label ('none' or one of the attack types)
    """
    attack_types = attack_types or AttackSimulator.ALL_ATTACK_TYPES
    n_clients = len(participants)
    n_byz = max(1, int(n_clients * byzantine_fraction))
    byzantine_ids = list(range(n_byz))

    records: List[Tuple[np.ndarray, bool, str]] = []
    for attack_type in ["none"] + list(attack_types):
        learner = RecordingFederatedLearner(
            n_rounds=n_rounds,
            aggregation=aggregation,
            n_classes=n_classes,
            attack_type=attack_type,
            byzantine_ids=byzantine_ids,
            seed=seed,
        )
        learner.train(participants, root_data=root_data, verbose=False)
        records.extend(learner.records)

    X = np.stack([r[0] for r in records])
    y_binary = np.array([1 if r[1] else 0 for r in records], dtype=np.int32)
    y_attack_type = np.array([r[2] for r in records])

    return X, y_binary, y_attack_type


def _selector_reward(attack_type: str, new_acc: float, baseline_acc: float, tol: float = 0.02) -> float:
    """Simplified reward from Definition 7:
    the dominant term is accuracy retention against the clean baseline; the
    latency and fairness terms are left to experiments with full pipeline
    visibility (multiple hospitals, timing measurement)."""
    retained = new_acc >= baseline_acc - tol
    reward = 1.0 if retained else -0.5
    if attack_type != "none" and retained:
        reward += 0.5  # attack neutralized: accuracy held despite the attack
    if attack_type == "none" and not retained:
        reward -= 0.1  # false positive: the strategy degraded an honest round
    return reward


def generate_selector_experiences(
    participants: List[ParticipantData],
    actions: List[str],
    attack_types: Optional[List[str]] = None,
    n_rounds: int = 5,
    byzantine_fraction: float = 0.3,
    n_classes: Optional[int] = None,
    seed: int = 0,
) -> List[Tuple[np.ndarray, int, float, np.ndarray]]:
    """Runs real FL episodes (attack type x aggregation strategy) and returns
    experiences (state, action_idx, reward, next_state) to train
    `RLDefenseSelector.train_step()`.
    """
    attack_types = attack_types or (["none"] + list(AttackSimulator.ALL_ATTACK_TYPES))
    attack_to_idx = {a: i for i, a in enumerate(attack_types)}

    n_clients = len(participants)
    n_byz = max(1, int(n_clients * byzantine_fraction))
    byzantine_ids = list(range(n_byz))

    baseline_learner = AttackedFederatedLearner(
        n_rounds=n_rounds, aggregation="fedavg", n_classes=n_classes,
        attack_type="none", byzantine_ids=byzantine_ids, seed=seed,
    )
    baseline_start = time.perf_counter()
    baseline_history = baseline_learner.train(participants, verbose=False)
    baseline_latency = time.perf_counter() - baseline_start
    baseline_acc = baseline_history[-1].global_accuracy
    baseline_fairness = fairness_std(baseline_history[-1].per_participant_accuracy)

    experiences: List[Tuple[np.ndarray, int, float, np.ndarray]] = []
    for attack_type in attack_types:
        attack_idx = attack_to_idx[attack_type]
        state = RLDefenseSelector.build_state(attack_idx, 1.0, baseline_acc, baseline_fairness, baseline_latency)
        for action_idx, action in enumerate(actions):
            learner = AttackedFederatedLearner(
                n_rounds=n_rounds, aggregation=action, n_classes=n_classes,
                attack_type=attack_type, byzantine_ids=byzantine_ids, seed=seed,
            )
            ep_start = time.perf_counter()
            history = learner.train(participants, verbose=False)
            ep_latency = time.perf_counter() - ep_start
            new_acc = history[-1].global_accuracy
            new_fairness = fairness_std(history[-1].per_participant_accuracy)

            reward = _selector_reward(attack_type, new_acc, baseline_acc)
            next_state = RLDefenseSelector.build_state(attack_idx, 1.0, new_acc, new_fairness, ep_latency)
            experiences.append((state, action_idx, reward, next_state))

    return experiences


def generate_rotating_selector_experiences(
    participants: List[ParticipantData],
    actions: List[str],
    attack_sequence: List[str],
    n_rounds: int = 9,
    byzantine_fraction: float = 0.2,
    n_classes: Optional[int] = None,
    seed: int = 0,
) -> List[Tuple[np.ndarray, int, float, np.ndarray]]:
    """Like `generate_selector_experiences`, but each episode exposes the
    candidate action to a ROTATION of attacks (`attack_sequence`, via
    `RotatingAttackedFederatedLearner`), not a single fixed attack."""
    attack_labels = ["none"] + list(dict.fromkeys(attack_sequence))  # preserve order, drop duplicates
    attack_to_idx = {a: i for i, a in enumerate(attack_labels)}

    n_clients = len(participants)
    n_byz = max(1, int(n_clients * byzantine_fraction))
    byzantine_ids = list(range(n_byz))

    baseline_learner = AttackedFederatedLearner(
        n_rounds=n_rounds, aggregation="fedavg", n_classes=n_classes,
        attack_type="none", byzantine_ids=byzantine_ids, seed=seed,
    )
    baseline_start = time.perf_counter()
    baseline_history = baseline_learner.train(participants, verbose=False)
    baseline_latency = time.perf_counter() - baseline_start
    baseline_acc = baseline_history[-1].global_accuracy
    baseline_fairness = fairness_std(baseline_history[-1].per_participant_accuracy)

    experiences: List[Tuple[np.ndarray, int, float, np.ndarray]] = []
    for action_idx, action in enumerate(actions):
        learner = RotatingAttackedFederatedLearner(
            n_rounds=n_rounds, strategy=action, n_classes=n_classes,
            attack_sequence=attack_sequence, byzantine_ids=byzantine_ids, seed=seed,
        )
        ep_start = time.perf_counter()
        history = learner.train(participants, verbose=False)
        ep_latency = time.perf_counter() - ep_start
        new_acc = history[-1].global_accuracy
        new_fairness = fairness_std(history[-1].per_participant_accuracy)

        last_attack_type = attack_sequence[(n_rounds - 1) % len(attack_sequence)]
        attack_idx = attack_to_idx[last_attack_type]

        state = RLDefenseSelector.build_state(attack_idx, 1.0, baseline_acc, baseline_fairness, baseline_latency)
        next_state = RLDefenseSelector.build_state(attack_idx, 1.0, new_acc, new_fairness, ep_latency)
        reward = _selector_reward(last_attack_type, new_acc, baseline_acc)
        experiences.append((state, action_idx, reward, next_state))

    return experiences


def load_dataset_participants(
    dataset: str = "mnist",
    split: str = "non_iid",
    n_clients: int = 10,
    data_root: str = "data/processed",
    root_size: Optional[int] = None,
    root_seed: int = 0,
) -> Tuple[List[ParticipantData], Dict]:
    """Loads real participants from `data/processed/{dataset}/{split}/client_i`
    and the server validation set, produced by `data/download_datasets.py`
    (used by the experiments — src/experiments/ — and by the notebooks).
    """
    base = os.path.join(data_root, dataset, split)
    participants = [
        ParticipantData.from_npy(os.path.join(base, f"client_{i}"))
        for i in range(n_clients)
    ]
    root_data = ParticipantData.load_server_val(os.path.join(base, "server_val"))
    if root_size is not None and root_size < len(root_data["y"]):
        rng = np.random.default_rng(root_seed)
        idx = rng.choice(len(root_data["y"]), size=root_size, replace=False)
        root_data = {"X": root_data["X"][idx], "y": root_data["y"][idx]}
    return participants, root_data
