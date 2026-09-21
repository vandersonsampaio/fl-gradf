"""End-to-end integration smoke test for the full GRADF pipeline:
detection -> combiner -> classifier -> selector -> hardening -> aggregation -> XAI,
orchestrated by `GRADFFederatedLearner`."""

import numpy as np

from src.classification.rl_classifier import RLAttackClassifier
from src.fl.gradf_learner import GRADFFederatedLearner
from src.utils.data_loader import generate_detection_dataset
from src.xai.audit_trail import AuditTrail
from src.xai.explanation_generator import XAIExplainer


def test_gradf_learner_runs_end_to_end(tmp_path, synthetic_participants):
    n_rounds = 4
    explainer = XAIExplainer(audit_trail=AuditTrail(path=str(tmp_path / "audit.jsonl")))

    learner = GRADFFederatedLearner(
        n_rounds=n_rounds,
        aggregation="fedavg",
        n_classes=2,
        attack_type="sign_flipping",
        byzantine_ids=[0, 1],
        explainer=explainer,
    )

    history = learner.train(synthetic_participants, verbose=False)

    assert len(history) == n_rounds
    for result in history:
        assert 0.0 <= result.global_accuracy <= 1.0
        assert not np.isnan(result.global_accuracy)


def test_audit_trail_has_one_entry_per_hospital_per_round(tmp_path, synthetic_participants):
    n_rounds = 3
    audit_path = str(tmp_path / "audit.jsonl")
    explainer = XAIExplainer(audit_trail=AuditTrail(path=audit_path))

    learner = GRADFFederatedLearner(
        n_rounds=n_rounds,
        aggregation="fedavg",
        n_classes=2,
        attack_type="poisoning",
        byzantine_ids=[0],
        explainer=explainer,
    )
    learner.train(synthetic_participants, verbose=False)

    records = explainer.audit_trail.load()
    assert len(records) == n_rounds * len(synthetic_participants)
    assert all(r["decision"] in ("ACCEPTED", "REJECTED") for r in records)
    assert all("narrative" in r and "counterfactual" in r for r in records)


def test_gradf_classifier_flags_byzantine_clients_more_often_than_honest(tmp_path, synthetic_participants):
    attack_labels = ["none", "sign_flipping"]
    label_to_idx = {label: i for i, label in enumerate(attack_labels)}

    X, _y_binary, y_attack_type = generate_detection_dataset(
        synthetic_participants,
        n_rounds=10,
        attack_types=["sign_flipping"],
        byzantine_fraction=0.3,
        aggregation="fedavg",
        n_classes=2,
        seed=7,
    )
    y_idx = np.array([label_to_idx[t] for t in y_attack_type])

    classifier = RLAttackClassifier(n_attacks=len(attack_labels))
    classifier.train(X, y_idx, epochs=40, batch_size=16, verbose=False)

    n_rounds = 6
    byzantine_ids = {0, 1, 2}
    audit_path = str(tmp_path / "audit.jsonl")
    explainer = XAIExplainer(audit_trail=AuditTrail(path=audit_path))

    learner = GRADFFederatedLearner(
        n_rounds=n_rounds,
        aggregation="fedavg",
        n_classes=2,
        attack_type="sign_flipping",
        byzantine_ids=sorted(byzantine_ids),
        classifier=classifier,
        attack_labels=attack_labels,
        explainer=explainer,
    )
    learner.train(synthetic_participants, verbose=False)

    records = explainer.audit_trail.load()
    byz_client_ids = {f"client_{i}" for i in byzantine_ids}

    byz_records = [r for r in records if r["hospital_id"] in byz_client_ids]
    honest_records = [r for r in records if r["hospital_id"] not in byz_client_ids]

    byz_flagged_rate = np.mean([r["predicted_attack_type"] != "none" for r in byz_records])
    honest_flagged_rate = np.mean([r["predicted_attack_type"] != "none" for r in honest_records])

    assert byz_flagged_rate > honest_flagged_rate
