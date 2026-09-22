"""
GRADFFederatedLearner: orchestrates the full GRADF pipeline per FL round —

    multi-modal detection -> combiner -> attack classifier -> defense
    selector -> 5-layer hardening -> aggregation -> XAI explanation

on top of the existing FL/attack infrastructure (AttackedFederatedLearner),
following the same `_run_round` override pattern used there.

Online selector learning (post-Gate-1 correction, see
`references/plano_gradf_iclr2027.md` and `src/defense/rl_selector.py`):
`RLDefenseSelector` used to be trained only once, offline, before the first
round — "adaptive" described only the architecture, not the learning
behavior. `_run_round` now tracks the previous round's accuracy/fairness/
latency (`self._prev_*`), passes that as `round_context` to
`HardeningPipeline.full_pipeline` (which uses it to give the selector the
rich 5-dimensional state via `select_action`), and calls
`self.selector.train_step` every real round with the actually observed
reward — see `_update_selector_online`.
"""

import time
from typing import Dict, List, Optional

import numpy as np

# Ensures the 7 aggregation strategies are registered before any
# aggregation=... is resolved.
import src.defense.aggregation_methods  # noqa: F401
from src.classification.rl_classifier import RLAttackClassifier
from src.defense.hardening import HardeningPipeline
from src.defense.rl_selector import RLDefenseSelector
from src.detection.combiner_nn import DetectionCombiner
from src.detection.modality_recorder import ATTACK_LABELS, ModalityRecorder
from src.fl.attacked_learner import AttackedFederatedLearner, compute_param_updates_auto
from src.fl.federated_learner import ParticipantData, RoundResult
from src.utils.metrics import fairness_std, online_selector_reward
from src.xai.explanation_generator import XAIExplainer


class GRADFFederatedLearner(AttackedFederatedLearner):
    def __init__(
        self,
        *args,
        classifier: Optional[RLAttackClassifier] = None,
        selector: Optional[RLDefenseSelector] = None,
        combiner: Optional[DetectionCombiner] = None,
        explainer: Optional[XAIExplainer] = None,
        attack_labels: Optional[List[str]] = None,
        magnitude_threshold: float = 5.0,
        accuracy_tolerance: float = 0.02,
        dp_epsilon: float = 50.0,
        dp_clipping: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.attack_labels = attack_labels or ATTACK_LABELS
        self.recorder = ModalityRecorder()
        self.classifier = classifier or RLAttackClassifier(n_attacks=len(self.attack_labels))
        self.selector = selector or RLDefenseSelector()
        self.combiner = combiner or DetectionCombiner()
        self.explainer = explainer or XAIExplainer()
        self.hardening = HardeningPipeline(
            self.classifier,
            self.selector,
            magnitude_threshold=magnitude_threshold,
            accuracy_tolerance=accuracy_tolerance,
            dp_epsilon=dp_epsilon,
            dp_clipping=dp_clipping,
        )

        # Rolling context from the previous round (acc/fairness/latency) —
        # becomes the selector's rich state via `round_context` (see the
        # module docstring). Neutral (0.0) in round 1 — standard DQN cold start.
        self._prev_global_acc = 0.0
        self._prev_fairness_std = 0.0
        self._prev_latency = 0.0

    def _run_round(
        self,
        round_num: int,
        participants: List[ParticipantData],
        root_data: Optional[Dict],
    ) -> RoundResult:
        round_start = time.perf_counter()
        active = self._is_active(round_num)
        param_updates, _is_byz_list = compute_param_updates_auto(self, participants, active, root_data)

        eval_data = root_data or {"X": participants[0].X_test, "y": participants[0].y_test}
        old_acc = self._make_model(participants[0].n_features).accuracy(
            eval_data["X"], eval_data["y"]
        )

        # server_update: reference delta required by the 'fltrust' strategy.
        # Always computed (not only when self.aggregation == 'fltrust', unlike
        # AttackedFederatedLearner) because here the per-round strategy is
        # decided by the RL selector's VOTE in Layer 3 (`selected_strategy`),
        # which can choose 'fltrust' regardless of the top-level aggregation —
        # see the note in HardeningPipeline.full_pipeline about the invariant
        # this preserves (without server_update, any round where the selector
        # votes 'fltrust' would be silently discarded, freezing the global model).
        server_root = root_data or self._carve_root(participants[0])
        server_update = self._make_model(participants[0].n_features).fit(
            server_root["X"], server_root["y"]
        )

        # Detection scores + per-hospital isolated evaluation (used both by
        # the classifier and by the XAI counterfactual).
        modality_scores = []
        per_hospital_eval = []
        for i, (p, update) in enumerate(zip(participants, param_updates)):
            candidate = self._make_model(p.n_features)
            candidate.set_params(self._global_params + update)  # type: ignore[operator]
            new_acc_i = candidate.accuracy(eval_data["X"], eval_data["y"])
            per_hospital_eval.append((old_acc, new_acc_i))

            peers = [u for j, u in enumerate(param_updates) if j != i]
            modality_scores.append(self.recorder.score(p.id, update, peers, old_acc, new_acc_i))

        def evaluate_fn(delta: np.ndarray) -> float:
            model = self._make_model(participants[0].n_features)
            model.set_params(self._global_params + delta)  # type: ignore[operator]
            return model.accuracy(eval_data["X"], eval_data["y"])

        hospital_ids = [p.id for p in participants]
        sample_sizes = [p.n_train for p in participants]

        round_context = {
            "prev_accuracy": self._prev_global_acc,
            "prev_fairness_std": self._prev_fairness_std,
            "prev_latency": self._prev_latency,
            "round_num": round_num,
        }
        agg_delta, hardening_results = self.hardening.full_pipeline(
            hospital_ids, param_updates, modality_scores, evaluate_fn,
            sample_sizes=sample_sizes, server_update=server_update,
            round_context=round_context,
        )

        if hardening_results.get("round_accepted") and agg_delta is not None:
            self._global_params = self._global_params + agg_delta  # type: ignore[operator]

        self._log_explanations(hospital_ids, modality_scores, per_hospital_eval, hardening_results)

        per_acc = {
            p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test)
            for p in participants
        }
        global_acc = float(np.mean(list(per_acc.values())))
        round_latency = time.perf_counter() - round_start
        round_fairness = fairness_std(per_acc)

        self._update_selector_online(hardening_results, old_acc, global_acc, round_context, round_fairness, round_latency)

        self._prev_global_acc = global_acc
        self._prev_fairness_std = round_fairness
        self._prev_latency = round_latency

        return RoundResult(round_num, global_acc, per_acc, trust_scores=None, n_accepted=None)

    def _update_selector_online(
        self,
        hardening_results: Dict,
        old_acc: float,
        new_acc: float,
        round_context: Dict[str, float],
        new_fairness_std: float,
        new_latency: float,
    ) -> None:
        """Real online learning for the selector (see the module docstring):
        called EVERY real FL round, not just during offline pretraining.
        Only active for selectors that implement `train_step`
        (`RLDefenseSelector`) — fixed/random/oracle/TARS selectors (TARS
        already has its own online-update mechanism built into
        `TARSLearner`) are unaffected.

        A single (state, next_state) pair is used per hospital classified
        this round: `state` uses the PREVIOUS round's context (the same one
        passed to `select_action` via `round_context`), `next_state` uses
        the SAME attack index/confidence (the classification doesn't change
        within this round) but this round's REAL outcome — not a full
        round_t -> round_{t+1} temporal chain (that would require deferring
        the update by one step, since the NEXT round's classification
        doesn't exist yet), but a genuine transition (before/after this
        decision), not the degenerate copy (`next_state = state`) previously
        used when generating offline experiences.

        The action used for credit/blame is the strategy ACTUALLY applied
        this round (`hardening_results["selected_strategy"]`, the majority
        vote) — not each hospital's individual vote — because that is what
        produced the observed outcome, not the vote of hospitals outvoted in
        the majority.
        """
        selector = self.selector
        if not hasattr(selector, "train_step"):
            return

        classifications = hardening_results.get("classifications", {})
        strategy_name = hardening_results.get("selected_strategy")
        if not classifications or strategy_name is None or strategy_name not in selector.actions:
            return

        action_idx = selector.actions.index(strategy_name)
        any_attack_detected = any(
            cls.get("attack_idx") not in (None, 0) for cls in classifications.values()
        )
        reward = online_selector_reward(any_attack_detected, old_acc, new_acc)

        for cls_info in classifications.values():
            attack_idx = cls_info.get("attack_idx")
            confidence = cls_info.get("confidence", 0.0)
            if attack_idx is None:
                continue
            state = RLDefenseSelector.build_state(
                attack_idx, confidence,
                round_context["prev_accuracy"], round_context["prev_fairness_std"], round_context["prev_latency"],
            )
            next_state = RLDefenseSelector.build_state(
                attack_idx, confidence, new_acc, new_fairness_std, new_latency,
            )
            selector.train_step(state, action_idx, reward, next_state)

    def _log_explanations(
        self,
        hospital_ids: List[str],
        modality_scores: List[np.ndarray],
        per_hospital_eval: List[tuple],
        hardening_results: Dict,
    ) -> None:
        round_accepted = hardening_results.get("round_accepted", False)
        classifications = hardening_results.get("classifications", {})

        for i, hid in enumerate(hospital_ids):
            layer1_msg = hardening_results.get(f"layer1_{hid}", "")
            passed_layer1 = layer1_msg == "Passed validation"
            decision = "ACCEPTED" if (passed_layer1 and round_accepted) else "REJECTED"

            cls_info = classifications.get(hid, {})
            attack_idx = cls_info.get("attack_idx")
            attack_type = self.attack_labels[attack_idx] if attack_idx is not None else "unknown"

            g, a, c, t = modality_scores[i]
            old_acc_i, new_acc_i = per_hospital_eval[i]

            evidence = {
                "gradient_magnitude": float(g),
                "accuracy_drop": float(a),
                "similarity": float(c),
                "temporal": float(t),
                "confidence": float(cls_info.get("confidence", 0.0)),
                "attack_type": attack_type,
            }

            self.explainer.generate_explanation(
                hospital_id=hid,
                decision=decision,
                evidence=evidence,
                old_accuracy=old_acc_i,
                new_accuracy=new_acc_i,
            )
