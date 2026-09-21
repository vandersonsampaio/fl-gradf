"""
GRADF's 5-layer hardening pipeline.

Unlike the earlier (placeholder) version, this implementation:
  - operates on parameter vectors (deltas) via an `evaluate_fn` function
    supplied by the caller, instead of requiring `model.clone()`/`.update()`
    (which don't exist on `_LogisticModel`);
  - actually invokes `self.classifier`/`self.selector` in Layer 3, to pick the
    aggregation strategy from the attack type predicted per hospital
    (majority vote when the accepted hospitals disagree — see
    FORMALISMO_MATEMATICO_E_INEDITISMO.md, the Step 6 note).

Known limitation (Layer 2 — legacy DP, active by default): the Laplace noise
is added independently per coordinate, so its total L2 norm scales with
sqrt(dimension) of the parameter vector. For small models (tens of
parameters, as in the plan's original examples) `dp_epsilon=1.0` is
reasonable; for a real model (e.g. MNIST softmax, ~7850 parameters) that same
epsilon makes the noise completely dominate the signal (~30-40x larger),
degrading accuracy even with no attack at all. `dp_epsilon`/`dp_clipping`
need to be recalibrated to the model's dimensionality — see
`GRADFFederatedLearner(dp_epsilon=..., dp_clipping=...)`. This mechanism also
has no real (ε,δ) accounting, mixes L1 sensitivity (Laplace) with an L2 clip,
and adds noise per client before aggregation — see `src.defense.dp_accountant`
for the Gaussian mechanism with RDP accounting and correct per-rule
sensitivity that replaces it when a `dp_mechanism=` is passed to the
constructor (Decision B1, `references/plano_gradf_iclr2027.md`). Kept as the
default when `dp_mechanism=None` so already-reported results
(exp1/exp2/exp4/exp7) are not silently changed.
"""

from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


class HardeningPipeline:
    def __init__(
        self,
        classifier,
        selector,
        magnitude_threshold: float = 5.0,
        accuracy_tolerance: float = 0.02,
        dp_epsilon: float = 50.0,
        dp_clipping: float = 2.0,
        dp_mechanism: Optional[object] = None,
    ) -> None:
        """
        dp_mechanism: if provided (a `src.defense.dp_accountant.
            GaussianDPMechanism`), replaces the legacy Layer 2 (L2 clip +
            per-coordinate Laplace, `dp_epsilon`/`dp_clipping`) with an L2 clip
            + calibrated Gaussian noise with real (ε,δ) accounting via RDP,
            applied ONCE to the aggregated output (Layer 3) instead of per
            client before aggregation — see `GaussianDPMechanism`'s docstring
            for the trust model and the per-rule sensitivity table.
            `dp_epsilon`/`dp_clipping` are ignored when `dp_mechanism` is
            passed.
        """
        self.classifier = classifier
        self.selector = selector
        self.magnitude_threshold = magnitude_threshold
        self.accuracy_tolerance = accuracy_tolerance
        self.dp_epsilon = dp_epsilon
        self.dp_clipping = dp_clipping
        self.dp_mechanism = dp_mechanism

    # ------------------------------------------------------------------
    # Individual layers
    # ------------------------------------------------------------------

    def layer1_input_validation(self, gradient: np.ndarray, threshold: Optional[float] = None):
        """Layer 1: validates bounds (magnitude) and distribution (skewness)."""
        threshold = self.magnitude_threshold if threshold is None else threshold
        magnitude = np.linalg.norm(gradient)

        if magnitude > threshold:
            return False, "Magnitude exceeds threshold"

        skewness = stats.skew(gradient)
        if abs(skewness) > 10:
            return False, "Skewness indicates anomaly"

        return True, "Passed validation"

    def layer2_gradient_sanitization(
        self, gradient: np.ndarray, epsilon: Optional[float] = None, clipping: Optional[float] = None
    ):
        """Layer 2: clipping + Laplace noise (differential privacy)."""
        epsilon = self.dp_epsilon if epsilon is None else epsilon
        clipping = self.dp_clipping if clipping is None else clipping
        clipped = np.clip(gradient, -clipping, clipping)
        noise = np.random.laplace(0, 1 / epsilon, size=gradient.shape)
        return clipped + noise

    def layer3_secure_aggregation(
        self,
        updates: List[np.ndarray],
        strategy_name: str,
        sample_sizes: Optional[List[int]] = None,
        **agg_kwargs,
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Layer 3: dispatches to the aggregation strategy chosen by the
        selector (registered in `src.fl.federated_learner._STRATEGIES`)."""
        from src.fl.federated_learner import _STRATEGIES

        strategy_cls = _STRATEGIES.get(strategy_name, _STRATEGIES["fedavg"])
        strategy = strategy_cls()
        return strategy.aggregate(updates, sample_sizes=sample_sizes, **agg_kwargs)

    def layer4_model_validation(self, old_accuracy: float, new_accuracy: float):
        """Layer 4: rejects the aggregated update if accuracy degrades beyond tolerance."""
        if new_accuracy < old_accuracy - self.accuracy_tolerance:
            return False, f"Accuracy dropped: {old_accuracy:.2%} → {new_accuracy:.2%}"
        return True, f"Accuracy maintained: {new_accuracy:.2%}"

    def layer5_encryption(self, params: np.ndarray, use_he: bool = False) -> str:
        """Layer 5: optional encryption (conceptual stub)."""
        if use_he:
            return "Model encrypted with HE"
        return "Model sent plaintext"

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def full_pipeline(
        self,
        hospital_ids: Sequence[str],
        updates: Sequence[np.ndarray],
        modality_scores: Sequence[np.ndarray],
        evaluate_fn: Callable[[np.ndarray], float],
        sample_sizes: Optional[Sequence[int]] = None,
        server_update: Optional[np.ndarray] = None,
        use_he: bool = False,
        round_context: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """Applies the 5 layers and returns (aggregated_delta_or_None, results).

        Args:
            hospital_ids: identifiers, one per round participant.
            updates: parameter deltas per participant (same order).
            modality_scores: 4-detection-score vector per participant
                (output of `src.detection.modality_recorder.ModalityRecorder.score`).
            evaluate_fn: receives an aggregated delta and returns the
                validation accuracy of the candidate global model
                (global_params + delta). Pass `np.zeros_like(delta)` to
                evaluate the current model (no update).
            sample_sizes: sample sizes per participant (for FedAvg/FedProx).
            server_update: the server's reference delta, required only by the
                'fltrust' strategy. Layer 3 dispatches dynamically to the
                strategy voted by the selector (`selected_strategy`) — which
                can be 'fltrust' even when the learner's top-level
                aggregation is something else — so this kwarg must be
                available whenever the selector has that action in its action
                space (see `RLDefenseSelector.actions`). Without it,
                `FLTrustStrategy.aggregate` raises `ValueError`, is caught by
                the `except` below, and the whole round is discarded.
            round_context: `{"prev_accuracy", "prev_fairness_std", "prev_latency",
                "round_num"}` — the first 3 from the PREVIOUS round,
                `round_num` from the CURRENT round — passed through as
                `**kwargs` to
                `self.selector.select_action(attack_idx, confidence, **round_context)`,
                giving `RLDefenseSelector` the rich 5-dimensional state that
                the original proposal specifies (see its docstring) and
                enabling online decaying ε-greedy exploration (`round_num`).
                Simpler selectors (`FixedActionSelector`/`RandomSelector`/
                `OracleSelector`) accept and ignore these extra kwargs.
        """
        round_context = round_context or {}
        results: Dict = {"round_accepted": False}

        # Layer 1
        accepted_ids = []
        for hid, upd in zip(hospital_ids, updates):
            valid, msg = self.layer1_input_validation(upd)
            results[f"layer1_{hid}"] = msg
            if valid:
                accepted_ids.append(hid)

        if not accepted_ids:
            results["error"] = "All updates rejected at Layer 1"
            return None, results

        idx_by_hid = {hid: i for i, hid in enumerate(hospital_ids)}

        # Layer 2: clip (+ per-client noise, legacy mechanism) or clip only
        # (new Gaussian mechanism — its noise is added later, once, to the
        # aggregated output — see Layer 3/dp_mechanism.privatize_aggregate below).
        if self.dp_mechanism is not None:
            sanitized = {hid: self.dp_mechanism.clip(updates[idx_by_hid[hid]]) for hid in accepted_ids}
            results["layer2"] = f"Clipped (Gaussian mechanism) {len(sanitized)} updates"
        else:
            sanitized = {hid: self.layer2_gradient_sanitization(updates[idx_by_hid[hid]]) for hid in accepted_ids}
            results["layer2"] = f"Sanitized (legacy Laplace) {len(sanitized)} updates"

        # Classification + defense selection per accepted hospital
        classifications: Dict[str, Dict] = {}
        votes = []
        for hid in accepted_ids:
            scores = modality_scores[idx_by_hid[hid]]
            attack_idx, confidence = self.classifier.classify(scores)
            action, q_value = self.selector.select_action(attack_idx, confidence, **round_context)
            classifications[hid] = {
                "attack_idx": attack_idx,
                "confidence": confidence,
                "action": action,
                "q_value": q_value,
            }
            votes.append(action)

        strategy_name = Counter(votes).most_common(1)[0][0] if votes else "fedavg"
        results["selected_strategy"] = strategy_name
        results["classifications"] = classifications

        # Layer 3
        sanitized_updates = [sanitized[hid] for hid in accepted_ids]
        sample_sizes_accepted = None
        if sample_sizes is not None:
            sample_sizes_accepted = [sample_sizes[idx_by_hid[hid]] for hid in accepted_ids]

        try:
            agg_delta, agg_metadata = self.layer3_secure_aggregation(
                sanitized_updates, strategy_name,
                sample_sizes=sample_sizes_accepted, server_update=server_update,
            )
        except Exception as exc:  # strategy misconfigured for this batch (e.g. FLTrust without server_update)
            results["layer3_error"] = str(exc)
            return None, results

        results["layer3"] = f"Aggregated {len(sanitized_updates)}/{len(hospital_ids)} with '{strategy_name}'"
        if agg_metadata:
            results["layer3_metadata"] = agg_metadata

        # New Gaussian mechanism: noise added ONCE to the aggregated output
        # (not per client, like the legacy Layer 2) — Layer 4 below validates
        # the ALREADY noised delta, which is what actually gets applied to the
        # global model.
        if self.dp_mechanism is not None:
            agg_delta = self.dp_mechanism.privatize_aggregate(
                agg_delta, strategy_name, weights=sample_sizes_accepted,
            )
            results["dp_budget"] = self.dp_mechanism.budget_status()

        # Layer 4
        old_accuracy = evaluate_fn(np.zeros_like(agg_delta))
        new_accuracy = evaluate_fn(agg_delta)
        valid, msg = self.layer4_model_validation(old_accuracy, new_accuracy)
        results["layer4"] = msg
        results["old_accuracy"] = old_accuracy
        results["new_accuracy"] = new_accuracy

        if not valid:
            return None, results  # rejects the aggregated update; caller keeps the previous model

        # Layer 5
        results["layer5"] = self.layer5_encryption(agg_delta, use_he=use_he)
        results["round_accepted"] = True

        return agg_delta, results
