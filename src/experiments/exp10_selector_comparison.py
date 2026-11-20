"""
Experiment 10: competing adaptive selectors versus GRADF.

Evaluates adaptive aggregation mechanisms on the same dominance grid
with ``root=100``, a seven-rule aggregation arsenal, and identical
random seeds.

The experiment tests whether the performance gap between adaptive
selection and the best fixed aggregation rule is specific to GRADF's
DQN selector or persists across alternative adaptive mechanisms.

Compared systems:

- Random selector:
  selects one of the seven aggregation rules uniformly at random in
  each round. Reuses ``exp1_baseline.RandomSelector``.

- Oracle selector:
  selects the best fixed rule for the true attack type of each round,
  without using an attack detector. Reuses
  ``exp1_baseline.GroundTruthOracleLearner`` and serves as a
  detection-decoupled upper reference.

- GRADF:
  uses the corrected five-dimensional DQN selector.

- FedStrategist:
  uses the LinUCB contextual bandit proposed by Haque et al. (2025),
  adapted from the paper's three-rule arsenal to the full seven-rule
  arsenal used in this experiment.

- AdaAggRL:
  uses the adaptive aggregation mechanism proposed by Wang et al.
  (AAAI 2025), including gradient-inversion distribution learning,
  MMD-based environmental cues, and TD3-based continuous aggregation
  weights.

FedStrategist is evaluated with two detection configurations:

- Shared detection:
  uses GRADF's detection pipeline
  (``ModalityRecorder`` and ``RLAttackClassifier``). This isolates the
  selection policy because the detection signal is held constant.

- Own detection:
  uses FedStrategist's original three-dimensional update-geometry
  diagnostic. A performance difference relative to shared detection
  therefore reflects the effect of its detection mechanism as well as
  its selection policy.

- AdaAggRL detector + discrete selector (``adaaggrl_detector``,
  ``DiscreteWithAdaAggRLDetectorLearner``; Front 1, Stage 1.1):
  the inverse coupling of FedStrategist's shared-detection variant. It
  feeds AdaAggRL's own detector (gradient-inversion reconstruction and the
  four MMD-based cues, reused from ``src/defense/adaaggrl_agent.py``) into
  the same ``LinUCBAgent`` over the seven-rule arsenal, which picks one
  rule per round and aggregates the raw parameter deltas through it. If
  discrete selection catches up once given AdaAggRL's detector, the
  bottleneck was detection quality; if not, discreteness itself is
  implicated.

AdaAggRL has only one configuration because its gradient-inversion and
MMD-based cues are intrinsic to its aggregation mechanism and constitute
its detection signal.

Random and Oracle do not require an external detection signal and are
computed once for reuse across FedStrategist variants.

The best-fixed-rule reference is loaded from the existing
``exp9_dominance_grid_10seeds_ALL_root100_summary.csv`` results rather
than recomputing the fixed-rule baselines.

Example:

    python -m src.experiments.exp10_selector_comparison \
        --variant b --seeds 42 43 44

    python -m src.experiments.exp10_selector_comparison \
        --variant b --systems fedstrategist random \
        --alphas 0.5 --attack_types sign_flipping
"""

import argparse
from fractions import Fraction
import os
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import src.defense.aggregation_methods  # noqa: F401  registers median/trimmed_mean/krum/etc.
from src.defense.adaaggrl_agent import (
    AdaAggRLAgent,
    PretrainedCNNFeatureExtractor,
    RandomCNNFeatureExtractor,
    compute_weights_and_penalty,
    cue_similarity,
    mmd_rbf,
    reconstruct_client_distribution,
    reconstruct_client_distribution_cnn,
)
from src.defense.aggregation_methods import KrumStrategy
from src.defense.hardening import legacy_dp_epsilon_for
from src.defense.fedstrategist_selector import DiagnosticStateVector, LinUCBAgent
from src.defense.rl_selector import RLDefenseSelector
from src.defense.tars_selector import cross_entropy_loss
from src.detection.modality_recorder import ModalityRecorder
from src.experiments.exp1_baseline import (
    GroundTruthClassifierStub,
    GroundTruthOracleLearner,
    OracleSelector,
    RandomSelector,
    _derive_oracle_map,
    train_gradf_models,
)
from src.experiments.exp9_dominance_grid import BLIND_ATTACKS, INFORMED_ATTACKS, _split_name, round_accuracies
from src.fl.attacked_learner import AttackedFederatedLearner, compute_param_updates_auto
from src.fl.federated_learner import RoundResult, _CNNModel, _STRATEGIES
from src.fl.gradf_learner import GRADFFederatedLearner
from src.utils.data_loader import generate_selector_experiences, load_dataset_participants
from src.utils.logger import get_logger
from src.utils.stats import paired_significance, run_over_seeds, summarize

# Input shape per dataset — needed by AdaAggRL's RandomCNNFeatureExtractor
# (gradient inversion reconstructs flat feature vectors; the extractor
# needs the original image shape to reshape them back for its Conv2D
# layers).
DATASET_INPUT_SHAPE = {"mnist": (28, 28, 1), "cifar10": (32, 32, 3)}

logger = get_logger(__name__)

# Validity condition 2 (plan doc): SAME 7-rule arsenal as GRADF for every
# selector compared here — not FedStrategist's original 3-rule arsenal
# (fedavg/median/krum) used in `src/experiments/exp4_adaptive.py`.
FULL_ARSENAL = ["fedavg", "fedprox", "median", "trimmed_mean", "fltrust", "clustering", "krum"]

# Heuristic per-rule cost, extending the FedStrategist paper's cost design
# (which only fixes 3 values: fedavg=0.1, median=0.4, krum=0.8) to the full
# 7-rule arsenal required here, ordered by the same increasing
# robustness/computational-complexity intuition the paper uses. NOT from the
# paper — a documented extension, and unused by AdaAggRL (which has no cost
# term at all in this experiment; see `run_selector_comparison_grid`'s
# docstring on reward design).
ARSENAL_COST = {
    "fedavg": 0.1, "fedprox": 0.2, "median": 0.4, "trimmed_mean": 0.4,
    "fltrust": 0.5, "clustering": 0.6, "krum": 0.8,
}


def _make_arsenal_strategy(name: str, n_byzantine: int):
    return KrumStrategy(n_byzantine=n_byzantine) if name == "krum" else _STRATEGIES[name]()


def _shared_detection_state(
    recorder: ModalityRecorder,
    classifier,
    hospital_ids: List[str],
    updates: List[np.ndarray],
    old_acc: float,
    new_accs: List[float],
    n_labels: int,
) -> np.ndarray:
    """Variant (b): the SAME detection signal GRADF itself consumes —
    per-hospital `ModalityRecorder.score()` + the pretrained
    `RLAttackClassifier.classify()` — collapsed into one round-level 4-dim
    summary: [majority attack_idx (normalized), mean confidence, mean
    anomaly score across modalities/hospitals, std of that anomaly score].
    """
    scores_list, attack_idxs, confidences = [], [], []
    for i, hid in enumerate(hospital_ids):
        peers = [u for j, u in enumerate(updates) if j != i]
        scores = recorder.score(hid, updates[i], peers, old_acc, new_accs[i])
        scores_list.append(scores)
        attack_idx, confidence = classifier.classify(scores)
        attack_idxs.append(attack_idx)
        confidences.append(confidence)
    scores_arr = np.stack(scores_list, axis=0)
    mean_scores = scores_arr.mean(axis=0)
    mean_confidence = float(np.mean(confidences))
    vals, counts = np.unique(attack_idxs, return_counts=True)
    majority_idx = int(vals[np.argmax(counts)])
    return np.array([
        majority_idx / max(n_labels - 1, 1),
        mean_confidence,
        float(mean_scores.mean()),
        float(mean_scores.std()),
    ])


class FedStrategistGridLearner(AttackedFederatedLearner):
    """FedStrategist widened to this experiment's 7-rule arsenal, with a
    pluggable detection source (`variant='a'`: own 3-dim diagnostic;
    `variant='b'`: GRADF's shared detection layer, via `shared_classifier`).
    No cost term in the reward here (unlike `exp4_adaptive.FedStrategistLearner`,
    which keeps the paper's own lambda_cost/C_j for paper-fidelity reasons) —
    this experiment isolates the selection POLICY, so reward is plain
    delta-accuracy for every learned system, matching Random/Oracle/GRADF's
    own reward mechanics."""

    def __init__(
        self, *args, variant: str = "a", shared_classifier=None,
        n_labels: int = 2, alpha: float = 1.5, seed: int = 42, **kwargs,
    ):
        kwargs.setdefault("aggregation", FULL_ARSENAL[0])
        kwargs.setdefault("seed", seed)
        super().__init__(*args, **kwargs)
        self.variant = variant
        self.n_labels = n_labels
        self.bandit = LinUCBAgent(FULL_ARSENAL, alpha=alpha, context_dim=3, seed=seed)
        if variant == "b":
            if shared_classifier is None:
                raise ValueError("variant='b' requires shared_classifier")
            self.shared_classifier = shared_classifier
            self.recorder = ModalityRecorder()

    def _run_round(self, round_num, participants, root_data):
        active = self._is_active(round_num)
        param_updates, _is_byz_list = compute_param_updates_auto(self, participants, active, root_data)
        n_feat = participants[0].n_features

        eval_data = root_data or {"X": participants[0].X_test, "y": participants[0].y_test}
        old_model = self._make_model(n_feat)
        old_model.set_params(self._global_params)  # type: ignore[arg-type]
        old_acc = old_model.accuracy(eval_data["X"], eval_data["y"])

        if self.variant == "a":
            state = DiagnosticStateVector.compute(param_updates)
        else:
            new_accs = []
            for u in param_updates:
                cand = self._make_model(n_feat)
                cand.set_params(self._global_params + u)  # type: ignore[operator]
                new_accs.append(cand.accuracy(eval_data["X"], eval_data["y"]))
            hospital_ids = [p.id for p in participants]
            state4 = _shared_detection_state(
                self.recorder, self.shared_classifier, hospital_ids,
                param_updates, old_acc, new_accs, self.n_labels,
            )
            state = state4[:3]

        strategy_name = self.bandit.select_action(tuple(state))

        server_root = root_data or self._carve_root(participants[0])
        server_update = self._fit_server(self._make_model(n_feat), server_root, participants)
        sample_sizes = [p.n_train for p in participants]

        n_byz = sum(1 for b in _is_byz_list if b)
        strategy = _make_arsenal_strategy(strategy_name, n_byz)
        agg_delta, _ = strategy.aggregate(param_updates, sample_sizes=sample_sizes, server_update=server_update)
        self._global_params = self._global_step(agg_delta)

        new_model = self._make_model(n_feat)
        new_model.set_params(self._global_params)
        new_acc = new_model.accuracy(eval_data["X"], eval_data["y"])

        reward = new_acc - old_acc
        self.bandit.update(tuple(state), strategy_name, reward)

        per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
        global_acc = float(np.mean(list(per_acc.values())))
        return RoundResult(round_num, global_acc, per_acc, trust_scores=None, n_accepted=None)


class AdaAggRLGridLearner(AttackedFederatedLearner):
    """AdaAggRL (Wang, Zhang, Wen, Qiu & Guo, AAAI 2025) — see
    `src/defense/adaaggrl_agent.py` for the full reproduction (gradient-
    inversion distribution learning + MMD-based environmental cues + TD3
    over a [0,1]^5 action + exponential malicious-behavior penalty) and its
    documented deviations/instantiation choices.

    Mechanically different from every other learner in this file:
    aggregates clients' FULL PARAMETERS (theta_k^{t+1} = global_params +
    update_k) directly, weighted by the agent's decision — not a delta
    blended from other registered aggregation strategies, and not gated by
    the 7-rule `FULL_ARSENAL` at all (AdaAggRL has no notion of candidate
    rules; its "action" only ever weights clients, never picks/blends
    aggregation FUNCTIONS). No `variant` parameter: see the module
    docstring for why a "shared GRADF detection" mode doesn't apply here.
    With `model_type='logistic'` the inversion uses `_LogisticModel`'s flat
    W/b packing; with `model_type='cnn'` (P1 CIFAR-10 campaign) it runs on
    the CNN itself (`reconstruct_client_distribution_cnn`).

    `feature_extractor`: 'random' (default, reproduces the original floor
    result — `RandomCNNFeatureExtractor`, frozen random weights) or
    'pretrained' (`PretrainedCNNFeatureExtractor`, a real trained-then-
    frozen CNN — Step Zero, `scripts/pretrain_adaaggrl_extractor.py`).
    `feature_dim` defaults differ between the two (16 vs. 32) to match each
    extractor's own natural width; pass explicitly to override.
    """

    def __init__(
        self, *args, input_shape: tuple, seed: int = 42,
        num_images: int = 16, max_iters: int = 30, feature_dim: Optional[int] = None,
        lam: float = 2.0, feature_extractor: str = "random",
        feature_extractor_path: Optional[str] = None, dataset: str = "mnist",
        **kwargs,
    ):
        kwargs.setdefault("aggregation", "fedavg")  # placeholder to satisfy FederatedLearner's validation; unused
        kwargs.setdefault("seed", seed)
        if kwargs.get("model_type") == "cnn":
            kwargs.setdefault("input_shape", input_shape)
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.num_images = num_images
        self.max_iters = max_iters
        self.lam = lam
        self.agent = AdaAggRLAgent(state_dim=4, seed=seed)
        if feature_extractor == "random":
            self.feature_extractor = RandomCNNFeatureExtractor(
                input_shape, feature_dim=feature_dim or 16, seed=seed,
            )
        elif feature_extractor == "pretrained":
            self.feature_extractor = PretrainedCNNFeatureExtractor(
                input_shape, feature_dim=feature_dim or 32,
                weights_path=feature_extractor_path, dataset=dataset,
            )
        else:
            raise ValueError(f"Unknown feature_extractor '{feature_extractor}'; expected 'random' or 'pretrained'")
        self._v_history: Dict[str, np.ndarray] = {}
        self._h: Optional[np.ndarray] = None

    def _run_round(self, round_num, participants, root_data):
        active = self._is_active(round_num)
        param_updates, _is_byz_list = compute_param_updates_auto(self, participants, active, root_data)
        n_feat = participants[0].n_features
        K = 1 if self.n_classes == 2 else self.n_classes

        old_model = self._make_model(n_feat)
        old_model.set_params(self._global_params)  # type: ignore[arg-type]
        eval_data = root_data or {"X": participants[0].X_test, "y": participants[0].y_test}
        old_loss = cross_entropy_loss(old_model, eval_data["X"], eval_data["y"])

        theta_list = [self._global_params + u for u in param_updates]  # type: ignore[operator]  # theta_k^{t+1}

        v_current: Dict[str, np.ndarray] = {}
        s_r_list = []
        inv_model = self._instantiate_model(n_feat) if self.model_type == "cnn" else None
        for p, theta_k, u in zip(participants, theta_list, param_updates):
            if inv_model is not None:
                inv_model.set_params(theta_k)
                D_rec, S_R = reconstruct_client_distribution_cnn(
                    inv_model._model, u, self.local_lr, self.input_shape, self.n_classes,
                    num_images=self.num_images, max_iters=self.max_iters, seed=round_num,
                )
            else:
                W = theta_k[: n_feat * K].reshape(n_feat, K).astype(np.float32)
                b = theta_k[n_feat * K:].astype(np.float32)
                D_rec, S_R = reconstruct_client_distribution(
                    W, b, u, self.local_lr, n_feat, K,
                    num_images=self.num_images, max_iters=self.max_iters, seed=round_num,
                )
            v_current[p.id] = self.feature_extractor.extract(D_rec)  # (num_images, feature_dim)
            s_r_list.append(S_R)

        v_g = np.concatenate(list(v_current.values()), axis=0)  # pooled global distribution, see docstring

        if self._h is None:
            self._h = np.zeros(len(participants))

        state_rows = []
        for p, S_R in zip(participants, s_r_list):
            v_cur = v_current[p.id]
            v_hist = self._v_history.get(p.id, v_cur)  # bootstrap: no history yet on round 1
            S_cl = cue_similarity(mmd_rbf(v_cur, v_hist))
            S_cg = cue_similarity(mmd_rbf(v_cur, v_g))
            S_lg = cue_similarity(mmd_rbf(v_hist, v_g))
            state_rows.append([S_R, S_cl, S_cg, S_lg])
        state_matrix = np.array(state_rows)

        mean_state = state_matrix.mean(axis=0)  # fixed-size input to the policy (see module docstring, point 3b)
        action = self.agent.select_action(mean_state)

        weights, new_h, _delta = compute_weights_and_penalty(state_matrix, action, self._h, lam=self.lam)
        self._h = new_h

        total_w = float(weights.sum())
        if total_w < 1e-8:
            norm_weights = np.ones(len(participants)) / len(participants)  # degenerate: every client flagged
        else:
            norm_weights = weights / total_w  # documented renormalization, see adaaggrl_agent.py docstring

        aggregated = sum(w * theta for w, theta in zip(norm_weights, theta_list))
        if self.global_lr == 1.0:  # no subtract-and-add: keeps earlier runs bit for bit
            self._global_params = aggregated
        else:
            self._global_params = self._global_step(aggregated - self._global_params)

        for p in participants:
            self._v_history[p.id] = v_current[p.id]

        new_model = self._make_model(n_feat)
        new_model.set_params(self._global_params)
        new_loss = cross_entropy_loss(new_model, eval_data["X"], eval_data["y"])
        reward = old_loss - new_loss  # paper's r = f(theta^t) - f(theta^{t+1})
        self.agent.train_step(mean_state, action, reward, mean_state)  # single-step episode

        per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
        global_acc = float(np.mean(list(per_acc.values())))
        return RoundResult(
            round_num, global_acc, per_acc, trust_scores=None,
            n_accepted=int((weights > 0).sum()),
        )


class DiscreteWithAdaAggRLDetectorLearner(AttackedFederatedLearner):
    """Front 1, Stage 1.1 — the
    shared-detector experiment. Reuses AdaAggRL's REAL detector
    (gradient-inversion reconstruction + the 4 MMD-based environmental cues
    S_R/S_cl/S_cg/S_lg, `src/defense/adaaggrl_agent.py`) exactly as
    `AdaAggRLGridLearner` computes it, but hands the round-level summary to a
    clean discrete selector — a `LinUCBAgent` contextual bandit over the same
    `FULL_ARSENAL` of 7 aggregation rules `FedStrategistGridLearner` already
    uses — instead of AdaAggRL's own continuous [0,1]^5 TD3 weighting of full
    client parameters. Aggregation then dispatches through the chosen rule on
    the round's parameter DELTAS (the same mechanic every other discrete
    selector in this file uses), so the only thing borrowed from AdaAggRL is
    its detection signal, not its aggregation mechanic — keeping "detector"
    and "selection family" cleanly separated, per the study design.

    `feature_extractor`: 'pretrained' (default here — Step Zero's real,
    trained-then-frozen CNN; the Step Zero gate found the pretrained
    extractor does not help AdaAggRL itself, so it is used here as the more
    faithful choice, not because it's expected to be decisive) or 'random'.
    Only supports `model_type='logistic'` (gradient inversion needs
    `_LogisticModel`'s flat W/b packing), same restriction as
    `AdaAggRLGridLearner`.
    """

    def __init__(
        self, *args, input_shape: tuple, seed: int = 42,
        num_images: int = 16, max_iters: int = 30, feature_dim: Optional[int] = None,
        feature_extractor: str = "pretrained", feature_extractor_path: Optional[str] = None,
        dataset: str = "mnist", alpha: float = 1.5, **kwargs,
    ):
        kwargs.setdefault("aggregation", FULL_ARSENAL[0])
        kwargs.setdefault("seed", seed)
        super().__init__(*args, **kwargs)
        if self.model_type != "logistic":
            raise ValueError("DiscreteWithAdaAggRLDetectorLearner only supports model_type='logistic'")
        self.input_shape = input_shape
        self.num_images = num_images
        self.max_iters = max_iters
        self.bandit = LinUCBAgent(FULL_ARSENAL, alpha=alpha, context_dim=4, seed=seed)
        if feature_extractor == "random":
            self.feature_extractor = RandomCNNFeatureExtractor(
                input_shape, feature_dim=feature_dim or 16, seed=seed,
            )
        elif feature_extractor == "pretrained":
            self.feature_extractor = PretrainedCNNFeatureExtractor(
                input_shape, feature_dim=feature_dim or 32,
                weights_path=feature_extractor_path, dataset=dataset,
            )
        else:
            raise ValueError(f"Unknown feature_extractor '{feature_extractor}'; expected 'random' or 'pretrained'")
        self._v_history: Dict[str, np.ndarray] = {}

    def _adaaggrl_state(
        self, participants, theta_list: List[np.ndarray], param_updates: List[np.ndarray],
        n_feat: int, K: int, round_num: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Same reconstruction + MMD-cue computation as
        `AdaAggRLGridLearner._run_round`, factored out so both learners stay
        byte-for-byte comparable on the detection side."""
        v_current: Dict[str, np.ndarray] = {}
        s_r_list = []
        inv_model = self._instantiate_model(n_feat) if self.model_type == "cnn" else None
        for p, theta_k, u in zip(participants, theta_list, param_updates):
            if inv_model is not None:
                inv_model.set_params(theta_k)
                D_rec, S_R = reconstruct_client_distribution_cnn(
                    inv_model._model, u, self.local_lr, self.input_shape, self.n_classes,
                    num_images=self.num_images, max_iters=self.max_iters, seed=round_num,
                )
            else:
                W = theta_k[: n_feat * K].reshape(n_feat, K).astype(np.float32)
                b = theta_k[n_feat * K:].astype(np.float32)
                D_rec, S_R = reconstruct_client_distribution(
                    W, b, u, self.local_lr, n_feat, K,
                    num_images=self.num_images, max_iters=self.max_iters, seed=round_num,
                )
            v_current[p.id] = self.feature_extractor.extract(D_rec)
            s_r_list.append(S_R)

        v_g = np.concatenate(list(v_current.values()), axis=0)
        state_rows = []
        for p, S_R in zip(participants, s_r_list):
            v_cur = v_current[p.id]
            v_hist = self._v_history.get(p.id, v_cur)
            S_cl = cue_similarity(mmd_rbf(v_cur, v_hist))
            S_cg = cue_similarity(mmd_rbf(v_cur, v_g))
            S_lg = cue_similarity(mmd_rbf(v_hist, v_g))
            state_rows.append([S_R, S_cl, S_cg, S_lg])
        state_matrix = np.array(state_rows)

        for p in participants:
            self._v_history[p.id] = v_current[p.id]

        return state_matrix, state_matrix.mean(axis=0)

    def _run_round(self, round_num, participants, root_data):
        active = self._is_active(round_num)
        # compute_param_updates_auto: dispatches to the informed-attacker path
        # for INFORMED_ATTACK_TYPES instead of silently degrading them — see
        # src/fl/attacked_learner.py's docstring (blind attack paths silently
        # degraded the informed attack types before this dispatch existed).
        param_updates, _is_byz_list = compute_param_updates_auto(self, participants, active, root_data)
        n_feat = participants[0].n_features
        K = 1 if self.n_classes == 2 else self.n_classes

        eval_data = root_data or {"X": participants[0].X_test, "y": participants[0].y_test}
        old_model = self._make_model(n_feat)
        old_model.set_params(self._global_params)  # type: ignore[arg-type]
        old_acc = old_model.accuracy(eval_data["X"], eval_data["y"])

        theta_list = [self._global_params + u for u in param_updates]  # type: ignore[operator]  # theta_k^{t+1}
        _state_matrix, mean_state = self._adaaggrl_state(
            participants, theta_list, param_updates, n_feat, K, round_num,
        )

        strategy_name = self.bandit.select_action(tuple(mean_state))

        server_root = root_data or self._carve_root(participants[0])
        server_update = self._fit_server(self._make_model(n_feat), server_root, participants)
        sample_sizes = [p.n_train for p in participants]

        n_byz = sum(1 for b in _is_byz_list if b)
        strategy = _make_arsenal_strategy(strategy_name, n_byz)
        agg_delta, _ = strategy.aggregate(param_updates, sample_sizes=sample_sizes, server_update=server_update)
        self._global_params = self._global_step(agg_delta)

        new_model = self._make_model(n_feat)
        new_model.set_params(self._global_params)
        new_acc = new_model.accuracy(eval_data["X"], eval_data["y"])

        reward = new_acc - old_acc
        self.bandit.update(tuple(mean_state), strategy_name, reward)

        per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
        global_acc = float(np.mean(list(per_acc.values())))
        return RoundResult(round_num, global_acc, per_acc, trust_scores=None, n_accepted=None)


def run_selector_comparison_grid(
    dataset: str = "mnist",
    alphas: Optional[List[float]] = None,
    attack_types: Optional[List[str]] = None,
    n_clients: int = 10,
    n_rounds: int = 15,
    byzantine_fraction: float = 0.2,
    n_classes: int = 10,
    seed: int = 42,
    root_size: int = 100,
    systems: Optional[List[str]] = None,
    variant: str = "b",
    adaaggrl_feature_extractor: str = "random",
    model_type: str = "logistic",
    local_epochs: int = 5,
    local_lr: float = 0.01,
    seed_rng: bool = False,
    server_steps="native",
    global_lr: float = 1.0,
    magnitude_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """`root_size=100` by default — validity condition 1 of the plan doc.

    `adaaggrl_feature_extractor`: 'random' (default, original floor) or
    'pretrained' (Step Zero real extractor) — see
    `AdaAggRLGridLearner`'s docstring.

    P1 CIFAR-10 campaign: `model_type='cnn'` runs every system (and GRADF's
    pretraining) on the small CNN, with GRADF's legacy Layer-2 epsilon
    rescaled to the CNN's parameter count (`legacy_dp_epsilon_for`).
    `seed_rng` seeds each system run from (seed, alpha, attack, system), as
    in `exp9_dominance_grid.run_dominance_grid`. Defaults keep the
    historical behavior."""
    alphas = alphas or [0.5, 0.1, 0.05]
    attack_types = attack_types or (INFORMED_ATTACKS + BLIND_ATTACKS)
    systems = systems or ["random", "oracle", "gradf", "fedstrategist", "adaaggrl"]
    byzantine_ids = list(range(max(1, int(n_clients * byzantine_fraction))))

    needs_pretrain = bool({"gradf", "oracle"} & set(systems)) or (
        variant == "b" and "fedstrategist" in systems
    )

    model_kwargs = {"local_epochs": local_epochs, "local_lr": local_lr, "server_steps": server_steps,
                    "global_lr": global_lr}
    gradf_kwargs = dict(model_kwargs)
    if model_type == "cnn":
        input_shape = DATASET_INPUT_SHAPE[dataset]
        model_kwargs.update(model_type="cnn", input_shape=input_shape)
        n_params = _CNNModel(int(np.prod(input_shape)), input_shape=input_shape,
                             n_classes=n_classes).params.size
        gradf_kwargs = dict(model_kwargs, dp_epsilon=legacy_dp_epsilon_for(n_params))
    if magnitude_threshold is not None:
        gradf_kwargs["magnitude_threshold"] = magnitude_threshold
        if seed_rng:
            import tensorflow as tf
            tf.config.experimental.enable_op_determinism()

    def _seeded(alpha, attack_type, system):
        if not seed_rng:
            return {}
        run_seed = zlib.crc32(f"{seed}|{alpha}|{attack_type}|{system}".encode())
        np.random.seed(run_seed)
        if model_type == "cnn":
            import tensorflow as tf
            tf.keras.utils.set_random_seed(run_seed)
        return {"server_rng_seed": run_seed + 1}

    rows = []
    for alpha in alphas:
        split = _split_name(alpha)
        participants, root_data = load_dataset_participants(
            dataset, split, n_clients, root_size=root_size, root_seed=seed,
        )
        for attack_type in attack_types:
            attack_labels = ["none", attack_type]

            classifier = selector = oracle_map = None
            if needs_pretrain:
                _seeded(alpha, attack_type, "pretrain")
                classifier, selector, _labels, oracle_map = train_gradf_models(
                    participants, attack_type, byzantine_fraction, n_classes, n_rounds=8, seed=seed,
                    model_kwargs=model_kwargs,
                )

            def _add(system_label: str, sys_variant: str, acc: tuple) -> None:
                rows.append({
                    "alpha": alpha, "attack_type": attack_type, "system": system_label,
                    "variant": sys_variant, "accuracy": acc[0], "accuracy_trailing5": acc[1],
                })

            if "random" in systems:
                learner = GRADFFederatedLearner(
                    aggregation="fedavg", classifier=GroundTruthClassifierStub(),
                    selector=RandomSelector(RLDefenseSelector().actions, seed=seed),
                    attack_labels=attack_labels, n_rounds=n_rounds, n_classes=n_classes,
                    attack_type=attack_type, byzantine_ids=byzantine_ids, seed=seed,
                    **gradf_kwargs, **_seeded(alpha, attack_type, "random"),
                )
                acc = round_accuracies(learner.train(participants, root_data=root_data, verbose=False))
                _add("Random", "n/a", acc)

            if "oracle" in systems:
                learner = GroundTruthOracleLearner(
                    aggregation="fedavg", classifier=GroundTruthClassifierStub(),
                    selector=OracleSelector(oracle_map), attack_labels=attack_labels,
                    n_rounds=n_rounds, n_classes=n_classes, attack_type=attack_type,
                    byzantine_ids=byzantine_ids, seed=seed,
                    **gradf_kwargs, **_seeded(alpha, attack_type, "oracle"),
                )
                acc = round_accuracies(learner.train(participants, root_data=root_data, verbose=False))
                _add("Oracle (decoupled)", "n/a", acc)

            if "gradf" in systems:
                learner = GRADFFederatedLearner(
                    aggregation="fedavg", classifier=classifier, selector=selector,
                    attack_labels=attack_labels, n_rounds=n_rounds, n_classes=n_classes,
                    attack_type=attack_type, byzantine_ids=byzantine_ids, seed=seed,
                    **gradf_kwargs, **_seeded(alpha, attack_type, "gradf"),
                )
                acc = round_accuracies(learner.train(participants, root_data=root_data, verbose=False))
                _add("GRADF", "shared (own)", acc)

            if "fedstrategist" in systems:
                learner = FedStrategistGridLearner(
                    n_rounds=n_rounds, n_classes=n_classes, attack_type=attack_type,
                    byzantine_ids=byzantine_ids, seed=seed, variant=variant,
                    shared_classifier=classifier, n_labels=len(attack_labels),
                    **model_kwargs, **_seeded(alpha, attack_type, "fedstrategist"),
                )
                acc = round_accuracies(learner.train(participants, root_data=root_data, verbose=False))
                _add("FedStrategist", variant, acc)

            if "adaaggrl" in systems:
                learner = AdaAggRLGridLearner(
                    n_rounds=n_rounds, n_classes=n_classes, attack_type=attack_type,
                    byzantine_ids=byzantine_ids, seed=seed,
                    feature_extractor=adaaggrl_feature_extractor, dataset=dataset,
                    **{**model_kwargs, "input_shape": DATASET_INPUT_SHAPE[dataset]},
                    **_seeded(alpha, attack_type, "adaaggrl"),
                )
                # RandomCNNFeatureExtractor resets the global seeds; seed the run again.
                _seeded(alpha, attack_type, "adaaggrl")
                acc = round_accuracies(learner.train(participants, root_data=root_data, verbose=False))
                _add("AdaAggRL", f"own (gradient-inversion, {adaaggrl_feature_extractor} extractor)", acc)

            if "adaaggrl_detector" in systems:
                learner = DiscreteWithAdaAggRLDetectorLearner(
                    n_rounds=n_rounds, n_classes=n_classes, attack_type=attack_type,
                    byzantine_ids=byzantine_ids, seed=seed,
                    feature_extractor=adaaggrl_feature_extractor, dataset=dataset,
                    **{**model_kwargs, "input_shape": DATASET_INPUT_SHAPE[dataset]},
                    **_seeded(alpha, attack_type, "adaaggrl_detector"),
                )
                acc = round_accuracies(learner.train(participants, root_data=root_data, verbose=False))
                _add(
                    "Discrete+AdaAggRL-detector",
                    f"LinUCB/7-rule, AdaAggRL detector ({adaaggrl_feature_extractor} extractor)",
                    acc,
                )

            logger.info("seed=%d alpha=%s attack=%s variant=%s done", seed, alpha, attack_type, variant)

    return pd.DataFrame(rows)


def _load_best_fixed_rule_reference(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        logger.warning("Best-fixed-rule reference not found at %s; skipping that column.", path)
        return None
    ref = pd.read_csv(path)
    best = ref.loc[ref.groupby(["alpha", "attack_type"])["accuracy_mean"].idxmax()]
    return best[["alpha", "attack_type", "strategy", "accuracy_mean"]].rename(
        columns={"strategy": "best_fixed_rule", "accuracy_mean": "best_fixed_rule_accuracy"}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.1, 0.05])
    parser.add_argument("--attack_types", nargs="+", default=None)
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--n_rounds", type=int, default=15)
    parser.add_argument("--byzantine_fraction", type=float, default=0.2)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--root_size", type=int, default=100)
    parser.add_argument(
        "--systems", nargs="+", default=["random", "oracle", "gradf", "fedstrategist", "adaaggrl"],
        choices=["random", "oracle", "gradf", "fedstrategist", "adaaggrl", "adaaggrl_detector"],
    )
    parser.add_argument("--variant", default="b", choices=["a", "b"])
    parser.add_argument(
        "--adaaggrl_feature_extractor", default="random", choices=["random", "pretrained"],
        help="'random' reproduces the original floor result; 'pretrained' uses the Step Zero "
             "real extractor (needs `python scripts/pretrain_adaaggrl_extractor.py` run first).",
    )
    parser.add_argument("--tag", default="", help="Optional suffix for output filenames.")
    parser.add_argument("--model", default="logistic", choices=["logistic", "cnn"])
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--local_lr", type=float, default=0.01)
    parser.add_argument("--server_steps", default="native",
                        help="FLTrust server-fit steps: native (epochs over the root), "
                             "median (step parity with the clients) or an int; see "
                             "FederatedLearner. Non-native runs get '_steps<value>' in "
                             "the output names and a server_steps column.")
    parser.add_argument("--global_lr", type=lambda s: float(Fraction(s)), default=1.0,
                        help="Global rate, as a number or a fraction such as 1/3. Values "
                             "other than 1 get '_lr<den>' in the output names and a "
                             "global_lr column.")
    parser.add_argument("--seed_rng", action="store_true",
                        help="Reproducible runs (see run_selector_comparison_grid).")
    parser.add_argument("--out_dir", default="results/tables")
    args = parser.parse_args()

    raw = run_over_seeds(
        run_selector_comparison_grid, seeds=args.seeds,
        dataset=args.dataset, alphas=args.alphas, attack_types=args.attack_types,
        n_clients=args.n_clients, n_rounds=args.n_rounds,
        byzantine_fraction=args.byzantine_fraction, n_classes=args.n_classes,
        root_size=args.root_size, systems=args.systems, variant=args.variant,
        adaaggrl_feature_extractor=args.adaaggrl_feature_extractor,
        model_type=args.model, local_epochs=args.local_epochs, local_lr=args.local_lr,
        seed_rng=args.seed_rng,
        server_steps=args.server_steps, global_lr=args.global_lr,
    )
    if args.server_steps != "native":
        raw["server_steps"] = args.server_steps
    if args.global_lr != 1.0:
        raw["global_lr"] = args.global_lr

    suffix = f"_variant{args.variant}" + (f"_{args.tag}" if args.tag else "")
    if args.model == "cnn":
        suffix += f"_{args.dataset}_cnn"
    if args.seed_rng:
        suffix += "_seeded"
    if args.server_steps != "native":
        suffix += f"_steps{args.server_steps}"
    if args.global_lr != 1.0:
        suffix += f"_lr{round(1 / args.global_lr)}"
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    raw.to_csv(f"{out}/exp10_selector_comparison{suffix}_raw.csv", index=False)
    print("=== raw (per seed) ===")
    print(raw.to_string(index=False))

    summary = summarize(raw, ["alpha", "attack_type", "system"], "accuracy")
    summary = summary.rename(columns={"mean": "accuracy_mean", "std": "accuracy_std"})
    summary.to_csv(f"{out}/exp10_selector_comparison{suffix}_summary.csv", index=False)
    print("\n=== summary (mean/std over seeds) ===")
    print(summary.to_string(index=False))

    # The stored best-fixed reference is the MNIST logistic grid only.
    best_fixed = None if (args.dataset, args.model) != ("mnist", "logistic") else \
        _load_best_fixed_rule_reference("results/tables/exp9_dominance_grid_10seeds_ALL_root100_summary.csv")
    if best_fixed is not None:
        merged = summary.merge(best_fixed, on=["alpha", "attack_type"], how="left")
        merged.to_csv(f"{out}/exp10_selector_comparison{suffix}_vs_best_fixed.csv", index=False)
        print("\n=== vs. best fixed rule (root=100, 10-seed reference) ===")
        print(merged.to_string(index=False))

    per_run = raw.groupby(["seed", "system"])["accuracy"].mean().reset_index()
    if "Random" in per_run["system"].unique():
        sig = paired_significance(per_run, "system", "accuracy", baseline_label="Random")
        sig.to_csv(f"{out}/exp10_selector_comparison{suffix}_vs_random_significance.csv", index=False)
        print("\n=== paired t-test vs. Random (pooled over grid cells, by seed) ===")
        print(sig.to_string(index=False))
