"""
Experiment 10 — competing ADAPTIVE SELECTORS vs. GRADF, root=100
(`references/experimento_seletores_adaptativos.md`).

Goal (see the plan doc for full context): turn "our DQN selector doesn't
capture the headroom" into either "no adaptive selector in this space
captures it" (generalizes the negative result) or, if a competitor wins,
isolate whether the win comes from its DETECTION layer or its SELECTION
POLICY — either reading strengthens Paper 1's C2/C3.

Systems compared, all on the SAME dominance grid (alpha x attack_type),
root=100, 7-rule arsenal, same seeds:

  Random-selector       -> floor: picks 1 of 7 rules uniformly at random
                            each round (reuses `exp1_baseline.RandomSelector`)
  Oracle-selector        -> ceiling: best FIXED rule for the round's TRUE
  (decoupled)               attack type, decoupled from real detection
                            (reuses `exp1_baseline.GroundTruthOracleLearner`)
  GRADF (DQN)             -> our corrected 5-dim-state selector
  FedStrategist           -> LinUCB contextual bandit (Haque et al., 2025),
                            adapted to the FULL 7-rule arsenal (the paper's
                            own arsenal is only 3 rules — widened here per
                            validity condition 2 of the plan doc)
  AdaAggRL                -> continuous-weight TD3-family agent — NO
                            external paper; a from-spec instantiation of the
                            plan doc's one-line description (see
                            `src/defense/adaaggrl_agent.py`'s docstring)

Two detection variants (plan doc condition 4 — run BOTH, (b) first):
  (b) shared detection  -> FedStrategist/AdaAggRL are driven by GRADF's OWN
                            detection layer (ModalityRecorder + the same
                            pretrained RLAttackClassifier), not their own
                            diagnostics. Isolates the SELECTION POLICY as the
                            only remaining variable. If everyone ties here,
                            policy isn't the bottleneck.
  (a) own detection     -> each system uses its own original diagnostics
                            (FedStrategist: 3-dim update-geometry vector;
                            AdaAggRL: a 4th, distinct metric added — see
                            `_own_diagnostic_state_adaaggrl`). A win here
                            that disappears in (b) means the advantage came
                            from the DETECTOR, not the selector.

Random/Oracle don't consume a detection signal by construction (Random
ignores it; Oracle uses the round's TRUE attack type directly), so they are
computed once and reused across both variants' tables.

`results/tables/exp9_dominance_grid_10seeds_ALL_root100_summary.csv` (already
on disk, seeds 42-51, same 21-cell grid, root=100) is reused as the "best
fixed rule" reference column instead of recomputing FLTrust/Median/
Trimmed-Mean/Krum from scratch.

Usage:
    python -m src.experiments.exp10_selector_comparison --variant b --seeds 42 43 44
    python -m src.experiments.exp10_selector_comparison --variant b \\
        --systems fedstrategist random --alphas 0.5 --attack_types sign_flipping
"""

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import src.defense.aggregation_methods  # noqa: F401  registers median/trimmed_mean/krum/etc.
from src.defense.adaaggrl_agent import AdaAggRLAgent
from src.defense.aggregation_methods import KrumStrategy
from src.defense.fedstrategist_selector import DiagnosticStateVector, LinUCBAgent
from src.defense.rl_selector import RLDefenseSelector
from src.detection.modality_recorder import ModalityRecorder
from src.experiments.exp1_baseline import (
    GroundTruthClassifierStub,
    GroundTruthOracleLearner,
    OracleSelector,
    RandomSelector,
    _derive_oracle_map,
    train_gradf_models,
)
from src.experiments.exp9_dominance_grid import BLIND_ATTACKS, INFORMED_ATTACKS, _split_name
from src.fl.attacked_learner import AttackedFederatedLearner
from src.fl.federated_learner import RoundResult, _STRATEGIES
from src.fl.gradf_learner import GRADFFederatedLearner
from src.utils.data_loader import generate_selector_experiences, load_dataset_participants
from src.utils.logger import get_logger
from src.utils.stats import paired_significance, run_over_seeds, summarize

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


def _own_diagnostic_state_adaaggrl(updates: List[np.ndarray]) -> np.ndarray:
    """AdaAggRL's own ~4-metric diagnostic (variant a): FedStrategist's
    3-dim vector (variance of norms, avg pairwise cosine, mean norm) plus a
    4th, distinct signal — the fraction of clients whose update norm is a
    magnitude outlier (|z-score| > 2) this round — giving AdaAggRL genuinely
    different information from FedStrategist's own-detection variant, not
    just a subset/superset relationship."""
    var_norms, avg_cos, mean_norm = DiagnosticStateVector.compute(updates)
    stacked = np.stack(updates, axis=0)
    norms = np.linalg.norm(stacked, axis=1)
    z = (norms - norms.mean()) / (norms.std() + 1e-8)
    frac_flagged = float(np.mean(np.abs(z) > 2.0))
    return np.array([var_norms, avg_cos, mean_norm, frac_flagged])


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
        param_updates, _is_byz_list = self._compute_param_updates(participants, active)
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
        server_update = self._make_model(n_feat).fit(server_root["X"], server_root["y"])
        sample_sizes = [p.n_train for p in participants]

        n_byz = sum(1 for b in _is_byz_list if b)
        strategy = _make_arsenal_strategy(strategy_name, n_byz)
        agg_delta, _ = strategy.aggregate(param_updates, sample_sizes=sample_sizes, server_update=server_update)
        self._global_params = self._global_params + agg_delta  # type: ignore[operator]

        new_model = self._make_model(n_feat)
        new_model.set_params(self._global_params)
        new_acc = new_model.accuracy(eval_data["X"], eval_data["y"])

        reward = new_acc - old_acc
        self.bandit.update(tuple(state), strategy_name, reward)

        per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
        global_acc = float(np.mean(list(per_acc.values())))
        return RoundResult(round_num, global_acc, per_acc, trust_scores=None, n_accepted=None)


class AdaAggRLGridLearner(AttackedFederatedLearner):
    """AdaAggRL over the same 7-rule arsenal: blends every candidate rule's
    own aggregated delta by the agent's continuous weight vector, instead of
    picking one winner. Same pluggable detection source as
    `FedStrategistGridLearner`."""

    def __init__(
        self, *args, variant: str = "a", shared_classifier=None,
        n_labels: int = 2, seed: int = 42, **kwargs,
    ):
        kwargs.setdefault("aggregation", FULL_ARSENAL[0])
        kwargs.setdefault("seed", seed)
        super().__init__(*args, **kwargs)
        self.variant = variant
        self.n_labels = n_labels
        self.agent = AdaAggRLAgent(FULL_ARSENAL, state_dim=4, seed=seed)
        if variant == "b":
            if shared_classifier is None:
                raise ValueError("variant='b' requires shared_classifier")
            self.shared_classifier = shared_classifier
            self.recorder = ModalityRecorder()

    def _run_round(self, round_num, participants, root_data):
        active = self._is_active(round_num)
        param_updates, _is_byz_list = self._compute_param_updates(participants, active)
        n_feat = participants[0].n_features

        eval_data = root_data or {"X": participants[0].X_test, "y": participants[0].y_test}
        old_model = self._make_model(n_feat)
        old_model.set_params(self._global_params)  # type: ignore[arg-type]
        old_acc = old_model.accuracy(eval_data["X"], eval_data["y"])

        if self.variant == "a":
            state = _own_diagnostic_state_adaaggrl(param_updates)
        else:
            new_accs = []
            for u in param_updates:
                cand = self._make_model(n_feat)
                cand.set_params(self._global_params + u)  # type: ignore[operator]
                new_accs.append(cand.accuracy(eval_data["X"], eval_data["y"]))
            hospital_ids = [p.id for p in participants]
            state = _shared_detection_state(
                self.recorder, self.shared_classifier, hospital_ids,
                param_updates, old_acc, new_accs, self.n_labels,
            )

        weights = self.agent.select_weights(state)

        server_root = root_data or self._carve_root(participants[0])
        server_update = self._make_model(n_feat).fit(server_root["X"], server_root["y"])
        sample_sizes = [p.n_train for p in participants]

        n_byz = sum(1 for b in _is_byz_list if b)
        deltas = []
        for name in FULL_ARSENAL:
            strategy = _make_arsenal_strategy(name, n_byz)
            d, _ = strategy.aggregate(param_updates, sample_sizes=sample_sizes, server_update=server_update)
            deltas.append(d)
        blended = sum(w * d for w, d in zip(weights, deltas))
        self._global_params = self._global_params + blended  # type: ignore[operator]

        new_model = self._make_model(n_feat)
        new_model.set_params(self._global_params)
        new_acc = new_model.accuracy(eval_data["X"], eval_data["y"])

        reward = new_acc - old_acc
        self.agent.train_step(state, weights, reward, state)  # single-step episode

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
) -> pd.DataFrame:
    """`root_size=100` by default — validity condition 1 of the plan doc
    (the canonical FLTrust paper's root size, not the 3000-sample pool used
    elsewhere in this repo)."""
    alphas = alphas or [0.5, 0.1, 0.05]
    attack_types = attack_types or (INFORMED_ATTACKS + BLIND_ATTACKS)
    systems = systems or ["random", "oracle", "gradf", "fedstrategist", "adaaggrl"]
    byzantine_ids = list(range(max(1, int(n_clients * byzantine_fraction))))

    needs_pretrain = bool({"gradf", "oracle"} & set(systems)) or (
        variant == "b" and bool({"fedstrategist", "adaaggrl"} & set(systems))
    )

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
                classifier, selector, _labels, oracle_map = train_gradf_models(
                    participants, attack_type, byzantine_fraction, n_classes, n_rounds=8, seed=seed,
                )

            def _add(system_label: str, sys_variant: str, acc: float) -> None:
                rows.append({
                    "alpha": alpha, "attack_type": attack_type, "system": system_label,
                    "variant": sys_variant, "accuracy": acc,
                })

            if "random" in systems:
                learner = GRADFFederatedLearner(
                    aggregation="fedavg", classifier=GroundTruthClassifierStub(),
                    selector=RandomSelector(RLDefenseSelector().actions, seed=seed),
                    attack_labels=attack_labels, n_rounds=n_rounds, n_classes=n_classes,
                    attack_type=attack_type, byzantine_ids=byzantine_ids, seed=seed,
                )
                acc = learner.train(participants, root_data=root_data, verbose=False)[-1].global_accuracy
                _add("Random", "n/a", acc)

            if "oracle" in systems:
                learner = GroundTruthOracleLearner(
                    aggregation="fedavg", classifier=GroundTruthClassifierStub(),
                    selector=OracleSelector(oracle_map), attack_labels=attack_labels,
                    n_rounds=n_rounds, n_classes=n_classes, attack_type=attack_type,
                    byzantine_ids=byzantine_ids, seed=seed,
                )
                acc = learner.train(participants, root_data=root_data, verbose=False)[-1].global_accuracy
                _add("Oracle (decoupled)", "n/a", acc)

            if "gradf" in systems:
                learner = GRADFFederatedLearner(
                    aggregation="fedavg", classifier=classifier, selector=selector,
                    attack_labels=attack_labels, n_rounds=n_rounds, n_classes=n_classes,
                    attack_type=attack_type, byzantine_ids=byzantine_ids, seed=seed,
                )
                acc = learner.train(participants, root_data=root_data, verbose=False)[-1].global_accuracy
                _add("GRADF", "shared (own)", acc)

            if "fedstrategist" in systems:
                learner = FedStrategistGridLearner(
                    n_rounds=n_rounds, n_classes=n_classes, attack_type=attack_type,
                    byzantine_ids=byzantine_ids, seed=seed, variant=variant,
                    shared_classifier=classifier, n_labels=len(attack_labels),
                )
                acc = learner.train(participants, root_data=root_data, verbose=False)[-1].global_accuracy
                _add("FedStrategist", variant, acc)

            if "adaaggrl" in systems:
                learner = AdaAggRLGridLearner(
                    n_rounds=n_rounds, n_classes=n_classes, attack_type=attack_type,
                    byzantine_ids=byzantine_ids, seed=seed, variant=variant,
                    shared_classifier=classifier, n_labels=len(attack_labels),
                )
                acc = learner.train(participants, root_data=root_data, verbose=False)[-1].global_accuracy
                _add("AdaAggRL", variant, acc)

            logger.info("seed=%d alpha=%s attack=%s variant=%s done", seed, alpha, attack_type, variant)

    return pd.DataFrame(rows)


def _load_best_fixed_rule_reference(path: str) -> Optional[pd.DataFrame]:
    """Loads the already-computed root=100, 10-seed fixed-strategy grid
    (`exp9_dominance_grid_10seeds_ALL_root100_summary.csv`) and reduces it
    to the best fixed rule per (alpha, attack_type) — the 'melhor regra
    fixa' reference column the plan doc's output table asks for, without
    recomputing FLTrust/Median/Trimmed-Mean/Krum."""
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
        choices=["random", "oracle", "gradf", "fedstrategist", "adaaggrl"],
    )
    parser.add_argument("--variant", default="b", choices=["a", "b"])
    parser.add_argument("--tag", default="", help="Optional suffix for output filenames.")
    args = parser.parse_args()

    raw = run_over_seeds(
        run_selector_comparison_grid, seeds=args.seeds,
        dataset=args.dataset, alphas=args.alphas, attack_types=args.attack_types,
        n_clients=args.n_clients, n_rounds=args.n_rounds,
        byzantine_fraction=args.byzantine_fraction, n_classes=args.n_classes,
        root_size=args.root_size, systems=args.systems, variant=args.variant,
    )

    suffix = f"_variant{args.variant}" + (f"_{args.tag}" if args.tag else "")
    os.makedirs("results/tables", exist_ok=True)
    raw.to_csv(f"results/tables/exp10_selector_comparison{suffix}_raw.csv", index=False)
    print("=== raw (per seed) ===")
    print(raw.to_string(index=False))

    summary = summarize(raw, ["alpha", "attack_type", "system"], "accuracy")
    summary = summary.rename(columns={"mean": "accuracy_mean", "std": "accuracy_std"})
    summary.to_csv(f"results/tables/exp10_selector_comparison{suffix}_summary.csv", index=False)
    print("\n=== summary (mean/std over seeds) ===")
    print(summary.to_string(index=False))

    best_fixed = _load_best_fixed_rule_reference("results/tables/exp9_dominance_grid_10seeds_ALL_root100_summary.csv")
    if best_fixed is not None:
        merged = summary.merge(best_fixed, on=["alpha", "attack_type"], how="left")
        merged.to_csv(f"results/tables/exp10_selector_comparison{suffix}_vs_best_fixed.csv", index=False)
        print("\n=== vs. best fixed rule (root=100, 10-seed reference) ===")
        print(merged.to_string(index=False))

    per_run = raw.groupby(["seed", "system"])["accuracy"].mean().reset_index()
    if "Random" in per_run["system"].unique():
        sig = paired_significance(per_run, "system", "accuracy", baseline_label="Random")
        sig.to_csv(f"results/tables/exp10_selector_comparison{suffix}_vs_random_significance.csv", index=False)
        print("\n=== paired t-test vs. Random (pooled over grid cells, by seed) ===")
        print(sig.to_string(index=False))
