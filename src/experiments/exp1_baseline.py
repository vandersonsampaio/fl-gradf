"""
Experiment 1: baseline comparison between FL defense systems under attack,
using real non-IID MNIST/CIFAR-10 data (see data/download_datasets.py). It is
also GRADF's per-component ablation: each variant disables part of the
pipeline, so the same table serves both as a comparison against the
literature and as an ablation in the paper text.

Real calls to FederatedLearner, AttackedFederatedLearner and
GRADFFederatedLearner:

  FedAvg          -> AttackedFederatedLearner(aggregation='fedavg')       [no defense]
  FLTrust         -> AttackedFederatedLearner(aggregation='fltrust')      [fixed classical defense]
  Median          -> AttackedFederatedLearner(aggregation='median')       [fixed classical defense]
  Trimmed-Mean    -> AttackedFederatedLearner(aggregation='trimmed_mean') [fixed classical defense]
  RL-only         -> GRADFFederatedLearner with Layers 1/2/4 neutralized
                     (hardening ablation: only RL-based strategy selection stays active)
  Hardening-only  -> GRADFFederatedLearner with a fixed selector (RL ablation: no adaptive selection)
  Random-selector -> GRADFFederatedLearner with full hardening, RANDOM selection each
                     round (floor of the selection-policy ablation)
  Oracle-selector -> GRADFFederatedLearner with full hardening, ALWAYS selecting the best
                     fixed strategy observed for the true attack_type (ceiling/upper bound
                     of the selection-policy ablation — deliberately "cheats", not a learned policy)
  GRADF           -> full GRADFFederatedLearner (selection via a trained DQN)

Random-selector and Oracle-selector isolate what the SELECTION POLICY (DQN)
is worth, keeping the rest of the pipeline (Layers 1/2/4) identical to
GRADF's — if the DQN doesn't land clearly between the random floor and the
oracle ceiling, the policy hasn't learned anything useful beyond "don't use
FedAvg".

`__main__` runs the comparison across multiple seeds/attack types (see
`src.utils.stats`) and reports mean/std/CI95 per system, plus a paired
t-test of each system against the FedAvg baseline.
"""

import argparse
import os
import time
import types
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import src.defense.aggregation_methods  # noqa: F401  registers median/trimmed_mean/etc.
from src.classification.rl_classifier import RLAttackClassifier
from src.defense.rl_selector import RLDefenseSelector
from src.fl.attacked_learner import AttackedFederatedLearner
from src.fl.gradf_learner import GRADFFederatedLearner
from src.utils.data_loader import (
    generate_detection_dataset,
    generate_selector_experiences,
    load_dataset_participants,
)
from src.utils.logger import get_logger
from src.utils.stats import paired_significance, run_over_seeds, summarize
from src.utils.visualization import plot_bar_comparison

logger = get_logger(__name__)


class FixedActionSelector:
    """"Dumb" selector that always picks the same aggregation strategy —
    used in the 'Hardening-only' variant (hardening active, no adaptive selection)."""

    def __init__(self, action: str = "median"):
        self.action = action

    def select_action(self, attack_type, confidence, **_round_context):
        return self.action, 1.0


class RandomSelector:
    """Picks an aggregation strategy AT RANDOM each round, among the DQN's
    candidate actions — the floor of the selection-policy ablation.
    If the DQN doesn't beat this by a clear margin, it hasn't learned any policy."""

    def __init__(self, actions: List[str], seed: int = 0):
        self.actions = actions
        self._rng = np.random.RandomState(seed)

    def select_action(self, attack_type, confidence, **_round_context):
        return self._rng.choice(self.actions), 1.0


class OracleSelector:
    """Always picks the strategy with the highest observed reward for the
    current attack_idx — the ceiling/upper bound of the selection-policy
    ablation. The mapping comes from the SAME real experiences
    (`generate_selector_experiences`) used to train the DQN
    (`train_gradf_models`), not a guess: for each attack type, it's the
    strategy that actually produced the highest reward in a real FL episode
    against that attack. Deliberately "cheats" (knows in advance the outcome
    the DQN can only estimate) — not a learned policy, but the ceiling that
    learning could reach in the best case."""

    def __init__(self, best_action_by_attack_idx: Dict[int, str], fallback: str = "median"):
        self.best_action_by_attack_idx = best_action_by_attack_idx
        self.fallback = fallback

    def select_action(self, attack_type, confidence, **_round_context):
        return self.best_action_by_attack_idx.get(int(attack_type), self.fallback), 1.0


def _derive_oracle_map(experiences, actions: List[str]) -> Dict[int, str]:
    """For each attack_idx present in `experiences`, returns the action with
    the highest reward — used by `OracleSelector` (see its docstring)."""
    best: Dict[int, tuple] = {}
    for state, action_idx, reward, _next_state in experiences:
        attack_idx = int(state[0])
        if attack_idx not in best or reward > best[attack_idx][1]:
            best[attack_idx] = (actions[action_idx], reward)
    return {attack_idx: action for attack_idx, (action, _) in best.items()}


def train_gradf_models(participants, attack_type, byzantine_fraction, n_classes, n_rounds=8, seed=0):
    """Pretrains classifier and selector with real data generated from the
    SAME participants/attack used in the experiment — a freshly instantiated
    classifier/selector has random weights."""
    attack_labels = ["none", attack_type]
    label_to_idx = {label: i for i, label in enumerate(attack_labels)}

    X, _y_bin, y_attack = generate_detection_dataset(
        participants, n_rounds=n_rounds, attack_types=[attack_type],
        byzantine_fraction=byzantine_fraction, aggregation="fedavg",
        n_classes=n_classes, seed=seed,
    )
    y_idx = [label_to_idx[t] for t in y_attack]
    classifier = RLAttackClassifier(n_attacks=len(attack_labels))
    classifier.train(X, y_idx, epochs=30, batch_size=32, verbose=False)

    selector = RLDefenseSelector()
    experiences = generate_selector_experiences(
        participants, selector.actions, attack_types=attack_labels,
        n_rounds=n_rounds, byzantine_fraction=byzantine_fraction,
        n_classes=n_classes, seed=seed,
    )
    # epochs=200: a single pass leaves the DQN undertrained — see
    # RLDefenseSelector.train_on_experiences's docstring.
    selector.train_on_experiences(experiences, epochs=200, seed=seed, verbose=False)
    oracle_map = _derive_oracle_map(experiences, selector.actions)

    return classifier, selector, attack_labels, oracle_map


class GroundTruthClassifierStub:
    """Replaces `RLAttackClassifier` for the diagnostic "is the headroom
    capturable in principle?": ignores the 4 modality scores and always
    returns the TRUE `attack_idx` of the round (set by
    `GroundTruthOracleLearner` before each `_run_round`) — decouples the
    `OracleSelector`'s decision from the per-client detection flag.

    Isolates two questions that the `OracleSelector` gated by the real
    classifier (`Oracle-selector` in `run_baseline_comparison`) mixes
    together: "does a correct action to apply exist?" (yes — `oracle_map`)
    vs. "does the real classifier detect the attack well enough for that
    action to actually be applied?" (no, for `sign_flipping`/`label_flipping`).
    """

    def __init__(self) -> None:
        self.current_idx = 0

    def classify(self, scores):
        return self.current_idx, 1.0


class GroundTruthOracleLearner(GRADFFederatedLearner):
    """`GRADFFederatedLearner` whose classifier is `GroundTruthClassifierStub`:
    the strategy applied each round is the one from `oracle_map` indexed by
    the round's TRUE attack type (via `self.attack_type`/`self._is_active`),
    not a classifier's prediction — applied uniformly to all hospitals, so
    Layer 3's majority vote is always unanimous.

    Logs each round's `selected_strategy` in `self.strategy_log` —
    instrumentation that confirms (not just assumes) that the correct action
    is actually applied when decoupled from detection. Implemented as an
    instance-level wrap of `self.hardening.full_pipeline` (instead of
    rewriting `_run_round` entirely, like the other subclasses in the tree)
    to avoid duplicating `GRADFFederatedLearner._run_round`'s logic.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.strategy_log: List[Optional[str]] = []

        original_full_pipeline = self.hardening.full_pipeline.__func__

        def _instrumented_full_pipeline(pipeline_self, *p_args, **p_kwargs):
            delta, results = original_full_pipeline(pipeline_self, *p_args, **p_kwargs)
            self.strategy_log.append(results.get("selected_strategy"))
            return delta, results

        self.hardening.full_pipeline = types.MethodType(_instrumented_full_pipeline, self.hardening)

    def _run_round(self, round_num, participants, root_data):
        active = self._is_active(round_num)
        label = self.attack_type if active else "none"
        self.classifier.current_idx = self.attack_labels.index(label)
        return super()._run_round(round_num, participants, root_data)


def run_ground_truth_oracle_comparison(
    dataset: str = "mnist",
    split: str = "non_iid",
    n_clients: int = 10,
    n_rounds: int = 20,
    attack_type: str = "sign_flipping",
    byzantine_fraction: float = 0.2,
    n_classes: int = 10,
    seed: int = 42,
    root_size: Optional[int] = None,
) -> Dict[str, object]:
    """Reproduces the "decoupled oracle" comparison: derives `oracle_map`
    from the same real experiences as `train_gradf_models` (without paying
    the cost of training the classifier, which this variant doesn't use) and
    applies the correct action per the round's TRUE attack type, via
    `GroundTruthOracleLearner`."""
    participants, root_data = load_dataset_participants(
        dataset, split, n_clients, root_size=root_size, root_seed=seed,
    )
    byzantine_ids = list(range(max(1, int(n_clients * byzantine_fraction))))
    attack_labels = ["none", attack_type]

    selector_dummy = RLDefenseSelector()
    experiences = generate_selector_experiences(
        participants, selector_dummy.actions, attack_types=attack_labels,
        n_rounds=8, byzantine_fraction=byzantine_fraction, n_classes=n_classes, seed=seed,
    )
    oracle_map = _derive_oracle_map(experiences, selector_dummy.actions)

    learner = GroundTruthOracleLearner(
        aggregation="fedavg", classifier=GroundTruthClassifierStub(),
        selector=OracleSelector(oracle_map), attack_labels=attack_labels,
        n_rounds=n_rounds, n_classes=n_classes, attack_type=attack_type,
        byzantine_ids=byzantine_ids, seed=seed,
    )
    history = learner.train(participants, root_data=root_data, verbose=False)

    strategy_counts: Dict[str, int] = {}
    for strategy_name in learner.strategy_log:
        strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1

    logger.info(
        "ground_truth_oracle (seed=%d, attack=%s, root_size=%s): oracle_map=%s strategy_counts=%s",
        seed, attack_type, root_size, oracle_map, strategy_counts,
    )

    return {
        "final_accuracy": history[-1].global_accuracy,
        "oracle_map": oracle_map,
        "strategy_counts": strategy_counts,
    }


def run_baseline_comparison(
    dataset: str = "mnist",
    split: str = "non_iid",
    n_clients: int = 10,
    n_rounds: int = 20,
    attack_type: str = "sign_flipping",
    byzantine_fraction: float = 0.2,
    n_classes: int = 10,
    seed: int = 42,
    root_size: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """`root_size`: if given, subsamples FLTrust's `server_val`/root (and the
    selector/oracle's, which implicitly reuse the same `participants`/root
    via `train_gradf_models`) down to N elements."""
    participants, root_data = load_dataset_participants(
        dataset, split, n_clients, root_size=root_size, root_seed=seed,
    )
    byzantine_ids = list(range(max(1, int(n_clients * byzantine_fraction))))

    logger.info("Pretraining classifier/selector with real data...")
    pretrain_start = time.perf_counter()
    classifier, selector, attack_labels, oracle_map = train_gradf_models(
        participants, attack_type, byzantine_fraction, n_classes, seed=seed
    )
    pretrain_seconds = time.perf_counter() - pretrain_start
    # Logged (not just used) so the oracle mapping's coherence under a
    # smaller root can be checked post-hoc.
    logger.info(
        "oracle_map (seed=%d, attack=%s, root_size=%s): %s",
        seed, attack_type, root_size, oracle_map,
    )

    results: Dict[str, Dict[str, float]] = {}

    def _run(name: str, learner, pretrain: Optional[float] = None) -> None:
        """Times only the FL training loop (`learner.train`), which is what's
        directly comparable across the systems. `pretrain` (the cost of
        pretraining classifier/selector, paid once and shared by
        RL-only/Hardening-only/GRADF) is kept in a separate column — mixing
        the two would artificially inflate those 3 systems' "cost per round"
        relative to the classical baselines, which don't pay that cost."""
        logger.info("Running %s...", name)
        start = time.perf_counter()
        history = learner.train(participants, root_data=root_data, verbose=False)
        wall_clock_seconds = time.perf_counter() - start
        results[name] = {
            "final_accuracy": history[-1].global_accuracy,
            "wall_clock_seconds": wall_clock_seconds,
            "pretrain_seconds": pretrain if pretrain is not None else float("nan"),
        }

    common = dict(
        n_rounds=n_rounds, n_classes=n_classes, attack_type=attack_type,
        byzantine_ids=byzantine_ids, seed=seed,
    )

    _run("FedAvg", AttackedFederatedLearner(aggregation="fedavg", **common))
    _run("FLTrust", AttackedFederatedLearner(aggregation="fltrust", **common))
    _run("Median", AttackedFederatedLearner(aggregation="median", **common))
    _run("Trimmed-Mean", AttackedFederatedLearner(aggregation="trimmed_mean", **common))

    _run("RL-only", GRADFFederatedLearner(
        aggregation="fedavg", classifier=classifier, selector=selector,
        attack_labels=attack_labels,
        magnitude_threshold=1e12, accuracy_tolerance=1.0,
        dp_epsilon=1e12, dp_clipping=1e12,
        **common,
    ), pretrain=pretrain_seconds)

    _run("Hardening-only", GRADFFederatedLearner(
        aggregation="fedavg", classifier=classifier, selector=FixedActionSelector("median"),
        attack_labels=attack_labels,
        **common,
    ), pretrain=pretrain_seconds)

    # Floor (random) and ceiling (oracle) of the selection policy — same full
    # GRADF hardening, only the selection policy changes. See the module and
    # RandomSelector/OracleSelector docstrings.
    _run("Random-selector", GRADFFederatedLearner(
        aggregation="fedavg", classifier=classifier,
        selector=RandomSelector(selector.actions, seed=seed),
        attack_labels=attack_labels,
        **common,
    ), pretrain=pretrain_seconds)

    _run("Oracle-selector", GRADFFederatedLearner(
        aggregation="fedavg", classifier=classifier,
        selector=OracleSelector(oracle_map),
        attack_labels=attack_labels,
        **common,
    ), pretrain=pretrain_seconds)

    _run("GRADF", GRADFFederatedLearner(
        aggregation="fedavg", classifier=classifier, selector=selector,
        attack_labels=attack_labels,
        **common,
    ), pretrain=pretrain_seconds)

    return results


def _run_baseline_for_seed(
    seed: int,
    attack_types: List[str],
    dataset: str,
    split: str,
    n_clients: int,
    n_rounds: int,
    byzantine_fraction: float,
    n_classes: int,
    root_size: Optional[int] = None,
) -> pd.DataFrame:
    """Runs `run_baseline_comparison` for one seed across all `attack_types`,
    returning one row per (attack_type, system) — the format
    `src.utils.stats.run_over_seeds` expects (one seed per call)."""
    rows = []
    for attack_type in attack_types:
        results = run_baseline_comparison(
            dataset=dataset, split=split, n_clients=n_clients, n_rounds=n_rounds,
            attack_type=attack_type, byzantine_fraction=byzantine_fraction,
            n_classes=n_classes, seed=seed, root_size=root_size,
        )
        for system, metrics in results.items():
            rows.append({"attack_type": attack_type, "system": system, **metrics})
    return pd.DataFrame(rows)


def _run_ground_truth_oracle_for_seed(
    seed: int,
    attack_types: List[str],
    dataset: str,
    split: str,
    n_clients: int,
    n_rounds: int,
    byzantine_fraction: float,
    n_classes: int,
    root_size: Optional[int] = None,
) -> pd.DataFrame:
    """Runs `run_ground_truth_oracle_comparison` for one seed across all
    `attack_types` — same shape as `_run_baseline_for_seed`, so it can reuse
    `run_over_seeds`."""
    rows = []
    for attack_type in attack_types:
        result = run_ground_truth_oracle_comparison(
            dataset=dataset, split=split, n_clients=n_clients, n_rounds=n_rounds,
            attack_type=attack_type, byzantine_fraction=byzantine_fraction,
            n_classes=n_classes, seed=seed, root_size=root_size,
        )
        rows.append({
            "attack_type": attack_type,
            "final_accuracy": result["final_accuracy"],
            "strategy_counts": str(result["strategy_counts"]),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--split", default="non_iid", choices=["iid", "non_iid"])
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--n_rounds", type=int, default=20)
    parser.add_argument("--attack_types", nargs="+", default=["sign_flipping"])
    parser.add_argument("--byzantine_fraction", type=float, default=0.2)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--root_size", type=int, default=None,
        help="Subsamples FLTrust's server_val/root (and the classifier/"
             "selector/oracle pretraining) down to N elements. Suffixes the "
             "output CSVs with _root{N} so the full-server_val run isn't "
             "overwritten.",
    )
    parser.add_argument(
        "--ground_truth_oracle", action="store_true",
        help="Instead of the full 9-system ablation, runs only the oracle "
             "decoupled from per-client detection (GroundTruthOracleLearner). "
             "Saves to exp1_groundtruth_oracle{suffix}_raw.csv.",
    )
    args = parser.parse_args()

    if args.ground_truth_oracle:
        suffix = f"_root{args.root_size}" if args.root_size is not None else ""
        gt_raw = run_over_seeds(
            _run_ground_truth_oracle_for_seed,
            seeds=args.seeds,
            attack_types=args.attack_types,
            dataset=args.dataset, split=args.split, n_clients=args.n_clients,
            n_rounds=args.n_rounds, byzantine_fraction=args.byzantine_fraction,
            n_classes=args.n_classes, root_size=args.root_size,
        )
        os.makedirs("results/tables", exist_ok=True)
        gt_raw.to_csv(f"results/tables/exp1_groundtruth_oracle{suffix}_raw.csv", index=False)
        print("=== ground-truth oracle (per seed) ===")
        print(gt_raw.to_string(index=False))

        gt_summary = summarize(gt_raw, ["attack_type"], "final_accuracy")
        gt_summary.to_csv(f"results/tables/exp1_groundtruth_oracle{suffix}_summary.csv", index=False)
        print("\n=== ground-truth oracle summary (mean/std/CI95 over seeds) ===")
        print(gt_summary.to_string(index=False))
        raise SystemExit(0)

    raw = run_over_seeds(
        _run_baseline_for_seed,
        seeds=args.seeds,
        attack_types=args.attack_types,
        dataset=args.dataset, split=args.split, n_clients=args.n_clients,
        n_rounds=args.n_rounds, byzantine_fraction=args.byzantine_fraction,
        n_classes=args.n_classes, root_size=args.root_size,
    )

    suffix = f"_root{args.root_size}" if args.root_size is not None else ""
    os.makedirs("results/tables", exist_ok=True)
    raw.to_csv(f"results/tables/exp1_baseline{suffix}_raw.csv", index=False)
    print("=== raw (per seed) ===")
    print(raw.to_string(index=False))

    summaries = []
    for value_col in ["final_accuracy", "wall_clock_seconds", "pretrain_seconds"]:
        s = summarize(raw, ["attack_type", "system"], value_col)
        s = s.rename(columns={c: f"{value_col}_{c}" for c in ["mean", "std", "ci95_low", "ci95_high", "n"]})
        summaries.append(s.set_index(["attack_type", "system"]))
    summary = pd.concat(summaries, axis=1).reset_index()

    sig_frames = []
    for attack_type, group in raw.groupby("attack_type"):
        sig = paired_significance(group, "system", "final_accuracy", baseline_label="FedAvg")
        sig["attack_type"] = attack_type
        sig_frames.append(sig)
    sig_df = pd.concat(sig_frames, ignore_index=True) if sig_frames else pd.DataFrame(
        columns=["system", "mean_diff", "p_value", "n", "attack_type"]
    )

    summary_df = summary.merge(sig_df, on=["attack_type", "system"], how="left")
    summary_df.to_csv(f"results/tables/exp1_baseline{suffix}_summary.csv", index=False)
    print("\n=== summary (mean/std/CI95 over seeds, vs. FedAvg) ===")
    print(summary_df.to_string(index=False))

    mean_per_system = raw.groupby("system")["final_accuracy"].mean().to_dict()
    plot_bar_comparison(
        mean_per_system,
        title=f"Baseline comparison ({args.dataset}, mean over {len(args.seeds)} seed(s))",
        ylabel="Final accuracy",
        save_path=f"results/figures/exp1_baseline_comparison{suffix}.png",
    )
