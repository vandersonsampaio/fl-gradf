"""
Experiment 4: adaptive adversary — the attacker changes attack type each
round trying to evade the classifier. This is Gate 1 of the security claim:
it decides whether the paper has a defensible security claim, so the
baselines here need to be the BEST available, not just the weakest.

Compares, all exposed to the SAME attack rotation and the same seed:

  GRADF                 -> classifier+selector re-evaluate the attack every
                           round (adaptive defense, GRADF)
  Median/FLTrust/
  Trimmed-Mean/Krum
  (static)              -> fixed aggregation, no detection/selection —
                           classical static defenses, don't react to the rotation
  TARS                  -> reproduction of Ahmed et al. (2025) — competing
                           adaptive selection via trust score + tabular
                           Q-learning (see src/defense/tars_selector.py)
  AdaBFL                -> reproduction of Tang, Liu & Huang (2026) —
                           competing adaptive AGGREGATION (not selection):
                           filter malicious -> trimmed-mean -> derivative
                           model, fused via self-tuning weights (see
                           src/defense/adabfl_aggregator.py)
  FedStrategist          -> reproduction of Haque, Kamal & Hossain (2025) —
                           competing adaptive selection via a LinUCB
                           contextual bandit over {fedavg, median, krum}
                           (see src/defense/fedstrategist_selector.py)

Gate 1 success criterion: GRADF > FLTrust (the best fixed defense) AND
GRADF > TARS/AdaBFL/FedStrategist (the competing adaptive baselines) in at
least 2 of the 3 rotation phases, with a per-seed paired difference that is
statistically significant (p<0.05) and has a non-negligible Cohen's d — not
just non-overlapping CI95.
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd

import src.defense.aggregation_methods  # noqa: F401  registers median/trimmed_mean/krum/etc.
from src.classification.rl_classifier import RLAttackClassifier
from src.defense.aggregation_methods import KrumStrategy
from src.defense.adabfl_aggregator import AdaBFLAggregator
from src.defense.fedstrategist_selector import DiagnosticStateVector, LinUCBAgent
from src.defense.rl_selector import RLDefenseSelector
from src.defense.tars_selector import TARSSelector, TrustScorer, cross_entropy_loss
from src.fl.attacked_learner import AttackedFederatedLearner, RotatingAttackedFederatedLearner, attack_for_round
from src.fl.federated_learner import RoundResult, _STRATEGIES
from src.fl.gradf_learner import GRADFFederatedLearner
from src.utils.data_loader import (
    generate_detection_dataset,
    generate_rotating_selector_experiences,
    generate_selector_experiences,
    load_dataset_participants,
)
from src.utils.logger import get_logger
from src.utils.stats import paired_significance, run_over_seeds, summarize
from src.utils.visualization import plot_accuracy_over_rounds

logger = get_logger(__name__)

STATIC_BASELINE_STRATEGIES = ["median", "fltrust", "trimmed_mean", "krum"]
STATIC_BASELINE_LABELS = {
    "median": "Median (static)",
    "fltrust": "FLTrust (static)",
    "trimmed_mean": "Trimmed-Mean (static)",
    "krum": "Krum (static)",
}

# Alias: static attack rotation (no detection/selection) now lives in
# src.fl.attacked_learner, reusable by
# src.utils.data_loader.generate_rotating_selector_experiences without a
# circular import. Kept under the old name here for readability.
RotatingStrategyLearner = RotatingAttackedFederatedLearner


class AdaptiveAttacker(GRADFFederatedLearner):
    """GRADFFederatedLearner whose `attack_type` changes every round, cycling
    through `attack_sequence` — simulates an adversary trying to evade the classifier."""

    def __init__(self, *args, attack_sequence: Optional[List[str]] = None, **kwargs):
        self.attack_sequence = attack_sequence or ["sign_flipping", "gaussian_noise", "label_flipping"]
        kwargs.setdefault("attack_type", self.attack_sequence[0])
        super().__init__(*args, **kwargs)

    def _run_round(self, round_num, participants, root_data):
        self.attack_type = attack_for_round(self.attack_sequence, round_num)
        return super()._run_round(round_num, participants, root_data)


class TARSLearner(AttackedFederatedLearner):
    """Competing baseline: reproduction of TARS (Ahmed et al., 2025) — see
    `src/defense/tars_selector.py` for the reproduced architecture and the
    two instantiation choices documented there where the original paper
    doesn't specify a closed-form formula. Exposed to the SAME attack
    rotation as `AdaptiveAttacker` and `RotatingStrategyLearner`."""

    CANDIDATE_STRATEGIES = ["krum", "median", "trimmed_mean", "fltrust"]

    def __init__(self, *args, attack_sequence: Optional[List[str]] = None, n_rounds: int = 15, **kwargs):
        self.attack_sequence = attack_sequence or ["sign_flipping", "gaussian_noise", "label_flipping"]
        kwargs.setdefault("attack_type", self.attack_sequence[0])
        kwargs.setdefault("aggregation", self.CANDIDATE_STRATEGIES[0])
        seed = kwargs.get("seed", 42)
        super().__init__(*args, n_rounds=n_rounds, **kwargs)
        self.trust_scorer = TrustScorer()
        self.tars_selector = TARSSelector(self.CANDIDATE_STRATEGIES, n_rounds=n_rounds, seed=seed)

    def _run_round(self, round_num, participants, root_data):
        self.attack_type = attack_for_round(self.attack_sequence, round_num)
        active = self._is_active(round_num)
        param_updates, _ = self._compute_param_updates(participants, active)

        eval_data = root_data or {"X": participants[0].X_test, "y": participants[0].y_test}

        old_model = self._make_model(participants[0].n_features)
        old_model.set_params(self._global_params)  # type: ignore[arg-type]
        old_acc = old_model.accuracy(eval_data["X"], eval_data["y"])
        old_loss = cross_entropy_loss(old_model, eval_data["X"], eval_data["y"])

        trust_values = []
        for p, update in zip(participants, param_updates):
            candidate_params = self._global_params + update  # type: ignore[operator]
            candidate = self._make_model(p.n_features)
            candidate.set_params(candidate_params)
            cand_loss = cross_entropy_loss(candidate, eval_data["X"], eval_data["y"])
            tau = self.trust_scorer.score(
                p.id, self._global_params, candidate_params, update,
                loss_divergence=cand_loss - old_loss,
            )
            trust_values.append(tau)
        mean_trust = float(sum(trust_values) / len(trust_values))

        state = (old_acc, old_loss, mean_trust)
        strategy_name = self.tars_selector.select_action(state, round_num)

        # server_update: always computed because TARS's policy can pick
        # 'fltrust' in any round, independent of the top-level `aggregation=`
        # used only to satisfy FederatedLearner's constructor validation.
        server_root = root_data or self._carve_root(participants[0])
        server_update = self._make_model(participants[0].n_features).fit(server_root["X"], server_root["y"])
        sample_sizes = [p.n_train for p in participants]

        strategy = _STRATEGIES[strategy_name]()
        agg_delta, _ = strategy.aggregate(param_updates, sample_sizes=sample_sizes, server_update=server_update)
        self._global_params = self._global_params + agg_delta  # type: ignore[operator]

        new_model = self._make_model(participants[0].n_features)
        new_model.set_params(self._global_params)
        new_acc = new_model.accuracy(eval_data["X"], eval_data["y"])
        new_loss = cross_entropy_loss(new_model, eval_data["X"], eval_data["y"])

        # Weights (alpha1,alpha2,alpha3)=(1, 0.5, 0.5) in the reward: not
        # numerically specified by the original paper ("tunable weights"),
        # chosen to give accuracy (the target metric) dominant weight with
        # loss/trust as regularizing terms.
        reward = new_acc - 0.5 * new_loss + 0.5 * mean_trust
        next_state = (new_acc, new_loss, mean_trust)
        self.tars_selector.update(state, strategy_name, reward, next_state)

        per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
        global_acc = float(sum(per_acc.values()) / len(per_acc))
        return RoundResult(round_num, global_acc, per_acc, trust_scores=None, n_accepted=None)


class AdaBFLLearner(AttackedFederatedLearner):
    """Competing baseline: reproduction of AdaBFL (Tang, Liu & Huang, 2026)
    — see `src/defense/adabfl_aggregator.py` for the reproduced Algorithm
    1+2 (series/"AdaBFL-1" configuration) and the documented caveats where
    the paper is ambiguous or internally inconsistent. Exposed to the SAME
    attack rotation as `AdaptiveAttacker`/`TARSLearner`/`FedStrategistLearner`.

    Unlike TARS/FedStrategist (which SELECT among existing aggregation
    rules), AdaBFL fuses its own filtered/trimmed/derivative models every
    round via self-tuning weights — so `_run_round` bypasses
    `self._strategy`/`_STRATEGIES` entirely and drives
    `AdaBFLAggregator.aggregate()` directly. `aggregation='fedavg'` is set
    only to satisfy `FederatedLearner.__init__`'s validation, mirroring how
    `TARSLearner` sets a placeholder `aggregation=` for the same reason.
    """

    def __init__(self, *args, attack_sequence: Optional[List[str]] = None, **kwargs):
        self.attack_sequence = attack_sequence or ["sign_flipping", "gaussian_noise", "label_flipping"]
        kwargs.setdefault("attack_type", self.attack_sequence[0])
        kwargs.setdefault("aggregation", "fedavg")
        super().__init__(*args, **kwargs)
        self.adabfl = AdaBFLAggregator()

    def _run_round(self, round_num, participants, root_data):
        self.attack_type = attack_for_round(self.attack_sequence, round_num)
        active = self._is_active(round_num)
        param_updates, _is_byz_list = self._compute_param_updates(participants, active)
        sample_sizes = [p.n_train for p in participants]

        agg_delta, metadata = self.adabfl.aggregate(
            self._global_params, param_updates, round_num, sample_sizes=sample_sizes,
        )
        self._global_params = self._global_params + agg_delta  # type: ignore[operator]

        per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
        global_acc = float(np.mean(list(per_acc.values())))
        return RoundResult(
            round_num, global_acc, per_acc, trust_scores=None,
            n_accepted=len(metadata["benign_indices"]),
        )


class FedStrategistLearner(AttackedFederatedLearner):
    """Competing baseline: reproduction of FedStrategist (Haque, Kamal &
    Hossain, 2025) — see `src/defense/fedstrategist_selector.py` for the
    reproduced LinUCB contextual bandit and the documented note on why its
    public repo wasn't ported wholesale (different model representation).
    Exposed to the SAME attack rotation as
    `AdaptiveAttacker`/`TARSLearner`/`AdaBFLLearner`."""

    CANDIDATE_STRATEGIES = ["fedavg", "median", "krum"]
    STRATEGY_COST = {"fedavg": 0.1, "median": 0.4, "krum": 0.8}  # paper S1 Appendix

    def __init__(
        self, *args, attack_sequence: Optional[List[str]] = None,
        lambda_cost: float = 0.5, alpha: float = 1.5, **kwargs,
    ):
        self.attack_sequence = attack_sequence or ["sign_flipping", "gaussian_noise", "label_flipping"]
        kwargs.setdefault("attack_type", self.attack_sequence[0])
        kwargs.setdefault("aggregation", self.CANDIDATE_STRATEGIES[0])
        seed = kwargs.get("seed", 42)
        super().__init__(*args, **kwargs)
        self.lambda_cost = lambda_cost
        self.bandit = LinUCBAgent(self.CANDIDATE_STRATEGIES, alpha=alpha, seed=seed)

    def _run_round(self, round_num, participants, root_data):
        self.attack_type = attack_for_round(self.attack_sequence, round_num)
        active = self._is_active(round_num)
        param_updates, _is_byz_list = self._compute_param_updates(participants, active)

        eval_data = root_data or {"X": participants[0].X_test, "y": participants[0].y_test}
        old_model = self._make_model(participants[0].n_features)
        old_model.set_params(self._global_params)  # type: ignore[arg-type]
        old_acc = old_model.accuracy(eval_data["X"], eval_data["y"])

        state = DiagnosticStateVector.compute(param_updates)
        strategy_name = self.bandit.select_action(state)

        if strategy_name == "krum":
            n_byz = sum(1 for b in _is_byz_list if b)
            strategy = KrumStrategy(n_byzantine=n_byz)
        else:
            strategy = _STRATEGIES[strategy_name]()
        sample_sizes = [p.n_train for p in participants]
        agg_delta, _ = strategy.aggregate(param_updates, sample_sizes=sample_sizes)
        self._global_params = self._global_params + agg_delta  # type: ignore[operator]

        new_model = self._make_model(participants[0].n_features)
        new_model.set_params(self._global_params)
        new_acc = new_model.accuracy(eval_data["X"], eval_data["y"])

        reward = (new_acc - old_acc) - self.lambda_cost * self.STRATEGY_COST[strategy_name]
        self.bandit.update(state, strategy_name, reward)

        per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
        global_acc = float(np.mean(list(per_acc.values())))
        return RoundResult(round_num, global_acc, per_acc, trust_scores=None, n_accepted=None)


def run_adaptive_experiment(
    dataset: str = "mnist",
    split: str = "non_iid",
    n_clients: int = 10,
    n_rounds: int = 15,
    attack_sequence: Optional[List[str]] = None,
    byzantine_fraction: float = 0.2,
    n_classes: int = 10,
    seed: int = 42,
    root_size: Optional[int] = None,
) -> pd.DataFrame:
    """`root_size`: if given, subsamples FLTrust's `server_val`/root down to N
    elements — used by FLTrust (static), by `TARSLearner` (which has
    `fltrust` as one of its 4 candidate strategies), and by GRADF (Layer 3
    can vote `fltrust`), since they all share the SAME `root_data` loaded here."""
    attack_sequence = attack_sequence or ["sign_flipping", "gaussian_noise", "label_flipping"]
    participants, root_data = load_dataset_participants(
        dataset, split, n_clients, root_size=root_size, root_seed=seed,
    )
    byzantine_ids = list(range(max(1, int(n_clients * byzantine_fraction))))

    attack_labels = ["none"] + attack_sequence
    label_to_idx = {label: i for i, label in enumerate(attack_labels)}

    logger.info("Pretraining multi-class classifier (%s, seed=%d)...", attack_sequence, seed)
    X, _y_bin, y_attack = generate_detection_dataset(
        participants, n_rounds=8, attack_types=attack_sequence,
        byzantine_fraction=byzantine_fraction, aggregation="fedavg",
        n_classes=n_classes, seed=seed,
    )
    y_idx = [label_to_idx[t] for t in y_attack]
    classifier = RLAttackClassifier(n_attacks=len(attack_labels))
    classifier.train(X, y_idx, epochs=30, batch_size=32, verbose=False)

    selector = RLDefenseSelector()
    static_experiences = generate_selector_experiences(
        participants, selector.actions, attack_types=attack_labels,
        n_rounds=8, byzantine_fraction=byzantine_fraction, n_classes=n_classes, seed=seed,
    )
    # Train/deployment mismatch: training only on SINGLE, FIXED-attack
    # episodes (static_experiences) doesn't expose the selector to the
    # condition it will actually face here — a ROTATION of attacks. We
    # combine both sources in the same training pass instead of swapping one
    # for the other: static_experiences gives fine-grained signal per
    # isolated attack type, rotating_experiences gives signal under the real
    # deployment condition.
    rotating_experiences = generate_rotating_selector_experiences(
        participants, selector.actions, attack_sequence=attack_sequence,
        n_rounds=9, byzantine_fraction=byzantine_fraction, n_classes=n_classes, seed=seed,
    )
    # epochs=200: a single pass leaves the DQN undertrained — see
    # RLDefenseSelector.train_on_experiences's docstring.
    selector.train_on_experiences(static_experiences + rotating_experiences, epochs=200, seed=seed, verbose=False)

    gradf = AdaptiveAttacker(
        n_rounds=n_rounds, aggregation="fedavg", n_classes=n_classes,
        byzantine_ids=byzantine_ids, seed=seed, attack_sequence=attack_sequence,
        classifier=classifier, selector=selector, attack_labels=attack_labels,
    )
    histories = [("GRADF", gradf.train(participants, root_data=root_data, verbose=False))]

    for strategy in STATIC_BASELINE_STRATEGIES:
        static = RotatingStrategyLearner(
            strategy=strategy, n_rounds=n_rounds, n_classes=n_classes,
            byzantine_ids=byzantine_ids, seed=seed, attack_sequence=attack_sequence,
        )
        histories.append((
            STATIC_BASELINE_LABELS[strategy],
            static.train(participants, root_data=root_data, verbose=False),
        ))

    tars = TARSLearner(
        n_rounds=n_rounds, n_classes=n_classes,
        byzantine_ids=byzantine_ids, seed=seed, attack_sequence=attack_sequence,
    )
    histories.append(("TARS", tars.train(participants, root_data=root_data, verbose=False)))

    adabfl = AdaBFLLearner(
        n_rounds=n_rounds, n_classes=n_classes,
        byzantine_ids=byzantine_ids, seed=seed, attack_sequence=attack_sequence,
    )
    histories.append(("AdaBFL", adabfl.train(participants, root_data=root_data, verbose=False)))

    fedstrategist = FedStrategistLearner(
        n_rounds=n_rounds, n_classes=n_classes,
        byzantine_ids=byzantine_ids, seed=seed, attack_sequence=attack_sequence,
    )
    histories.append((
        "FedStrategist", fedstrategist.train(participants, root_data=root_data, verbose=False),
    ))

    rows = []
    for system, history in histories:
        for r in history:
            rows.append({
                "round": r.round_num,
                "accuracy": r.global_accuracy,
                "attack_type": attack_for_round(attack_sequence, r.round_num),
                "system": system,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--split", default="non_iid", choices=["iid", "non_iid"])
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--n_rounds", type=int, default=15)
    parser.add_argument("--attack_sequence", nargs="+", default=None)
    parser.add_argument("--byzantine_fraction", type=float, default=0.2)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--root_size", type=int, default=None,
        help="Subsamples FLTrust's server_val/root (used by FLTrust static, "
             "TARS and GRADF) down to N elements. Suffixes the output CSVs "
             "with _root{N} so the full-server_val run isn't overwritten.",
    )
    args = parser.parse_args()

    raw = run_over_seeds(
        run_adaptive_experiment,
        seeds=args.seeds,
        dataset=args.dataset, split=args.split, n_clients=args.n_clients,
        n_rounds=args.n_rounds, attack_sequence=args.attack_sequence,
        byzantine_fraction=args.byzantine_fraction, n_classes=args.n_classes,
        root_size=args.root_size,
    )

    suffix = f"_root{args.root_size}" if args.root_size is not None else ""
    os.makedirs("results/tables", exist_ok=True)
    raw.to_csv(f"results/tables/exp4_adaptive{suffix}_raw.csv", index=False)
    print("=== raw (per seed/round) ===")
    print(raw.to_string(index=False))

    per_run_mean = raw.groupby(["seed", "system", "attack_type"])["accuracy"].mean().reset_index()
    summary = summarize(per_run_mean, ["attack_type", "system"], "accuracy")
    summary.to_csv(f"results/tables/exp4_adaptive{suffix}_summary.csv", index=False)
    print("\n=== summary (mean accuracy per attack type, over seeds) ===")
    print(summary.to_string(index=False))

    # Gate 1: GRADF vs. the best fixed defense (FLTrust) and vs. TARS, per
    # phase, with a per-seed paired t-test + Cohen's d — not just non-overlapping CI95.
    gate1_rows = []
    for attack_type, group in per_run_mean.groupby("attack_type"):
        for baseline_label in ("FLTrust (static)", "TARS", "AdaBFL", "FedStrategist"):
            sig = paired_significance(group, "system", "accuracy", baseline_label=baseline_label)
            sig = sig[sig["system"] == "GRADF"].copy()
            sig["attack_type"] = attack_type
            sig["baseline"] = baseline_label
            gate1_rows.append(sig)
    gate1 = pd.concat(gate1_rows, ignore_index=True)[
        ["attack_type", "baseline", "mean_diff", "p_value", "cohens_d", "n"]
    ]
    gate1.to_csv(f"results/tables/exp4_gate1_significance{suffix}.csv", index=False)
    print("\n=== Gate 1: GRADF vs. FLTrust (static) / TARS, per phase (paired t-test + Cohen's d) ===")
    print(gate1.to_string(index=False))

    first_seed = args.seeds[0]
    curves = {
        system: raw[(raw["seed"] == first_seed) & (raw["system"] == system)]
        .sort_values("round")["accuracy"].tolist()
        for system in raw["system"].unique()
    }
    plot_accuracy_over_rounds(
        curves,
        title=f"Accuracy under adaptive adversary ({args.dataset}, seed={first_seed})",
        save_path=f"results/figures/exp4_adaptive{suffix}.png",
    )
