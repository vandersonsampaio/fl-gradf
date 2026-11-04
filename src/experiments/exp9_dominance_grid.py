"""
Cheap proof that headroom exists for the GRADF thesis BEFORE investing in
CNN/medical datasets. This is Gate 1 of this line of work: it decides
whether it's worth continuing down this path or pivoting to a different
strategy.

Premise being tested: the GRADF thesis ("different attacks require different
rules -> adapt") can only win in a regime where NO fixed rule dominates the
entire attack suite. Today FLTrust dominates the three phases tested in
`exp4_adaptive.py` — headroom is approximately zero. This experiment
deliberately builds a stress regime (attacks designed against each defense +
severe non-IID heterogeneity) and checks whether FLTrust still wins every
time, even so.

Dominance grid: for each (heterogeneity alpha x attack type), which of the 4
fixed defenses (FLTrust/Median/Trimmed-Mean/Krum) has the highest final
accuracy? Headroom only exists if the winning cell CHANGES across conditions
— see `dominance_table()`/the Gate 1 criterion in `__main__`.

Two attack groups:
  - INFORMED_ATTACKS (`AttackSimulator.INFORMED_ATTACK_TYPES`): omniscient
    attacker via `InformedAttackedFederatedLearner` — designed specifically
    against FLTrust/Median/Trimmed-Mean/Krum (Fang et al., 2020-style).
  - BLIND_ATTACKS: classical attacks (TARS) via `AttackedFederatedLearner`
    (blind attacker, the same used by exp1/exp2/exp4/exp7) — kept as a
    control/comparison against the regime already measured in exp4's Gate 1.

Usage:
    python -m src.experiments.exp9_dominance_grid --seeds 42 43 44
    python -m src.experiments.exp9_dominance_grid --alphas 0.5 0.1 0.05 --seeds 42 43 44

Requires the severe non-IID partitions generated beforehand:
    python data/download_datasets.py --extra_alphas 0.1 0.05
"""

import argparse
from fractions import Fraction
import os
import zlib
from typing import List, Optional

import numpy as np
import pandas as pd

import src.defense.aggregation_methods  # noqa: F401  registers median/trimmed_mean/krum/etc.
from src.classification.attack_simulator import AttackSimulator
from src.fl.attacked_learner import AttackedFederatedLearner, InformedAttackedFederatedLearner
from src.utils.data_loader import load_dataset_participants
from src.utils.logger import get_logger
from src.utils.stats import run_over_seeds, summarize

logger = get_logger(__name__)

FIXED_STRATEGIES = ["fltrust", "median", "trimmed_mean", "krum"]
INFORMED_ATTACKS = AttackSimulator.INFORMED_ATTACK_TYPES
INPUT_SHAPE = {"mnist": (28, 28, 1), "cifar10": (32, 32, 3)}
BLIND_ATTACKS = ["sign_flipping", "gaussian_noise", "label_flipping"]


def round_accuracies(history, window: int = 5) -> tuple:
    """Final-round accuracy and its trailing mean over the last `window` rounds."""
    values = [r.global_accuracy for r in history]
    return values[-1], float(np.mean(values[-window:]))


def _split_name(alpha: float) -> str:
    """alpha=0.5 uses the default `non_iid/` split (already existing); any
    other alpha uses `non_iid_a{alpha}/`, produced by
    `python data/download_datasets.py --extra_alphas`."""
    return "non_iid" if alpha == 0.5 else f"non_iid_a{alpha}"


def run_dominance_grid(
    dataset: str = "mnist",
    alphas: Optional[List[float]] = None,
    attack_types: Optional[List[str]] = None,
    n_clients: int = 10,
    n_rounds: int = 15,
    byzantine_fraction: float = 0.2,
    n_classes: int = 10,
    seed: int = 42,
    root_size: Optional[int] = None,
    strategies: Optional[List[str]] = None,
    seed_rng: bool = False,
    model_type: str = "logistic",
    local_epochs: int = 5,
    local_lr: float = 0.01,
    server_steps="native",
    global_lr: float = 1.0,
) -> pd.DataFrame:
    """`root_size`: if given, subsamples `server_val` (~3000 samples on
    MNIST) down to exactly `root_size` elements before running the grid —
    used to test whether FLTrust's dominance depends on the root's size, not
    just on the anchoring mechanism itself. Only affects the `fltrust`
    strategy (the only one that consumes `server_update`); Median/
    Trimmed-Mean/Krum are, by construction, invariant to this parameter — so
    `strategies` lets the grid be restricted to just `["fltrust"]` for a
    sensitivity re-run, reusing (via `strategies=None`, all 4) the values
    already measured for the others instead of recomputing them.

    `seed_rng`: makes each run reproducible. NumPy's global RNG (used for
    mini-batch shuffling) is seeded from (seed, alpha, attack, strategy) —
    never from the root size, so runs are paired across roots — and the
    server's root-data fit gets its own RNG stream, so the root size does not
    shift the clients' draws. With it, the invariance of Median/Trimmed-Mean/
    Krum to `root_size` becomes testable exactly (except under
    `fltrust_aligned`, whose attacker copies the server's root update).
    Off by default: the historical, unseeded behavior is unchanged.

    `model_type='cnn'` uses the small CNN (`_CNNModel`) with the dataset's
    image shape (P1 CIFAR-10 campaign). With `seed_rng`, TensorFlow is seeded
    from the same run seed (initial global weights) and runs with
    deterministic ops, and the mini-batch order comes from NumPy, as for the
    logistic model. `local_epochs`/`local_lr` keep the historical defaults."""
    alphas = alphas or [0.5, 0.1, 0.05]
    attack_types = attack_types or (INFORMED_ATTACKS + BLIND_ATTACKS)
    strategies = strategies or FIXED_STRATEGIES
    byzantine_ids = list(range(max(1, int(n_clients * byzantine_fraction))))
    model_kwargs = {"local_epochs": local_epochs, "local_lr": local_lr, "server_steps": server_steps,
                    "global_lr": global_lr}
    if model_type == "cnn":
        model_kwargs.update(model_type="cnn", input_shape=INPUT_SHAPE[dataset])
        if seed_rng:
            import tensorflow as tf
            tf.config.experimental.enable_op_determinism()

    rows = []
    for alpha in alphas:
        split = _split_name(alpha)
        participants, root_data = load_dataset_participants(
            dataset, split, n_clients, root_size=root_size, root_seed=seed,
        )
        for attack_type in attack_types:
            learner_cls = (
                InformedAttackedFederatedLearner
                if attack_type in INFORMED_ATTACKS
                else AttackedFederatedLearner
            )
            for strategy in strategies:
                extra = {}
                if seed_rng:
                    run_seed = zlib.crc32(f"{seed}|{alpha}|{attack_type}|{strategy}".encode())
                    np.random.seed(run_seed)
                    extra["server_rng_seed"] = run_seed + 1
                    if model_type == "cnn":
                        import tensorflow as tf
                        tf.keras.utils.set_random_seed(run_seed)  # Keras 3 has its own seed state
                learner = learner_cls(
                    n_rounds=n_rounds, aggregation=strategy, n_classes=n_classes,
                    attack_type=attack_type, byzantine_ids=byzantine_ids, seed=seed,
                    **model_kwargs, **extra,
                )
                acc, trailing = round_accuracies(learner.train(participants, root_data=root_data, verbose=False))
                rows.append({
                    "alpha": alpha, "attack_type": attack_type,
                    "strategy": strategy, "accuracy": acc, "accuracy_trailing5": trailing,
                })
                logger.info(
                    "seed=%d alpha=%s attack=%s strategy=%s acc=%.4f",
                    seed, alpha, attack_type, strategy, acc,
                )

    return pd.DataFrame(rows)


def dominance_table(summary: pd.DataFrame) -> pd.DataFrame:
    """For each (alpha, attack_type), finds the winning strategy (highest
    accuracy_mean) and whether its edge over the runner-up exceeds the sum
    of both their standard deviations (gap > combined std) — the Gate 1
    criterion ("gap larger than the std across seeds")."""
    winners = []
    for (alpha, attack_type), group in summary.groupby(["alpha", "attack_type"]):
        ranked = group.sort_values("accuracy_mean", ascending=False)
        best, second = ranked.iloc[0], ranked.iloc[1]
        gap = best["accuracy_mean"] - second["accuracy_mean"]
        combined_std = best["accuracy_std"] + second["accuracy_std"]
        winners.append({
            "alpha": alpha, "attack_type": attack_type,
            "winner": best["strategy"], "winner_accuracy": best["accuracy_mean"],
            "runner_up": second["strategy"], "runner_up_accuracy": second["accuracy_mean"],
            "gap": gap, "combined_std": combined_std,
            "gap_exceeds_std": bool(gap > combined_std),
        })
    return pd.DataFrame(winners)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.1, 0.05])
    parser.add_argument("--attack_types", nargs="+", default=None)
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--n_rounds", type=int, default=15)
    parser.add_argument("--byzantine_fraction", type=float, default=0.2)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--root_size", type=int, default=None,
        help="Subsamples server_val down to N elements (e.g. 50, 100) before "
             "the grid, to test FLTrust's sensitivity to the root's size. "
             "Suffixes the output CSVs with _root{N} so the full-server_val "
             "run isn't overwritten.",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=None,
        help="Restricts the grid to a subset of FIXED_STRATEGIES (e.g. "
             "'fltrust') — useful with --root_size, since only FLTrust "
             "consumes root_data; avoids recomputing Median/Trimmed-Mean/"
             "Krum, which don't change with root_size.",
    )
    parser.add_argument(
        "--seed_rng", action="store_true",
        help="Reproducible runs (see run_dominance_grid). Adds '_seeded' to "
             "the output names.",
    )
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
    parser.add_argument("--out_dir", default="results/tables")
    args = parser.parse_args()

    raw = run_over_seeds(
        run_dominance_grid, seeds=args.seeds,
        dataset=args.dataset, alphas=args.alphas, attack_types=args.attack_types,
        n_clients=args.n_clients, n_rounds=args.n_rounds,
        byzantine_fraction=args.byzantine_fraction, n_classes=args.n_classes,
        root_size=args.root_size, strategies=args.strategies, seed_rng=args.seed_rng,
        model_type=args.model, local_epochs=args.local_epochs, local_lr=args.local_lr,
        server_steps=args.server_steps, global_lr=args.global_lr,
    )
    if args.server_steps != "native":
        raw["server_steps"] = args.server_steps
    if args.global_lr != 1.0:
        raw["global_lr"] = args.global_lr

    suffix = f"_{args.dataset}_{args.model}" if args.model == "cnn" else ""
    suffix += f"_root{args.root_size}" if args.root_size is not None else ""
    if args.seed_rng:
        suffix += "_seeded"
    if args.server_steps != "native":
        suffix += f"_steps{args.server_steps}"
    if args.global_lr != 1.0:
        suffix += f"_lr{round(1 / args.global_lr)}"
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    raw.to_csv(f"{out}/exp9_dominance_grid{suffix}_raw.csv", index=False)
    print("=== raw (per seed) ===")
    print(raw.to_string(index=False))

    summary = summarize(raw, ["alpha", "attack_type", "strategy"], "accuracy")
    summary = summary.rename(columns={"mean": "accuracy_mean", "std": "accuracy_std"})
    summary.to_csv(f"{out}/exp9_dominance_grid{suffix}_summary.csv", index=False)
    print("\n=== summary (mean/std over seeds) ===")
    print(summary.to_string(index=False))

    dom = dominance_table(summary)
    dom.to_csv(f"{out}/exp9_dominance_grid{suffix}_verdict.csv", index=False)
    print("\n=== dominance grid: winner per (alpha, attack_type) ===")
    print(dom.to_string(index=False))

    non_fltrust_wins = dom[(dom["winner"] != "fltrust") & dom["gap_exceeds_std"]]
    n = len(non_fltrust_wins)
    print(f"\nGate 1: {n} regime(s) where a defense != FLTrust wins, with gap > combined std.")
    if n > 0:
        print(non_fltrust_wins[["alpha", "attack_type", "winner", "gap"]].to_string(index=False))
    if n >= 2:
        print("\nPASS -> headroom exists -> proceed to realistic datasets + CNN.")
    else:
        print("\nFAIL -> no headroom even under stress -> pivot to an alternative strategy.")
