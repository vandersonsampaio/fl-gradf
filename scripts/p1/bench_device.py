"""
E0 cost probe: seconds per FL round on CIFAR-10 with the small CNN, on the
device visible to TensorFlow (select it with CUDA_VISIBLE_DEVICES), with or
without deterministic ops. Also records the honest clients' update L2 norms
(GRADF Layer-1 calibration, results/p1_cifar10/DEVIATIONS.md, D5).

Not a confirmatory run: alpha 0.5, seed 0 (outside the 42-51 block),
canonical root (100), a warm-up round discarded before timing.

Usage:
  CUDA_VISIBLE_DEVICES="" python scripts/p1/bench_device.py --tag cpu
  python scripts/p1/bench_device.py --tag gpu [--deterministic]
"""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())

import src.defense.aggregation_methods  # noqa: E402,F401
from src.experiments.exp10_selector_comparison import AdaAggRLGridLearner  # noqa: E402
from src.fl.attacked_learner import AttackedFederatedLearner  # noqa: E402
from src.utils.data_loader import load_dataset_participants  # noqa: E402

SHAPE = (32, 32, 3)
OUT = "results/p1_cifar10/e0_bench"


class NormRecorder(AttackedFederatedLearner):
    """Records the L2 norm of every client update, per round."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norms = []

    def _compute_param_updates(self, participants, active):
        ups, byz = super()._compute_param_updates(participants, active)
        self.norms.append([float(np.linalg.norm(u)) for u in ups])
        return ups, byz


def timed(make, participants, root, rounds):
    warm = make(1)
    warm.train(participants, root_data=root, verbose=False)
    learner = make(rounds)
    t0 = time.perf_counter()
    hist = learner.train(participants, root_data=root, verbose=False)
    return learner, (time.perf_counter() - t0) / rounds, hist[-1].global_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--systems", nargs="+", default=["trimmed_mean", "fltrust", "adaaggrl"])
    args = ap.parse_args()

    import tensorflow as tf
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

    participants, root = load_dataset_participants("cifar10", "non_iid", 10, root_size=100, root_seed=0)
    common = dict(n_classes=10, model_type="cnn", input_shape=SHAPE,
                  local_epochs=args.local_epochs, byzantine_ids=[0, 1], seed=0)
    res = {"tag": args.tag, "gpus": [g.name for g in gpus], "deterministic": args.deterministic,
           "rounds_timed": args.rounds, "local_epochs": args.local_epochs,
           "n_train_total": int(sum(p.n_train for p in participants)), "systems": {}}

    for system in args.systems:
        tf.keras.utils.set_random_seed(0)
        if system == "adaaggrl":
            def make(r):
                return AdaAggRLGridLearner(n_rounds=r, attack_type="sign_flipping", **common)
        else:
            def make(r, system=system):
                return NormRecorder(n_rounds=r, aggregation=system, attack_type="none", **common)
        learner, s_round, acc = timed(make, participants, root, args.rounds)
        entry = {"s_per_round": s_round, "final_accuracy": acc}
        if isinstance(learner, NormRecorder):
            learner_norms = learner.norms
            entry["honest_update_l2"] = {"per_round_median": [float(np.median(n)) for n in learner_norms],
                                         "min": float(np.min(learner_norms)),
                                         "max": float(np.max(learner_norms))}
        res["systems"][system] = entry
        print(system, json.dumps(entry), flush=True)

    res["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print("peak_rss_gb", round(res["peak_rss_gb"], 2), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{args.tag}{'_det' if args.deterministic else ''}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
