"""
E0 memory diagnosis: one seeded CIFAR-10 CNN run, resident memory (RSS)
recorded after every round, to find whether memory grows with the number of
rounds (the 2026-10-29 E0 separation grid ran out of RAM).

Usage: python scripts/p1/e0_memdiag.py [--rounds 50] [--rule fltrust] [--attack trim_attack]
Writes results/p1_cifar10/e0_bench/memdiag_<rule>_<attack>.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import src.defense.aggregation_methods  # noqa: E402,F401
from src.experiments.exp9_dominance_grid import INFORMED_ATTACKS, _split_name  # noqa: E402
from src.fl.attacked_learner import AttackedFederatedLearner, InformedAttackedFederatedLearner  # noqa: E402
from src.utils.data_loader import load_dataset_participants  # noqa: E402


def rss_gb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--rule", default="fltrust")
    ap.add_argument("--attack", default="trim_attack")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(3)
    participants, root = load_dataset_participants("cifar10", _split_name(args.alpha), 10,
                                                   root_size=100, root_seed=3)
    cls = InformedAttackedFederatedLearner if args.attack in INFORMED_ATTACKS else AttackedFederatedLearner
    learner = cls(n_rounds=args.rounds, aggregation=args.rule, n_classes=10, attack_type=args.attack,
                  byzantine_ids=[0, 1], seed=3, server_rng_seed=4, model_type="cnn",
                  input_shape=(32, 32, 3), local_epochs=2)

    trace = {"rss_after_load_gb": rss_gb(), "rounds": []}
    inner = learner._run_round

    def run_round(round_num, *a, **kw):
        t0 = time.perf_counter()
        out = inner(round_num, *a, **kw)
        trace["rounds"].append({"round": round_num, "rss_gb": rss_gb(),
                                "seconds": time.perf_counter() - t0})
        print(round_num, f"rss={trace['rounds'][-1]['rss_gb']:.2f} GB", flush=True)
        return out

    learner._run_round = run_round
    learner.train(participants, root_data=root, verbose=False)
    os.makedirs("results/p1_cifar10/e0_bench", exist_ok=True)
    with open(f"results/p1_cifar10/e0_bench/memdiag_{args.rule}_{args.attack}.json", "w") as f:
        json.dump(trace, f, indent=1)


if __name__ == "__main__":
    main()
