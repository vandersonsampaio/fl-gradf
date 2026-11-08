"""
E0 feasibility pilot on CIFAR-10 (results/p1_plan/PLAN.md, Sec. 5, E0).
One job = one seeded FL run; the per-round global accuracy is saved, so the
outputs can be read at any number of rounds up to --rounds.

  curves:     FedAvg without attack, to fix rounds and local epochs (E0.1)
  separation: the 7 fixed rules under one attack at one alpha (E0.2)
  e0b:        the E0b runs (results/p1_cifar10/e0b/PLAN.md): long FedAvg
              curves, FedAvg and FLTrust (roots 100 and 2,500) under attack;
              the L2 norm of every client update is also saved, per round;
              with --server_steps other than native (P1 step-parity rerun,
              results/p1_parity/PLAN.md), the run goes to
              results/p1_parity/cifar_blockC/raw and ||g0|| is saved too

Seeds come from outside the confirmatory block 42-51. Runs are seeded as in
exp9_dominance_grid (--seed_rng): crc32 run seed, Keras/NumPy seeds,
deterministic ops, NumPy mini-batch order, separate server RNG. CPU, one
thread per process (results/p1_cifar10/e0_bench/SUMMARY.md).

Usage (one job):
  python scripts/p1/e0_pilot.py --mode separation --alpha 0.05 \
      --attack sign_flipping --rule trimmed_mean --seed 0 [--rounds 50]
Grid launcher: scripts/p1/run_e0.sh
"""

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.getcwd())

import src.defense.aggregation_methods  # noqa: E402,F401
from src.experiments.exp9_dominance_grid import INFORMED_ATTACKS, _split_name  # noqa: E402
from src.fl.attacked_learner import AttackedFederatedLearner, InformedAttackedFederatedLearner  # noqa: E402
from src.utils.data_loader import load_dataset_participants  # noqa: E402

SHAPE = (32, 32, 3)
OUT = "results/p1_cifar10/e0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("curves", "separation", "e0b"), required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--attack", default="none")
    ap.add_argument("--rule", default="fedavg")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--local_lr", type=float, default=0.01)
    ap.add_argument("--root_size", type=int, default=100)
    ap.add_argument("--server_steps", default="native")
    args = ap.parse_args()
    if args.seed in range(42, 52):
        raise SystemExit("E0 must not use the confirmatory seed block 42-51")

    name = (f"a{args.alpha}_{args.attack}_{args.rule}_e{args.local_epochs}"
            f"_s{args.seed}")
    if args.mode == "e0b":
        name += f"_r{args.root_size}"
    out_dir = f"{OUT}/{args.mode}" if args.mode != "e0b" else "results/p1_cifar10/e0b/raw"
    if args.server_steps != "native":
        name += f"_st{args.server_steps}"
        out_dir = "results/p1_parity/cifar_blockC/raw"
    path = f"{out_dir}/{name}.json"
    if os.path.exists(path):
        print("exists", path)
        return

    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    run_seed = zlib.crc32(f"e0|{args.seed}|{args.alpha}|{args.attack}|{args.rule}|"
                          f"{args.local_epochs}".encode())
    tf.keras.utils.set_random_seed(run_seed)

    participants, root = load_dataset_participants(
        "cifar10", _split_name(args.alpha), 10, root_size=args.root_size, root_seed=args.seed)
    cls = InformedAttackedFederatedLearner if args.attack in INFORMED_ATTACKS else AttackedFederatedLearner
    learner = cls(
        n_rounds=args.rounds, aggregation=args.rule, n_classes=10, attack_type=args.attack,
        byzantine_ids=[0, 1], seed=args.seed, server_rng_seed=run_seed + 1,
        model_type="cnn", input_shape=SHAPE, local_epochs=args.local_epochs,
        local_lr=args.local_lr, server_steps=args.server_steps,
    )
    norms, g0 = [], []
    if args.mode == "e0b":
        # Wrap the rule's aggregate() to record every update it receives; this
        # reads the updates only, so the run itself is unchanged.
        aggregate = learner._strategy.aggregate

        def recording(updates, **kw):
            norms.append([float(np.linalg.norm(u)) for u in updates])
            if "server_update" in kw and args.server_steps != "native":
                g0.append(float(np.linalg.norm(kw["server_update"])))
            return aggregate(updates, **kw)
        learner._strategy.aggregate = recording
    t0 = time.perf_counter()
    hist = learner.train(participants, root_data=root, verbose=False)
    res = {**vars(args), "run_seed": run_seed, "seconds": time.perf_counter() - t0,
           "accuracy": [float(h.global_accuracy) for h in hist]}
    if args.mode == "e0b":
        res["byzantine_ids"] = [] if args.attack == "none" else [0, 1]
        res["update_l2"] = norms          # rounds x clients, in client order
        if args.server_steps != "native":
            res["g0_l2"] = g0
            res["server_steps_n"] = learner._server_step_count(participants)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f)
    print(name, f"final={res['accuracy'][-1]:.4f}", f"{res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
