"""
FLTrust server-step diagnostic (results/p1_cifar10/e0b/ADDENDUM3.md).

The server update g0 is computed by the same fit() as the clients, i.e. by
local epochs over the root, so the number of server SGD steps (and ||g0||,
to which FLTrust rescales every client update) grows with the root size.
FLTrust (Cao et al., NDSS 2021, Alg. 1-2) gives server and clients the same
number of local iterations instead. This probe records, per round, ||g0||,
the server steps and the norm of the aggregate, and optionally fixes the
server steps independently of the root size.

  --server_steps native   unchanged code path (epochs over the root)
  --server_steps N        N mini-batches sampled from the root (as in
                          FLTrust's ModelUpdate), 1 pass, same lr and batch
  --server_steps median   N = median over clients of ceil(n_i / b) * epochs
                          (computed from the partition only)

Seeding as in exp9_dominance_grid --seed_rng (MNIST) and e0_pilot.py
(CIFAR-10), so native runs reproduce those code paths. Seeds outside the
confirmatory block 42-51.

Usage (one job):
  python scripts/p1/g0_probe.py --dataset mnist --alpha 0.5 \
      --attack sign_flipping --root_size 100 --server_steps 470 --seed 0
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

OUT = "results/p1_g0_probe/raw"
BATCH = 32
CONFIG = {   # model settings of the MNIST grid (exp9 defaults) and of E0b
    "mnist": dict(model={}, local_epochs=5, rounds=15),
    "cifar10": dict(model={"model_type": "cnn", "input_shape": (32, 32, 3)}, local_epochs=2, rounds=20),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=tuple(CONFIG), required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--attack", required=True)
    ap.add_argument("--root_size", type=int, required=True)
    ap.add_argument("--server_steps", default="native")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int)
    args = ap.parse_args()
    if args.seed in range(42, 52):
        raise SystemExit("the probe must not use the confirmatory seed block 42-51")
    cfg = CONFIG[args.dataset]
    rounds = args.rounds or cfg["rounds"]
    epochs = cfg["local_epochs"]

    name = (f"{args.dataset}_a{args.alpha}_{args.attack}_r{args.root_size}"
            f"_st{args.server_steps}_s{args.seed}")
    path = f"{OUT}/{name}.json"
    if os.path.exists(path):
        print("exists", path)
        return

    if args.dataset == "cifar10":
        import tensorflow as tf
        tf.config.experimental.enable_op_determinism()
        run_seed = zlib.crc32(f"e0|{args.seed}|{args.alpha}|{args.attack}|fltrust|{epochs}".encode())
        tf.keras.utils.set_random_seed(run_seed)
    else:
        run_seed = zlib.crc32(f"{args.seed}|{args.alpha}|{args.attack}|fltrust".encode())
        np.random.seed(run_seed)

    participants, root = load_dataset_participants(
        args.dataset, _split_name(args.alpha), 10, root_size=args.root_size, root_seed=args.seed)
    cls = InformedAttackedFederatedLearner if args.attack in INFORMED_ATTACKS else AttackedFederatedLearner
    learner = cls(
        n_rounds=rounds, aggregation="fltrust", n_classes=10, attack_type=args.attack,
        byzantine_ids=[0, 1], seed=args.seed, server_rng_seed=run_seed + 1,
        local_epochs=epochs, local_lr=0.01, **cfg["model"],
    )

    n_root = len(root["y"])
    native_steps = int(np.ceil(n_root / BATCH)) * epochs
    if args.server_steps == "native":
        steps = native_steps
    elif args.server_steps == "median":
        steps = int(np.median([int(np.ceil(p.n_train / BATCH)) * epochs for p in participants]))
    else:
        steps = int(args.server_steps)

    if args.server_steps != "native":
        # Server fit = `steps` mini-batches drawn from the root with the server
        # RNG, one pass (FLTrust's ModelUpdate); clients are unchanged.
        make_server = learner._server_model

        def fixed_steps_server(n_features):
            model = make_server(n_features)
            fit = model.fit

            def fit_steps(X, y):
                idx = learner._server_rng.integers(0, len(y), steps * BATCH)
                model.n_epochs = 1
                return fit(X[idx], y[idx])
            model.fit = fit_steps
            return model
        learner._server_model = fixed_steps_server

    g0, agg_l2, upd_l2 = [], [], []
    aggregate = learner._strategy.aggregate

    def recording(updates, **kw):
        g0.append(float(np.linalg.norm(kw["server_update"])))
        upd_l2.append([float(np.linalg.norm(u)) for u in updates])
        out = aggregate(updates, **kw)
        agg_l2.append(float(np.linalg.norm(out[0])))
        return out
    learner._strategy.aggregate = recording

    t0 = time.perf_counter()
    hist = learner.train(participants, root_data=root, verbose=False)
    res = {**vars(args), "rounds": rounds, "local_epochs": epochs, "run_seed": run_seed,
           "root_n": n_root, "server_steps_n": steps, "native_steps": native_steps,
           "client_steps": [int(np.ceil(p.n_train / BATCH)) * epochs for p in participants],
           "seconds": time.perf_counter() - t0,
           "accuracy": [float(h.global_accuracy) for h in hist],
           "g0_l2": g0, "agg_l2": agg_l2, "update_l2": upd_l2, "byzantine_ids": [0, 1]}
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f)
    print(name, f"final={res['accuracy'][-1]:.4f}", f"g0={np.median(g0):.3f}", f"{res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
