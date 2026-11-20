"""
One job of the CIFAR-10 step pilot (results/p1_parity/cifar_step_pilot/ADDENDUM.md):
one FLTrust run of block C (results/p1_parity/PLAN.md, item 3) under step parity,
with the aggregate multiplied by a constant factor. Same seeds and streams as
e0_pilot.py, seeds outside the confirmatory block 42-51.

Usage: venv/bin/python scripts/p1/cifar_step_pilot.py --attack sign_flipping --root_size 100 --seed 0
       venv/bin/python scripts/p1/cifar_step_pilot.py --check OUT_TXT
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
ALPHA, RULE, EPOCHS, LR = 0.05, "fltrust", 2, 0.01
FACTOR, ROUNDS = 1 / 3, 110
OUT = "results/p1_parity/cifar_step_pilot/raw"
BLOCK_C = "results/p1_parity/cifar_blockC/raw"


def run(attack, root_size, seed, rounds=ROUNDS, factor=FACTOR):
    if seed in range(42, 52):
        raise SystemExit("the pilot must not use the confirmatory seed block 42-51")
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    run_seed = zlib.crc32(f"e0|{seed}|{ALPHA}|{attack}|{RULE}|{EPOCHS}".encode())
    tf.keras.utils.set_random_seed(run_seed)
    participants, root = load_dataset_participants(
        "cifar10", _split_name(ALPHA), 10, root_size=root_size, root_seed=seed)
    cls = InformedAttackedFederatedLearner if attack in INFORMED_ATTACKS else AttackedFederatedLearner
    learner = cls(
        n_rounds=rounds, aggregation=RULE, n_classes=10, attack_type=attack,
        byzantine_ids=[0, 1], seed=seed, server_rng_seed=run_seed + 1,
        model_type="cnn", input_shape=SHAPE, local_epochs=EPOCHS,
        local_lr=LR, server_steps="median",
    )
    aggregate = learner._strategy.aggregate
    log = []

    def scaled(updates, **kw):
        delta, metadata = aggregate(updates, **kw)
        delta = factor * delta
        g0 = kw["server_update"]
        log.append({"g0_norm": float(np.linalg.norm(g0)), "delta_norm": float(np.linalg.norm(delta)),
                    "step_along_g0": float(np.dot(delta, g0) / np.dot(g0, g0))})
        return delta, metadata
    learner._strategy.aggregate = scaled
    t0 = time.perf_counter()
    hist = learner.train(participants, root_data=root, verbose=False)
    return {"attack": attack, "root_size": root_size, "seed": seed, "factor": factor, "run_seed": run_seed,
            "seconds": time.perf_counter() - t0, "accuracy": [float(h.global_accuracy) for h in hist], "rounds": log}


def check(out):
    lines, ok = [], True

    def report(name, passed, detail=""):
        nonlocal ok
        ok = ok and bool(passed)
        lines.append(f"{'PASS' if passed else 'FAIL'} {name} {detail}")

    for attack in ("sign_flipping", "trim_attack"):
        with open(f"{BLOCK_C}/a{ALPHA}_{attack}_{RULE}_e{EPOCHS}_s0_r100_stmedian.json") as f:
            ref = json.load(f)["accuracy"][:2]
        t = time.perf_counter()
        one = run(attack, 100, 0, rounds=2, factor=1.0)
        report(f"check1 factor 1 reproduces block C, {attack}", one["accuracy"] == ref, f"{one['accuracy']} vs {ref}")
        third = run(attack, 100, 0, rounds=2)
        ratio = third["rounds"][0]["step_along_g0"] / one["rounds"][0]["step_along_g0"]
        report(f"check2 the first round's step is the factor times factor 1's, {attack}", abs(ratio - FACTOR) < 1e-6,
               f"ratio {ratio:.6f}; {(time.perf_counter() - t) / 4:.1f} s per round")
    lines.append("CHECKS_PASS" if ok else "CHECKS_FAIL")
    text = "\n".join(lines) + "\n"
    with open(out, "w") as f:
        f.write(text)
    print(text)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack")
    ap.add_argument("--root_size", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--check")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check(args.check) else 1)
    path = f"{OUT}/{args.attack}_r{args.root_size}_s{args.seed}.json"
    if os.path.exists(path):
        print("exists", path)
        return
    res = run(args.attack, args.root_size, args.seed)
    os.makedirs(OUT, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
