"""
One job of the common factor runs on CIFAR-10 (results/p1_parity/cifar_factor_rules/ADDENDUM.md):
the trimmed mean or FedAvg under block C's attacks, with the aggregate multiplied by a
constant factor. Same seeds and streams as e0_pilot.py, seeds outside the confirmatory
block 42-51. These rules do not use the root here, so it stays at 100.

Usage: venv/bin/python scripts/p1/cifar_factor_rules.py --rule trimmed_mean --attack sign_flipping --seed 0 --den 5
       venv/bin/python scripts/p1/cifar_factor_rules.py --check OUT_TXT
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
ALPHA, EPOCHS, LR, ROOT, ROUNDS = 0.05, 2, 0.01, 100, 110
OUT = "results/p1_parity/cifar_factor_rules/raw"
E0B = "results/p1_cifar10/e0b/raw"


def run(rule, attack, seed, den, rounds=ROUNDS):
    if seed in range(42, 52):
        raise SystemExit("this pilot must not use the confirmatory seed block 42-51")
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    run_seed = zlib.crc32(f"e0|{seed}|{ALPHA}|{attack}|{rule}|{EPOCHS}".encode())
    tf.keras.utils.set_random_seed(run_seed)
    participants, root = load_dataset_participants(
        "cifar10", _split_name(ALPHA), 10, root_size=ROOT, root_seed=seed)
    cls = InformedAttackedFederatedLearner if attack in INFORMED_ATTACKS else AttackedFederatedLearner
    learner = cls(
        n_rounds=rounds, aggregation=rule, n_classes=10, attack_type=attack,
        byzantine_ids=[0, 1], seed=seed, server_rng_seed=run_seed + 1,
        model_type="cnn", input_shape=SHAPE, local_epochs=EPOCHS,
        local_lr=LR, server_steps="median",
    )
    aggregate = learner._strategy.aggregate
    delta_norm = []

    def scaled(updates, **kw):
        delta, metadata = aggregate(updates, **kw)
        delta = delta / den
        delta_norm.append(float(np.linalg.norm(delta)))
        return delta, metadata
    learner._strategy.aggregate = scaled
    t0 = time.perf_counter()
    hist = learner.train(participants, root_data=root, verbose=False)
    return {"rule": rule, "attack": attack, "seed": seed, "den": den, "run_seed": run_seed,
            "seconds": time.perf_counter() - t0, "accuracy": [float(h.global_accuracy) for h in hist],
            "delta_norm": delta_norm}


def check(out):
    lines, ok = [], True

    def report(name, passed, detail=""):
        nonlocal ok
        ok = ok and bool(passed)
        lines.append(f"{'PASS' if passed else 'FAIL'} {name} {detail}")

    for attack in ("sign_flipping", "trim_attack"):
        with open(f"{E0B}/a{ALPHA}_{attack}_fedavg_e2_s0_r100.json") as f:
            ref = json.load(f)["accuracy"][:2]
        a = run("fedavg", attack, 0, 1, rounds=2)
        report(f"check1 FedAvg at factor 1 reproduces E0b, {attack}", a["accuracy"] == ref, f"{a['accuracy']} vs {ref}")
        t = time.perf_counter()
        one = run("trimmed_mean", attack, 0, 1, rounds=1)["delta_norm"][0]
        five = run("trimmed_mean", attack, 0, 5, rounds=1)["delta_norm"][0]
        report(f"check2 the trimmed mean's first aggregate is 1/5 of factor 1's, {attack}", abs(five / one - 0.2) < 1e-6,
               f"ratio {five / one:.6f}; {(time.perf_counter() - t) / 2:.1f} s per round")
    lines.append("CHECKS_PASS" if ok else "CHECKS_FAIL")
    text = "\n".join(lines) + "\n"
    with open(out, "w") as f:
        f.write(text)
    print(text)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule")
    ap.add_argument("--attack")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--den", type=int)
    ap.add_argument("--check")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check(args.check) else 1)
    path = f"{OUT}/{args.rule}_{args.attack}_s{args.seed}_f{args.den}.json"
    if os.path.exists(path):
        print("exists", path)
        return
    res = run(args.rule, args.attack, args.seed, args.den)
    os.makedirs(OUT, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
