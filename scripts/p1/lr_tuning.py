"""
One job of the global learning rate tuning (results/p1_parity/lr_tuning/ADDENDUM.md): a fixed rule
with no attack and the aggregate multiplied by a constant factor, logging the global model's
accuracy on a fixed sample of the pooled training data after every round.

Usage: venv/bin/python scripts/p1/lr_tuning.py --dataset mnist --rule fltrust --seed 0 --den 5
       venv/bin/python scripts/p1/lr_tuning.py --check OUT_TXT
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
from src.experiments.exp9_dominance_grid import _split_name  # noqa: E402
from src.fl.attacked_learner import AttackedFederatedLearner  # noqa: E402
from src.utils.data_loader import load_dataset_participants  # noqa: E402

ALPHA, ROOT, TRAIN_SAMPLE = 0.05, 100, 5000
SETUP = {
    "mnist": {"rounds": 15, "epochs": 5, "model": {}},
    "cifar10": {"rounds": 110, "epochs": 2, "model": {"model_type": "cnn", "input_shape": (32, 32, 3)}},
}
OUT = "results/p1_parity/lr_tuning/raw"
E0B = "results/p1_cifar10/e0b/raw"


def _run_seed(dataset, rule, seed):
    # The seed strings of exp9 (MNIST) and e0_pilot (CIFAR-10).
    if dataset == "mnist":
        return zlib.crc32(f"{seed}|{ALPHA}|none|{rule}".encode())
    return zlib.crc32(f"e0|{seed}|{ALPHA}|none|{rule}|{SETUP[dataset]['epochs']}".encode())


def _learner(dataset, rule, seed, run_seed, rounds):
    return AttackedFederatedLearner(
        n_rounds=rounds, aggregation=rule, n_classes=10, attack_type="none",
        byzantine_ids=[0, 1], seed=seed, server_rng_seed=run_seed + 1,
        local_epochs=SETUP[dataset]["epochs"], local_lr=0.01, server_steps="median",
        **SETUP[dataset]["model"],
    )


def run(dataset, rule, seed, den, rounds=None, log_train=True):
    if seed in range(42, 52):
        raise SystemExit("the tuning must not use the confirmatory seed block 42-51")
    rounds = rounds or SETUP[dataset]["rounds"]
    run_seed = _run_seed(dataset, rule, seed)
    participants, root = load_dataset_participants(
        dataset, _split_name(ALPHA), 10, root_size=ROOT, root_seed=seed)
    X = np.concatenate([p.X_train for p in participants])
    y = np.concatenate([p.y_train for p in participants])
    pick = np.random.default_rng(seed).choice(len(y), TRAIN_SAMPLE, replace=False)
    X_sample, y_sample = X[pick], y[pick]
    del X, y
    # Built before the seeds are set, so that it draws nothing from the run's streams.
    eval_model = _learner(dataset, rule, seed, run_seed, rounds)._instantiate_model(participants[0].n_features)

    if dataset == "cifar10":
        import tensorflow as tf
        tf.config.experimental.enable_op_determinism()
        tf.keras.utils.set_random_seed(run_seed)
    else:
        np.random.seed(run_seed)
    learner = _learner(dataset, rule, seed, run_seed, rounds)

    aggregate = learner._strategy.aggregate
    delta_norm = []

    def scaled(updates, **kw):
        delta, metadata = aggregate(updates, **kw)
        delta = delta / den
        delta_norm.append(float(np.linalg.norm(delta)))
        return delta, metadata
    learner._strategy.aggregate = scaled

    train_acc = []
    if log_train:
        run_round = learner._run_round

        def logged(*a, **kw):
            result = run_round(*a, **kw)
            eval_model.set_params(learner.global_params)
            train_acc.append(eval_model.accuracy(X_sample, y_sample))
            return result
        learner._run_round = logged

    t0 = time.perf_counter()
    hist = learner.train(participants, root_data=root, verbose=False)
    out = {"dataset": dataset, "rule": rule, "seed": seed, "den": den, "run_seed": run_seed,
           "seconds": time.perf_counter() - t0, "train_accuracy": [float(x) for x in train_acc],
           "test_accuracy": [float(h.global_accuracy) for h in hist], "delta_norm": delta_norm}
    if log_train:
        direct = learner._make_model(participants[0].n_features).accuracy(X_sample, y_sample)
        out["_direct_train_accuracy"] = float(direct)
    return out


def check(path):
    lines, ok = [], True

    def report(name, passed, detail=""):
        nonlocal ok
        ok = ok and bool(passed)
        lines.append(f"{'PASS' if passed else 'FAIL'} {name} {detail}")

    with open(f"{E0B}/a{ALPHA}_none_fedavg_e2_s0_r100.json") as f:
        ref = json.load(f)["accuracy"][:2]
    a = run("cifar10", "fedavg", 0, 1, rounds=2)
    report("check1 CIFAR-10 FedAvg, no attack, factor 1, reproduces E0b with the training log on", a["test_accuracy"] == ref, f"{a['test_accuracy']} vs {ref}")

    for dataset, rounds in (("mnist", 3), ("cifar10", 2)):
        on = run(dataset, "fltrust", 0, 1, rounds=rounds)
        off = run(dataset, "fltrust", 0, 1, rounds=rounds, log_train=False)
        report(f"check2 the training log leaves the run unchanged, {dataset}", on["test_accuracy"] == off["test_accuracy"], f"{on['test_accuracy']} vs {off['test_accuracy']}")
        report(f"check3 the logged value is the global model's training accuracy, {dataset}", on["train_accuracy"][-1] == on["_direct_train_accuracy"],
               f"{on['train_accuracy'][-1]} vs {on['_direct_train_accuracy']}; {on['seconds'] / rounds:.1f} s per round")
        for rule in ("fltrust", "trimmed_mean", "fedavg"):
            one = run(dataset, rule, 0, 1, rounds=1)["delta_norm"][0]
            five = run(dataset, rule, 0, 5, rounds=1)["delta_norm"][0]
            report(f"check4 the first aggregate at factor 1/5 has 1/5 of the norm, {dataset} {rule}", abs(five / one - 0.2) < 1e-6, f"ratio {five / one:.6f}")
    try:
        run("mnist", "fedavg", 42, 1, rounds=1)
        report("check5 confirmatory seeds are refused", False)
    except SystemExit:
        report("check5 confirmatory seeds are refused", True)

    lines.append("CHECKS_PASS" if ok else "CHECKS_FAIL")
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    print(text)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(SETUP))
    ap.add_argument("--rule")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--den", type=int)
    ap.add_argument("--check")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check(args.check) else 1)
    path = f"{OUT}/{args.dataset}_{args.rule}_s{args.seed}_f{args.den}.json"
    if os.path.exists(path):
        print("exists", path)
        return
    res = run(args.dataset, args.rule, args.seed, args.den)
    res.pop("_direct_train_accuracy")
    os.makedirs(OUT, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
