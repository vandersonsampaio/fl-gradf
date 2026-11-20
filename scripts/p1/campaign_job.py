"""
One job of the P1 campaign on CIFAR-10 (results/p1_campaign/PREREGISTRATION.md).

Every setting comes from the job's block, and a job refuses any setting that differs from it.
On seeds 42-51 it also refuses a code hash other than the frozen one and an unexpected
numerical environment. Each job writes one JSON with the whole curve and its machine.

Usage: campaign_job.py run --block e1_main --arm trimmed_mean_r100 --alpha 0.05 --attack trim_attack --seed 42
       campaign_job.py list --block e1_main
       campaign_job.py hash
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import socket
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.getcwd())

OUT = "results/p1_campaign"
FROZEN_HASH = f"{OUT}/CODE.sha256"
CONFIRMATORY = range(42, 52)
ALPHAS = (0.5, 0.1, 0.05)
ATTACKS = ("fltrust_aligned", "trim_attack", "krum_collusion", "low_mag_backdoor",
           "sign_flipping", "gaussian_noise", "label_flipping")
E3_ATTACKS = ("sign_flipping", "label_flipping", "trim_attack", "krum_collusion")
RULES = ("fedavg", "fedprox", "median", "trimmed_mean", "fltrust", "clustering", "krum")
SEEDS = tuple(CONFIRMATORY)
ROUNDS, EPOCHS, LOCAL_LR, TAU_CNN, THIRD = 110, 2, 0.01, 7.86, 1 / 3
ALL_CELLS = [(a, t) for a in ALPHAS for t in ATTACKS]
E3_CELLS = [(a, t) for a in ALPHAS for t in E3_ATTACKS]


def fixed(rule, root):
    return {"kind": "fixed", "rule": rule, "root": root}


def system(name, **kw):
    return {"kind": "system", "system": name, "root": 100, "variant": "a", "extractor": "random", **kw}


BLOCKS = {
    "e1_main": {"factor": THIRD, "steps": "median", "cells": ALL_CELLS,
                "arms": {**{f"{r}_r100": fixed(r, 100) for r in RULES}, "fltrust_r2500": fixed("fltrust", 2500)}},
    "e1_invariant": {"factor": THIRD, "steps": "median", "cells": ALL_CELLS,
                     "arms": {"trimmed_mean_r2500": fixed("trimmed_mean", 2500)}},
    "e1_sens_factor": {"factor": 1.0, "steps": "median", "cells": ALL_CELLS,
                       "arms": {"fltrust_r100": fixed("fltrust", 100), "fltrust_r2500": fixed("fltrust", 2500),
                                "trimmed_mean_r100": fixed("trimmed_mean", 100),
                                "trimmed_mean_r2500": fixed("trimmed_mean", 2500)}},
    "e1_sens_steps": {"factor": 1.0, "steps": "native", "cells": ALL_CELLS,
                      "arms": {"fltrust_r100": fixed("fltrust", 100), "fltrust_r2500": fixed("fltrust", 2500)}},
    "e3_main": {"factor": THIRD, "steps": "median", "cells": E3_CELLS,
                "arms": {"random": system("random"), "attack_aware": system("oracle"), "gradf": system("gradf"),
                         "fedstrategist": system("fedstrategist"), "adaaggrl": system("adaaggrl")}},
    "e3_hardened": {"factor": THIRD, "steps": "median", "cells": E3_CELLS,
                    "arms": {f"{r}_hardened": {"kind": "hardened", "rule": r, "root": 100}
                             for r in ("fltrust", "median", "trimmed_mean", "krum")}},
    "e3_sens": {"factor": THIRD, "steps": "median", "cells": E3_CELLS,
                "arms": {"adaaggrl_pretrained": system("adaaggrl", extractor="pretrained"),
                         "fedstrategist_detector": system("fedstrategist", variant="b")}},
}


def job_keys(block):
    """(arm, alpha, attack, seed) of a block, as fixed by the preregistration."""
    spec = BLOCKS[block]
    keys = []
    for arm in spec["arms"]:
        for alpha, attack in spec["cells"]:
            for seed in SEEDS:
                if block == "e1_invariant" and attack != "fltrust_aligned" and seed != SEEDS[0]:
                    continue
                if block == "e1_sens_factor" and arm == "trimmed_mean_r2500" and attack != "fltrust_aligned":
                    continue
                keys.append((arm, alpha, attack, seed))
    return keys


def code_hash():
    files = sorted(glob.glob("src/**/*.py", recursive=True)) + [os.path.abspath(__file__)]
    h = hashlib.sha256()
    for path in files:
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def cpu_has_avx512():
    with open("/proc/cpuinfo") as f:
        return "avx512f" in f.read()


def check_environment():
    wanted = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}
    wanted.update(CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0")
    if cpu_has_avx512():
        wanted.update(ONEDNN_MAX_CPU_ISA="AVX2", OPENBLAS_CORETYPE="Haswell")
    wrong = {k: os.environ.get(k) for k, v in wanted.items() if os.environ.get(k) != v}
    if wrong:
        raise SystemExit(f"refused: environment differs from the preregistration: {wrong}")


def guard(args, spec, arm):
    """Refuses a job whose settings are not the ones its block assigns."""
    if arm is None:
        raise SystemExit(f"refused: arm {args.arm!r} is not in block {args.block}")
    if (args.alpha, args.attack) not in spec["cells"]:
        raise SystemExit(f"refused: cell {(args.alpha, args.attack)} is not in block {args.block}")
    stated = {"global_lr": (args.global_lr, spec["factor"]), "server_steps": (args.server_steps, spec["steps"]),
              "root": (args.root, arm["root"])}
    for name, (given, assigned) in stated.items():
        if given is not None and given != assigned:
            raise SystemExit(f"refused: {name}={given} but block {args.block} assigns {assigned}")
    if args.seed in CONFIRMATORY:
        if (args.arm, args.alpha, args.attack, args.seed) not in job_keys(args.block):
            raise SystemExit("refused: this job is not in the block's list")
        if args.rounds != ROUNDS:
            raise SystemExit(f"refused: R={args.rounds} on a confirmatory seed; the block assigns {ROUNDS}")
        if args.out != OUT:
            raise SystemExit("refused: confirmatory jobs write only to the campaign's directory")
        if not os.path.exists(FROZEN_HASH):
            raise SystemExit("refused: the preregistration is not frozen")
        with open(FROZEN_HASH) as f:
            frozen = f.read().split()[0]
        if frozen != code_hash():
            raise SystemExit("refused: the code differs from the frozen hash")
        check_environment()


def run_hardened(rule, alpha, attack, seed, rounds, spec, participants, root):
    """A fixed rule through Layers 1, 2 and 4, built like exp10's Random system."""
    from src.defense.hardening import legacy_dp_epsilon_for
    from src.experiments.exp1_baseline import FixedActionSelector, GroundTruthClassifierStub
    from src.fl.federated_learner import _CNNModel
    from src.fl.gradf_learner import GRADFFederatedLearner
    import tensorflow as tf

    shape = (32, 32, 3)
    n_params = _CNNModel(int(np.prod(shape)), input_shape=shape, n_classes=10).params.size
    run_seed = zlib.crc32(f"{seed}|{alpha}|{attack}|{rule}_hardened".encode())
    np.random.seed(run_seed)
    tf.keras.utils.set_random_seed(run_seed)
    learner = GRADFFederatedLearner(
        aggregation="fedavg", classifier=GroundTruthClassifierStub(), selector=FixedActionSelector(rule),
        attack_labels=["none", attack], n_rounds=rounds, n_classes=10, attack_type=attack,
        byzantine_ids=[0, 1], seed=seed, model_type="cnn", input_shape=shape, local_epochs=EPOCHS,
        local_lr=LOCAL_LR, server_steps=spec["steps"], global_lr=spec["factor"],
        dp_epsilon=legacy_dp_epsilon_for(n_params), magnitude_threshold=TAU_CNN, server_rng_seed=run_seed + 1,
    )
    learner.train(participants, root_data=root, verbose=False)


def run_job(block, arm_name, alpha, attack, seed, rounds):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    import src.defense.aggregation_methods  # noqa: F401
    from src.defense.hardening import HardeningPipeline
    from src.experiments import exp9_dominance_grid as exp9
    from src.experiments import exp10_selector_comparison as exp10
    from src.fl.federated_learner import FederatedLearner
    from src.utils.data_loader import load_dataset_participants
    from src.xai.audit_trail import AuditTrail

    spec, arm = BLOCKS[block], BLOCKS[block]["arms"][arm_name]
    AuditTrail.add = lambda self, *a, **k: None   # no shared file across jobs; nothing here reads it

    trains, layers = [], []
    train = FederatedLearner.train

    def recording_train(self, participants, root_data=None, verbose=True, **kw):
        history = train(self, participants, root_data=root_data, verbose=verbose, **kw)
        trains.append({"cls": type(self).__name__, "global_lr": self.global_lr, "server_steps": self.server_steps,
                       "n_rounds": self.n_rounds, "root": None if root_data is None else len(root_data["y"]),
                       "accuracy": [float(r.global_accuracy) for r in history],
                       "params_sha256": hashlib.sha256(np.asarray(self._global_params).tobytes()).hexdigest()})
        return history
    FederatedLearner.train = recording_train

    pipeline = HardeningPipeline.full_pipeline

    def recording_pipeline(self, hospital_ids, *a, **k):
        delta, results = pipeline(self, hospital_ids, *a, **k)
        layers.append({"strategy": results.get("selected_strategy"), "accepted": bool(results.get("round_accepted")),
                       "layer1_rejected": [i for i, h in enumerate(hospital_ids)
                                           if results.get(f"layer1_{h}") != "Passed validation"],
                       "old_accuracy": results.get("old_accuracy"), "new_accuracy": results.get("new_accuracy")})
        return delta, results
    HardeningPipeline.full_pipeline = recording_pipeline

    common = dict(dataset="cifar10", alphas=[alpha], attack_types=[attack], n_clients=10, n_rounds=rounds,
                  byzantine_fraction=0.2, n_classes=10, seed=seed, root_size=arm["root"], seed_rng=True,
                  model_type="cnn", local_epochs=EPOCHS, local_lr=LOCAL_LR, server_steps=spec["steps"],
                  global_lr=spec["factor"])
    if arm["kind"] == "fixed":
        exp9.run_dominance_grid(strategies=[arm["rule"]], **common)
    elif arm["kind"] == "system":
        exp10.run_selector_comparison_grid(systems=[arm["system"]], variant=arm["variant"],
                                           adaaggrl_feature_extractor=arm["extractor"],
                                           magnitude_threshold=TAU_CNN, **common)
    else:
        participants, root = load_dataset_participants("cifar10", exp9._split_name(alpha), 10,
                                                       root_size=arm["root"], root_seed=seed)
        run_hardened(arm["rule"], alpha, attack, seed, rounds, spec, participants, root)

    last = trains[-1]
    for t in trains:
        if t["global_lr"] != spec["factor"]:
            raise SystemExit(f"guard: {t['cls']} ran at global_lr={t['global_lr']}")
        if t["server_steps"] != spec["steps"]:
            raise SystemExit(f"guard: {t['cls']} ran with server_steps={t['server_steps']}")
    if last["root"] != arm["root"] or last["n_rounds"] != rounds or len(last["accuracy"]) != rounds:
        raise SystemExit(f"guard: the run does not match the block: {last['root']}, {last['n_rounds']}")
    return last, trains[:-1], layers


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--block", required=True, choices=sorted(BLOCKS))
    r.add_argument("--arm", required=True)
    r.add_argument("--alpha", type=float, required=True)
    r.add_argument("--attack", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--rounds", type=int, default=ROUNDS)
    r.add_argument("--out", default=OUT)
    r.add_argument("--global_lr", type=float, default=None)
    r.add_argument("--server_steps", default=None)
    r.add_argument("--root", type=int, default=None)
    li = sub.add_parser("list")
    li.add_argument("--block", required=True, choices=sorted(BLOCKS))
    sub.add_parser("hash")
    args = ap.parse_args()

    if args.cmd == "hash":
        print(code_hash())
        return
    if args.cmd == "list":
        for arm, alpha, attack, seed in job_keys(args.block):
            print(f"--block {args.block} --arm {arm} --alpha {alpha} --attack {attack} --seed {seed}")
        return

    spec = BLOCKS[args.block]
    guard(args, spec, spec["arms"].get(args.arm))
    path = f"{args.out}/{args.block}/raw/{args.arm}_a{args.alpha}_{args.attack}_s{args.seed}.json"
    if os.path.exists(path):
        print(f"{path} already exists, skipping")
        return
    start = time.time()
    last, pretraining, layers = run_job(args.block, args.arm, args.alpha, args.attack, args.seed, args.rounds)
    curve = last.pop("accuracy")
    result = {"block": args.block, "arm": args.arm, "alpha": args.alpha, "attack": args.attack, "seed": args.seed,
              "global_lr": spec["factor"], "server_steps": spec["steps"], "root": spec["arms"][args.arm]["root"],
              "rounds": args.rounds, "accuracy": curve, "final": curve[-1],
              "primary_trailing5": float(np.mean(curve[-5:])), "learner": last, "pretraining": pretraining,
              "layers": layers, "machine": socket.gethostname(), "platform": platform.processor(),
              "code_sha256": code_hash(), "seconds": time.time() - start}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
