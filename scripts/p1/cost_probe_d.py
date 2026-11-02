"""
E0b part D cost probe (results/p1_cifar10/e0b/PLAN.md, Sec. 5): seconds per
round of every campaign system on the CIFAR-10 CNN, and the wall time of the
pretraining shared by GRADF, the attack-aware reference and FedStrategist,
measured under the campaign's load (12 processes x 1 thread, CPU).

Costs only: the accuracies are not saved. alpha 0.5, sign_flipping, seed 0
(outside the confirmatory block), root 100, 2 local epochs, deterministic ops.
The server's root fit uses step parity with the clients (server_steps =
"median", results/p1_parity/PLAN.md), the campaign's setting, so the cost of
g0 at every round is included; FLTrust is timed as its own fixed rule.
Each process runs the same sequence: pretraining (train_gradf_models, as in
exp10_selector_comparison), then for each system one warm-up round and
3 timed rounds.

Usage:
  bash scripts/p1/run_cost_probe_d.sh [n_procs=12]          (launcher)
  python scripts/p1/cost_probe_d.py --index I               (one process)
  python scripts/p1/cost_probe_d.py --aggregate             (writes costs.json)
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())

OUT = "results/p1_cifar10/e0b/cost_probe"
COSTS = "results/p1_cifar10/e0b/costs.json"
SHAPE = (32, 32, 3)
ALPHA, ATTACK, SEED, TIMED = 0.5, "sign_flipping", 0, 3
GRID_RUNS = 18 * 10            # cells x seeds of the full common confirmatory grid
PROCS = 12


def probe(index):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(SEED)

    import src.defense.aggregation_methods  # noqa: F401
    from src.defense.hardening import legacy_dp_epsilon_for
    from src.defense.rl_selector import RLDefenseSelector
    from src.experiments.exp1_baseline import (GroundTruthClassifierStub, GroundTruthOracleLearner,
                                               OracleSelector, RandomSelector, train_gradf_models)
    from src.experiments.exp10_selector_comparison import AdaAggRLGridLearner, FedStrategistGridLearner
    from src.experiments.exp9_dominance_grid import _split_name
    from src.fl.attacked_learner import AttackedFederatedLearner
    from src.fl.federated_learner import _CNNModel
    from src.fl.gradf_learner import GRADFFederatedLearner
    from src.utils.data_loader import load_dataset_participants

    participants, root = load_dataset_participants("cifar10", _split_name(ALPHA), 10,
                                                   root_size=100, root_seed=SEED)
    model_kwargs = {"local_epochs": 2, "local_lr": 0.01, "model_type": "cnn", "input_shape": SHAPE,
                    "server_steps": "median"}
    n_params = _CNNModel(int(np.prod(SHAPE)), input_shape=SHAPE, n_classes=10).params.size
    gradf_kwargs = dict(model_kwargs, dp_epsilon=legacy_dp_epsilon_for(n_params))
    labels = ["none", ATTACK]
    common = dict(n_classes=10, attack_type=ATTACK, byzantine_ids=[0, 1], seed=SEED)
    res = {"index": index, "alpha": ALPHA, "attack": ATTACK, "seed": SEED, "timed_rounds": TIMED,
           "s_round": {}}

    t0 = time.perf_counter()
    classifier, selector, _, oracle_map = train_gradf_models(
        participants, ATTACK, 0.2, 10, n_rounds=8, seed=SEED, model_kwargs=model_kwargs)
    res["pretraining_s"] = time.perf_counter() - t0
    print(index, "pretraining", f"{res['pretraining_s']:.0f}s", flush=True)

    makers = {
        "fixed": lambda r: AttackedFederatedLearner(n_rounds=r, aggregation="trimmed_mean",
                                                    **common, **model_kwargs),
        "fltrust": lambda r: AttackedFederatedLearner(n_rounds=r, aggregation="fltrust",
                                                      **common, **model_kwargs),
        "random": lambda r: GRADFFederatedLearner(
            aggregation="fedavg", classifier=GroundTruthClassifierStub(),
            selector=RandomSelector(RLDefenseSelector().actions, seed=SEED),
            attack_labels=labels, n_rounds=r, **common, **gradf_kwargs),
        "attack_aware": lambda r: GroundTruthOracleLearner(
            aggregation="fedavg", classifier=GroundTruthClassifierStub(),
            selector=OracleSelector(oracle_map), attack_labels=labels, n_rounds=r,
            **common, **gradf_kwargs),
        "gradf": lambda r: GRADFFederatedLearner(
            aggregation="fedavg", classifier=classifier, selector=selector,
            attack_labels=labels, n_rounds=r, **common, **gradf_kwargs),
        "fedstrategist": lambda r: FedStrategistGridLearner(
            n_rounds=r, variant="b", shared_classifier=classifier, n_labels=len(labels),
            **common, **model_kwargs),
        "adaaggrl": lambda r: AdaAggRLGridLearner(
            n_rounds=r, feature_extractor="random", dataset="cifar10",   # input_shape via model_kwargs
            **common, **model_kwargs),
    }
    for name, make in makers.items():
        make(1).train(participants, root_data=root, verbose=False)        # warm-up
        learner = make(TIMED)
        t0 = time.perf_counter()
        learner.train(participants, root_data=root, verbose=False)
        res["s_round"][name] = (time.perf_counter() - t0) / TIMED
        print(index, name, f"{res['s_round'][name]:.1f} s/round", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/probe_{index}.json", "w") as f:
        json.dump(res, f, indent=1)


def aggregate():
    probes = [json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/probe_*.json"))]
    if len(probes) != PROCS:
        raise SystemExit(f"expected {PROCS} probe files, found {len(probes)}")
    names = probes[0]["s_round"].keys()
    costs = {
        "s_round": {n: float(np.mean([p["s_round"][n] for p in probes])) for n in names},
        "pretraining_s_per_run": float(np.mean([p["pretraining_s"] for p in probes])),
        "n_processes": len(probes),
        "source": "scripts/p1/cost_probe_d.py",
    }
    # one pretraining per (cell, seed) of the common grid, shared by GRADF, attack-aware and FedStrategist
    costs["pretraining_h_full_grid"] = GRID_RUNS * costs["pretraining_s_per_run"] / PROCS / 3600
    with open(COSTS, "w") as f:
        json.dump(costs, f, indent=1)
    print(json.dumps(costs, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        probe(args.index)


if __name__ == "__main__":
    main()
