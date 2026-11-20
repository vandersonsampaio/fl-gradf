"""
MNIST hardened fixed rules (results/p1_parity/mnist_trailing/ADDENDUM_3.md).

FLTrust, median, trimmed mean and Krum through GRADF's Layers 1, 2 and 4, built exactly like
exp10's Random system in the MNIST rerun (mnist_rq3: logistic model, R = 15, 5 local epochs,
local_lr 0.01, server_steps = median, root 100, seeded per run), with the random selector
replaced by a fixed one. The run seed is crc32("{seed}|{alpha}|{attack}|{rule}_hardened"), as in
the CIFAR-10 campaign's hardened block.

`--check` runs the Random system through this same path for one (seed, alpha) and compares it
with the stored mnist_rq3 runs: it must match exactly.

Usage: ../fl-fedmed/venv/bin/python scripts/p1/mnist_hardened.py run [--workers 10]
       ../fl-fedmed/venv/bin/python scripts/p1/mnist_hardened.py check --seed 42 --alpha 0.1
"""

import argparse
import glob
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

OUT = "results/p1_parity/mnist_trailing/mnist_hardened"
RULES = ("fltrust", "median", "trimmed_mean", "krum")
SEEDS = range(42, 52)
ALPHAS = (0.5, 0.1, 0.05)


def build(rule_or_random, alpha, attack, seed, participants, root):
    from src.defense.rl_selector import RLDefenseSelector
    from src.experiments.exp1_baseline import FixedActionSelector, GroundTruthClassifierStub, RandomSelector
    from src.experiments.exp9_dominance_grid import round_accuracies
    from src.fl.gradf_learner import GRADFFederatedLearner

    if rule_or_random == "random":
        label, selector = "random", RandomSelector(RLDefenseSelector().actions, seed=seed)
    else:
        label, selector = f"{rule_or_random}_hardened", FixedActionSelector(rule_or_random)
    run_seed = zlib.crc32(f"{seed}|{alpha}|{attack}|{label}".encode())
    np.random.seed(run_seed)
    learner = GRADFFederatedLearner(
        aggregation="fedavg", classifier=GroundTruthClassifierStub(), selector=selector,
        attack_labels=["none", attack], n_rounds=15, n_classes=10, attack_type=attack,
        byzantine_ids=[0, 1], seed=seed, local_epochs=5, local_lr=0.01, server_steps="median",
        server_rng_seed=run_seed + 1,
    )
    final, trailing = round_accuracies(learner.train(participants, root_data=root, verbose=False))
    return final, trailing


def group(args):
    seed, alpha, arms = args
    sys.path.insert(0, os.getcwd())
    from src.experiments.exp9_dominance_grid import BLIND_ATTACKS, INFORMED_ATTACKS, _split_name
    from src.utils.data_loader import load_dataset_participants

    participants, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    rows = []
    for attack in INFORMED_ATTACKS + BLIND_ATTACKS:
        for arm in arms:
            t0 = time.time()
            final, trailing = build(arm, alpha, attack, seed, participants, root)
            rows.append({"alpha": alpha, "attack_type": attack, "strategy": arm, "seed": seed,
                         "accuracy": final, "accuracy_trailing5": trailing, "seconds": time.time() - t0})
    return rows


def run(workers):
    os.makedirs(f"{OUT}/raw", exist_ok=True)
    todo = [(s, a, RULES) for s in SEEDS for a in ALPHAS
            if not os.path.exists(f"{OUT}/raw/s{s}_a{a}.csv")]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for args, rows in zip(todo, ex.map(group, todo)):
            pd.DataFrame(rows).to_csv(f"{OUT}/raw/s{args[0]}_a{args[1]}.csv", index=False)
            print(time.strftime("%F %T"), "DONE", args[0], args[1], len(rows), flush=True)


def check(seed, alpha):
    rows = pd.DataFrame(group((seed, alpha, ("random",))))
    stored = pd.read_csv(glob.glob(f"results/p1_parity/mnist_trailing/mnist_rq3/raw/variantb_seed{seed}/*_raw.csv")[0])
    stored = stored[(stored.system == "Random") & (stored.alpha == alpha)]
    m = rows.merge(stored, on=["alpha", "attack_type"], suffixes=("_new", "_old"))
    d = max((m.accuracy_new - m.accuracy_old).abs().max(), (m.accuracy_trailing5_new - m.accuracy_trailing5_old).abs().max())
    print(f"CHECK Random through this path vs stored mnist_rq3, seed {seed}, alpha {alpha}: "
          f"{len(m)} cells, max |diff| = {d:.3g}, {rows.seconds.mean():.1f} s/run")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=10)
    c = sub.add_parser("check")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    if a.cmd == "run":
        run(a.workers)
    else:
        check(a.seed, a.alpha)


if __name__ == "__main__":
    main()
