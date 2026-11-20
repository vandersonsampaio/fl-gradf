"""
E4a, temporal headroom on MNIST (results/p1_e4a/PLAN.md): runs the phase-wise
greedy reference and the fixed rules over one attack sequence, and the checks of
the plan's Sec. 4.

  run    one seed: 5 arms x 2 scenarios, per-round accuracy, attack and rule
  check  the four checks; prints PASS or FAIL for each and exits non-zero on a failure

Usage: venv/bin/python scripts/p1/e4a_phases.py run --seed 42 --out DIR [--phase_len 15]
       venv/bin/python scripts/p1/e4a_phases.py check [--seed 42] [--phase_len 3] [--no_regression]
"""

import argparse
import glob
import os
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from src.experiments.exp9_dominance_grid import _split_name, round_accuracies  # noqa: E402
from src.classification.attack_simulator import AttackSimulator  # noqa: E402
from src.fl.attacked_learner import AttackedFederatedLearner, InformedAttackedFederatedLearner  # noqa: E402
from src.fl.federated_learner import _STRATEGIES  # noqa: E402
from src.utils.data_loader import load_dataset_participants  # noqa: E402

ATTACKS = ["sign_flipping", "gaussian_noise", "trim_attack"]
GREEDY_RULES = {0.1: ["fltrust", "trimmed_mean", "fltrust"], 0.5: ["krum", "trimmed_mean", "trimmed_mean"]}
FIXED = ["fltrust", "median", "trimmed_mean", "krum"]
GREEDY = "greedy"
ROOT = 100
MODEL = {"local_epochs": 5, "local_lr": 0.01, "server_steps": "median"}
BYZANTINE = [0, 1]
FIXED_RAW = "results/p1_parity/mnist_trailing/mnist_fixed/raw/root100_seed{}/*_stepsmedian_raw.csv"


class PhaseLearner(InformedAttackedFederatedLearner):
    """Takes the attack and the aggregation rule of each round from `schedule`, a list of
    (attack, rule) pairs. The global model carries over when either changes."""

    def __init__(self, *args, schedule, **kwargs):
        self.schedule = schedule
        attack, rule = schedule[0]
        super().__init__(*args, n_rounds=len(schedule), attack_type=attack, aggregation=rule, **kwargs)

    def _run_round(self, round_num, participants, root_data):
        self.attack_type, self.aggregation = self.schedule[round_num - 1]
        self._strategy = _STRATEGIES[self.aggregation]()
        cls = InformedAttackedFederatedLearner if self.attack_type in AttackSimulator.INFORMED_ATTACK_TYPES \
            else AttackedFederatedLearner
        return cls._run_round(self, round_num, participants, root_data)


def schedule_of(alpha, arm, phase_len, attacks=ATTACKS):
    rules = GREEDY_RULES[alpha] if arm == GREEDY else [arm] * len(attacks)
    return [(a, r) for a, r in zip(attacks, rules) for _ in range(phase_len)]


def train(participants, root_data, schedule, seed, seed_string):
    run_seed = zlib.crc32(seed_string.encode())
    np.random.seed(run_seed)
    learner = PhaseLearner(schedule=schedule, n_classes=10, byzantine_ids=BYZANTINE, seed=seed,
                           server_rng_seed=run_seed + 1, **MODEL)
    return learner.train(participants, root_data=root_data, verbose=False)


def load(alpha, seed):
    return load_dataset_participants("mnist", _split_name(alpha), 10, root_size=ROOT, root_seed=seed)


def run(seed, phase_len):
    rows = []
    for alpha in GREEDY_RULES:
        participants, root_data = load(alpha, seed)
        for arm in [GREEDY] + FIXED:
            schedule = schedule_of(alpha, arm, phase_len)
            history = train(participants, root_data, schedule, seed, f"{seed}|{alpha}|e4a")
            for r, (attack, rule) in zip(history, schedule):
                rows.append({"alpha": alpha, "arm": arm, "seed": seed, "round": r.round_num,
                             "phase": (r.round_num - 1) // phase_len + 1, "attack": attack, "rule": rule,
                             "accuracy": r.global_accuracy})
    return pd.DataFrame(rows)


def check_regression(seed):
    """Check 1: one attack and one rule for 15 rounds, under exp9's seed string, equal mnist_fixed."""
    ref = pd.read_csv(glob.glob(FIXED_RAW.format(seed))[0])
    alpha, ok = 0.1, True
    participants, root_data = load(alpha, seed)
    for attack in ("sign_flipping", "trim_attack"):
        for rule in ("fltrust", "trimmed_mean"):
            history = train(participants, root_data, [(attack, rule)] * 15, seed, f"{seed}|{alpha}|{attack}|{rule}")
            acc, trailing = round_accuracies(history)
            row = ref[(ref.alpha == alpha) & (ref.attack_type == attack) & (ref.strategy == rule)].iloc[0]
            same = acc == row.accuracy and trailing == row.accuracy_trailing5
            ok &= same
            print(f"  {attack} {rule}: {acc} {trailing} against {row.accuracy} {row.accuracy_trailing5}: {same}")
    return ok


def check_switch(seed, phase_len):
    """Check 2: G follows fixed FLTrust during its first phase and leaves it after the switch."""
    alpha = 0.1
    participants, root_data = load(alpha, seed)
    string = f"{seed}|{alpha}|e4a"
    g = [r.global_accuracy for r in train(participants, root_data, schedule_of(alpha, GREEDY, phase_len), seed, string)]
    f = [r.global_accuracy for r in train(participants, root_data, schedule_of(alpha, "fltrust", phase_len), seed, string)]
    first, second = g[:phase_len] == f[:phase_len], g[phase_len:2 * phase_len] != f[phase_len:2 * phase_len]
    print(f"  equal in the first phase: {first}; different in the second: {second}")
    return first and second, g


def check_repeat(seed, phase_len, first):
    """Check 3: a second run of G gives the same per-round accuracy."""
    alpha = 0.1
    participants, root_data = load(alpha, seed)
    history = train(participants, root_data, schedule_of(alpha, GREEDY, phase_len), seed, f"{seed}|{alpha}|e4a")
    return [r.global_accuracy for r in history] == first


def check_schedule(phase_len):
    """Check 4: every round's attack and rule follow the plan's Sec. 2, in what `run` saves."""
    expected = {0.1: ["fltrust", "trimmed_mean", "fltrust"], 0.5: ["krum", "trimmed_mean", "trimmed_mean"]}
    for alpha, rules in expected.items():
        s = schedule_of(alpha, GREEDY, phase_len)
        for i, (attack, rule) in enumerate(s):
            if (attack, rule) != (ATTACKS[i // phase_len], rules[i // phase_len]):
                return False
        for arm in FIXED:
            s = schedule_of(alpha, arm, phase_len)
            if [r for _, r in s] != [arm] * len(s) or [a for a, _ in s] != [ATTACKS[i // phase_len] for i in range(len(s))]:
                return False
    return True


def checks(seed, phase_len, regression):
    results = {}
    if regression:
        print("Check 1, regression:")
        results["regression"] = check_regression(seed)
    print("Check 2, switch:")
    results["switch"], g = check_switch(seed, phase_len)
    print("Check 3, repetition:")
    results["repetition"] = check_repeat(seed, phase_len, g)
    print("Check 4, schedule:")
    results["schedule"] = check_schedule(phase_len)
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    return all(results.values())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--phase_len", type=int, default=15)
    c = sub.add_parser("check")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--phase_len", type=int, default=3)
    c.add_argument("--no_regression", action="store_true")
    args = ap.parse_args()
    if args.cmd == "run":
        os.makedirs(args.out, exist_ok=True)
        run(args.seed, args.phase_len).to_csv(f"{args.out}/per_round.csv", index=False)
    elif not checks(args.seed, args.phase_len, not args.no_regression):
        sys.exit(1)


if __name__ == "__main__":
    main()
