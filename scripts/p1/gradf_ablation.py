"""
One job of the GRADF layer ablation (results/p1_parity/gradf_ablation/PLAN.md):
one seed, 12 arms, 21 cells, exp10's seeded configuration (root 100,
server_steps=median), with Layers 1, 2 and 4 of the hardening pipeline switched
on or off. No arm needs pretraining.

Usage: venv/bin/python scripts/p1/gradf_ablation.py --seed 42 --out FILE
       venv/bin/python scripts/p1/gradf_ablation.py --check OUT_TXT
"""

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from src.defense.hardening import HardeningPipeline  # noqa: E402
from src.experiments import exp10_selector_comparison as exp10  # noqa: E402
from src.experiments.exp1_baseline import FixedActionSelector, GroundTruthClassifierStub, RandomSelector  # noqa: E402
from src.fl.gradf_learner import GRADFFederatedLearner  # noqa: E402
from src.xai.audit_trail import AuditTrail  # noqa: E402
from src.xai.explanation_generator import XAIExplainer  # noqa: E402

N_CLIENTS, N_ROUNDS, N_CLASSES = 10, 15, 10
BYZANTINE_IDS = [0, 1]
ALPHAS = [0.5, 0.1, 0.05]
ACTIONS = exp10.RLDefenseSelector().actions

# arm: (rule or "random", layer 1, layer 2, layer 4)
ARMS = {
    "F000": ("fltrust", False, False, False),
    "F100": ("fltrust", True, False, False),
    "F010": ("fltrust", False, True, False),
    "F001": ("fltrust", False, False, True),
    "F110": ("fltrust", True, True, False),
    "F101": ("fltrust", True, False, True),
    "F011": ("fltrust", False, True, True),
    "F111": ("fltrust", True, True, True),
    "M111": ("median", True, True, True),
    "T111": ("trimmed_mean", True, True, True),
    "K111": ("krum", True, True, True),
    "R000": ("random", False, False, False),
}


class AblatedPipeline(HardeningPipeline):
    """HardeningPipeline with Layers 1, 2 and 4 switchable. A layer that is off
    accepts everything and draws no random numbers."""

    def __init__(self, *args, layer1=True, layer2=True, layer4=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.switches = (layer1, layer2, layer4)
        self.rounds = []

    def layer1_input_validation(self, gradient, threshold=None):
        if not self.switches[0]:
            return True, "Passed validation"
        return super().layer1_input_validation(gradient, threshold)

    def layer2_gradient_sanitization(self, gradient, epsilon=None, clipping=None):
        if not self.switches[1]:
            return gradient
        return super().layer2_gradient_sanitization(gradient, epsilon, clipping)

    def layer4_model_validation(self, old_accuracy, new_accuracy):
        if not self.switches[2]:
            return True, "Layer 4 off"
        return super().layer4_model_validation(old_accuracy, new_accuracy)

    def full_pipeline(self, hospital_ids, *args, **kwargs):
        delta, results = super().full_pipeline(hospital_ids, *args, **kwargs)
        self.rounds.append({
            "strategy": results.get("selected_strategy"),
            "accepted": bool(results.get("round_accepted")),
            "layer1_rejected": [i for i, h in enumerate(hospital_ids)
                                if results.get(f"layer1_{h}") != "Passed validation"],
            "layer3_error": results.get("layer3_error"),
            "old_accuracy": results.get("old_accuracy"),
            "new_accuracy": results.get("new_accuracy"),
        })
        return delta, results


def run_arm(rule, layer1, layer2, layer4, seed, alpha, attack_type, participants, root_data):
    """Returns (final accuracy, per-round log). Same construction as exp10's Random system."""
    if rule == "random":
        selector = RandomSelector(ACTIONS, seed=seed)
    else:
        selector = FixedActionSelector(rule)
    classifier = GroundTruthClassifierStub()
    run_seed = zlib.crc32(f"{seed}|{alpha}|{attack_type}|{rule}".encode())
    np.random.seed(run_seed)
    learner = GRADFFederatedLearner(
        aggregation="fedavg", classifier=classifier, selector=selector,
        explainer=XAIExplainer(AuditTrail(os.devnull)),
        attack_labels=["none", attack_type], n_rounds=N_ROUNDS, n_classes=N_CLASSES,
        attack_type=attack_type, byzantine_ids=BYZANTINE_IDS, seed=seed,
        local_epochs=5, local_lr=0.01, server_steps="median", server_rng_seed=run_seed + 1,
    )
    old = learner.hardening
    learner.hardening = AblatedPipeline(
        classifier, selector, magnitude_threshold=old.magnitude_threshold,
        accuracy_tolerance=old.accuracy_tolerance, dp_epsilon=old.dp_epsilon,
        dp_clipping=old.dp_clipping, layer1=layer1, layer2=layer2, layer4=layer4,
    )
    history = learner.train(participants, root_data=root_data, verbose=False)
    for row, res in zip(learner.hardening.rounds, history):
        row["global_accuracy"] = res.global_accuracy
    return history[-1].global_accuracy, learner.hardening.rounds


def run_job(seed, out):
    start = time.time()
    accuracy, rounds = [], []
    for alpha in ALPHAS:
        participants, root_data = exp10.load_dataset_participants(
            "mnist", exp10._split_name(alpha), N_CLIENTS, root_size=100, root_seed=seed,
        )
        for attack_type in exp10.INFORMED_ATTACKS + exp10.BLIND_ATTACKS:
            for arm, spec in ARMS.items():
                acc, log = run_arm(*spec, seed, alpha, attack_type, participants, root_data)
                cell = {"alpha": alpha, "attack_type": attack_type, "arm": arm}
                accuracy.append({**cell, "accuracy": acc})
                rounds += [{**cell, "round": i, **r} for i, r in enumerate(log)]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"seed": seed, "seconds": time.time() - start, "accuracy": accuracy, "rounds": rounds}, f)
    os.replace(tmp, out)


def check(out):
    seed, alpha = 42, 0.05
    participants, root_data = exp10.load_dataset_participants(
        "mnist", exp10._split_name(alpha), N_CLIENTS, root_size=100, root_seed=seed,
    )
    lines, ok = [], True

    def report(name, passed, detail=""):
        nonlocal ok
        ok = ok and bool(passed)
        lines.append(f"{'PASS' if passed else 'FAIL'} {name} {detail}")

    def run(arm_spec, attack):
        return run_arm(*arm_spec, seed, alpha, attack, participants, root_data)

    ref = pd.read_csv("results/p1_parity/mnist_rq3/raw/variantb_seed42/"
                      "exp10_selector_comparison_variantb_seeded_stepsmedian_raw.csv")
    ref = ref[(ref.system == "Random") & (ref.alpha == alpha)].set_index("attack_type").accuracy
    for attack in ("sign_flipping", "gaussian_noise"):
        t = time.time()
        acc, _ = run(("random", True, True, True), attack)
        report(f"check1 all layers on reproduce Random, {attack}", abs(acc - ref[attack]) < 1e-12,
               f"{acc:.6f} vs {ref[attack]:.6f}; {time.time() - t:.1f} s")

    _, on = run(("fltrust", True, True, True), "gaussian_noise")
    _, off = run(("fltrust", False, True, True), "gaussian_noise")
    report("check2 layer 1 on rejects, off rejects none", any(r["layer1_rejected"] for r in on)
           and not any(r["layer1_rejected"] for r in off))

    _, on = run(("random", True, True, True), "sign_flipping")
    _, off = run(("random", True, True, False), "sign_flipping")
    report("check2 layer 4 on rejects, off rejects none", not all(r["accepted"] for r in on)
           and all(r["accepted"] for r in off))

    pipe = AblatedPipeline(None, None, layer2=False)
    x = np.array([3.0, -3.0, 0.5])
    state = np.random.get_state()[1].copy()
    same = pipe.layer2_gradient_sanitization(x) is x
    report("check2 layer 2 off returns the update and draws nothing",
           same and np.array_equal(state, np.random.get_state()[1]))

    a0, log = run(("fltrust", True, True, True), "sign_flipping")
    report("check3 fixed selector votes FLTrust in every round", {r["strategy"] for r in log} == {"fltrust"})
    a1, _ = run(("fltrust", True, True, True), "sign_flipping")
    report("check determinism, F111 twice", a0 == a1, f"{a0:.6f} {a1:.6f}")

    lines.append("CHECKS_PASS" if ok else "CHECKS_FAIL")
    text = "\n".join(lines) + "\n"
    with open(out, "w") as f:
        f.write(text)
    print(text)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    ap.add_argument("--check")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check(args.check) else 1)
    run_job(args.seed, args.out)


if __name__ == "__main__":
    main()
