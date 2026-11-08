"""
One execution of the GRADF repetition (results/p1_parity/gradf_repeat/PLAN.md):
exp10's seeded run, variant b, root 100, server_steps=median, with the systems
Random and GRADF only. Each system reseeds itself, so leaving the others out
changes neither stream.

Besides the accuracies it records, per round and without touching any RNG, what
the hardening pipeline decided: the voted rule, whether the round was accepted
(Layer 4), and which clients Layer 1 rejected.

Usage: venv/bin/python scripts/p1/gradf_repeat.py --seed 42 --execution 1 --out FILE
       [--alphas 0.5 ...] [--attack_types sign_flipping ...]  (checks only)
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from src.defense.hardening import HardeningPipeline  # noqa: E402
from src.experiments import exp10_selector_comparison as exp10  # noqa: E402
from src.fl.gradf_learner import GRADFFederatedLearner  # noqa: E402

CONTEXT = {"alpha": None, "run": None}
ROUNDS = []


def instrument():
    split_name = exp10._split_name
    train = GRADFFederatedLearner.train
    pipeline = HardeningPipeline.full_pipeline

    def tracked_split(alpha):
        CONTEXT["alpha"] = alpha
        return split_name(alpha)

    def tracked_train(self, *args, **kwargs):
        system = "GRADF" if type(self.selector).__name__ == "RLDefenseSelector" else "Random"
        CONTEXT["run"] = {"alpha": CONTEXT["alpha"], "attack_type": self.attack_type, "system": system}
        start = len(ROUNDS)
        history = train(self, *args, **kwargs)
        for row, res in zip(ROUNDS[start:], history):
            row["global_accuracy"] = res.global_accuracy
        return history

    def tracked_pipeline(self, hospital_ids, *args, **kwargs):
        delta, results = pipeline(self, hospital_ids, *args, **kwargs)
        ROUNDS.append({
            **CONTEXT["run"],
            "strategy": results.get("selected_strategy"),
            "accepted": bool(results.get("round_accepted")),
            "layer1_rejected": [i for i, h in enumerate(hospital_ids)
                                if results.get(f"layer1_{h}") != "Passed validation"],
            "layer3_error": results.get("layer3_error"),
            "old_accuracy": results.get("old_accuracy"),
            "new_accuracy": results.get("new_accuracy"),
        })
        return delta, results

    exp10._split_name = tracked_split
    GRADFFederatedLearner.train = tracked_train
    HardeningPipeline.full_pipeline = tracked_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--execution", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=None)
    ap.add_argument("--attack_types", nargs="+", default=None)
    args = ap.parse_args()

    instrument()
    start = time.time()
    df = exp10.run_selector_comparison_grid(
        seed=args.seed, alphas=args.alphas, attack_types=args.attack_types, root_size=100,
        systems=["random", "gradf"], variant="b", seed_rng=True, server_steps="median",
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"seed": args.seed, "execution": args.execution, "seconds": time.time() - start,
                   "accuracy": df.to_dict("records"), "rounds": ROUNDS}, f)
    os.replace(tmp, args.out)


if __name__ == "__main__":
    main()
