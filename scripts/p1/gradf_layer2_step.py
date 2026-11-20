"""
One job of the noise-free step arm (results/p1_parity/gradf_layer2/step/ADDENDUM.md):
one seed, 21 cells, fixed FLTrust with Layers 1 and 4 off, no clipping and no noise,
and the aggregate multiplied by a constant factor.

Usage: venv/bin/python scripts/p1/gradf_layer2_step.py --seed 42 --out FILE
       venv/bin/python scripts/p1/gradf_layer2_step.py --check OUT_TXT
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))

import gradf_layer2_split as split  # noqa: E402
from gradf_ablation import ALPHAS, N_CLIENTS  # noqa: E402
from src.experiments import exp10_selector_comparison as exp10  # noqa: E402

FACTOR = 1 / 3
ARM = "S010"
LAYER2_RAW = "results/p1_parity/gradf_layer2/raw"


class ScaledPipeline(split.SplitPipeline):
    """No clipping and no noise; Layer 3's aggregate is multiplied by a constant."""

    factor = FACTOR

    def __init__(self, *args, clip=False, noise=False, **kwargs):
        super().__init__(*args, clip=False, noise=False, **kwargs)

    def layer3_secure_aggregation(self, *args, **kwargs):
        delta, metadata = super().layer3_secure_aggregation(*args, **kwargs)
        return self.factor * delta, metadata


def run_arm(seed, alpha, attack_type, participants, root_data, factor=FACTOR):
    """Same construction as run_arm of gradf_layer2_split.py, through its pipeline hook."""
    original = split.SplitPipeline
    scaled = type("Scaled", (ScaledPipeline,), {"factor": factor})
    split.SplitPipeline = scaled
    try:
        return split.run_arm(False, False, seed, alpha, attack_type, participants, root_data)
    finally:
        split.SplitPipeline = original


def run_job(seed, out):
    start = time.time()
    accuracy, rounds = [], []
    for alpha in ALPHAS:
        participants, root_data = exp10.load_dataset_participants(
            "mnist", exp10._split_name(alpha), N_CLIENTS, root_size=100, root_seed=seed,
        )
        for attack_type in exp10.INFORMED_ATTACKS + exp10.BLIND_ATTACKS:
            acc, log = run_arm(seed, alpha, attack_type, participants, root_data)
            cell = {"alpha": alpha, "attack_type": attack_type, "arm": ARM}
            accuracy.append({**cell, "accuracy": acc})
            rounds += [{**cell, "round": i, **r} for i, r in enumerate(log)]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"seed": seed, "seconds": time.time() - start, "accuracy": accuracy, "rounds": rounds}, f)
    os.replace(tmp, out)


def check(out):
    seed, alpha, attack = 42, 0.05, "sign_flipping"
    participants, root_data = exp10.load_dataset_participants(
        "mnist", exp10._split_name(alpha), N_CLIENTS, root_size=100, root_seed=seed,
    )
    with open(f"{LAYER2_RAW}/seed{seed}.json") as f:
        ref = json.load(f)
    key = (alpha, attack, "F000")
    ref_acc = next(a["accuracy"] for a in ref["accuracy"] if (a["alpha"], a["attack_type"], a["arm"]) == key)
    ref_rounds = [r for r in ref["rounds"] if (r["alpha"], r["attack_type"], r["arm"]) == key]
    lines, ok = [], True

    def report(name, passed, detail=""):
        nonlocal ok
        ok = ok and bool(passed)
        lines.append(f"{'PASS' if passed else 'FAIL'} {name} {detail}")

    acc, _ = run_arm(seed, alpha, attack, participants, root_data, factor=1.0)
    report("check1 factor 1 reproduces F000", acc == ref_acc, f"{acc:.6f} vs {ref_acc:.6f}")
    state = np.random.get_state()[1].copy()
    pipe = ScaledPipeline(None, None)
    x = np.array([3.0, -3.0, 0.5])
    report("check2 Layer 2 returns the update unclipped and draws nothing",
           np.array_equal(pipe.layer2_gradient_sanitization(x), x) and np.array_equal(state, np.random.get_state()[1]))
    t = time.time()
    _, log = run_arm(seed, alpha, attack, participants, root_data)
    ratio = log[0]["step_along_g0"] / ref_rounds[0]["step_along_g0"]
    report("check3 the first round's step is the factor times F000's", abs(ratio - FACTOR) < 1e-9, f"ratio {ratio:.6f}; {time.time() - t:.1f} s")
    report("check3 every round votes FLTrust and is accepted", all(r["strategy"] == split.RULE and r["accepted"] for r in log))

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
