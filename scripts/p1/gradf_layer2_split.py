"""
One job of the Layer 2 split (results/p1_parity/gradf_layer2/PLAN.md): one seed,
4 arms, 21 cells, the configuration of the GRADF ablation with fixed FLTrust and
Layers 1 and 4 off. Layer 2 runs with clipping and noise switched separately, and
each round logs the norms that show the size of the server's step.

Usage: venv/bin/python scripts/p1/gradf_layer2_split.py --seed 42 --out FILE
       venv/bin/python scripts/p1/gradf_layer2_split.py --check OUT_TXT
"""

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))

from gradf_ablation import ALPHAS, BYZANTINE_IDS, N_CLASSES, N_CLIENTS, N_ROUNDS, AblatedPipeline  # noqa: E402
from src.experiments import exp10_selector_comparison as exp10  # noqa: E402
from src.experiments.exp1_baseline import FixedActionSelector, GroundTruthClassifierStub  # noqa: E402
from src.fl.gradf_learner import GRADFFederatedLearner  # noqa: E402
from src.xai.audit_trail import AuditTrail  # noqa: E402
from src.xai.explanation_generator import XAIExplainer  # noqa: E402

RULE = "fltrust"
ABLATION_RAW = "results/p1_parity/gradf_ablation/raw"

# arm: (clipping, noise)
ARMS = {
    "F000": (False, False),
    "C010": (True, False),
    "N010": (False, True),
    "F010": (True, True),
}


class SplitPipeline(AblatedPipeline):
    """Layers 1 and 4 off; Layer 2 with clipping and noise switched separately."""

    def __init__(self, *args, clip=True, noise=True, **kwargs):
        super().__init__(*args, layer1=False, layer2=True, layer4=False, **kwargs)
        self.clip, self.noise = clip, noise
        self.clipped_coords = self.total_coords = 0

    def layer2_gradient_sanitization(self, gradient, epsilon=None, clipping=None):
        epsilon = self.dp_epsilon if epsilon is None else epsilon
        clipping = self.dp_clipping if clipping is None else clipping
        self.clipped_coords += int((np.abs(gradient) > clipping).sum())
        self.total_coords += gradient.size
        out = np.clip(gradient, -clipping, clipping) if self.clip else gradient
        if self.noise:
            out = out + np.random.laplace(0, 1 / epsilon, size=gradient.shape)
        return out

    def full_pipeline(self, hospital_ids, updates, *args, server_update=None, **kwargs):
        self.clipped_coords = self.total_coords = 0
        delta, results = super().full_pipeline(hospital_ids, updates, *args, server_update=server_update, **kwargs)
        g0 = np.linalg.norm(server_update)
        trust = results.get("layer3_metadata", {}).get("trust_scores", [])
        honest = [i for i in range(len(updates)) if i not in BYZANTINE_IDS]
        self.rounds[-1].update({
            "g0_norm": float(g0),
            "client_norm_honest": float(np.mean([np.linalg.norm(updates[i]) for i in honest])),
            "delta_norm": float(np.linalg.norm(delta)),
            "step_along_g0": float(np.dot(delta, server_update) / g0 ** 2),
            "trust_honest": float(np.mean([trust[i] for i in honest])),
            "trust_byzantine": float(np.mean([trust[i] for i in BYZANTINE_IDS])),
            "clipped_fraction": self.clipped_coords / self.total_coords,
        })
        return delta, results


def run_arm(clip, noise, seed, alpha, attack_type, participants, root_data):
    """Returns (final accuracy, per-round log), constructed as run_arm of gradf_ablation.py."""
    selector, classifier = FixedActionSelector(RULE), GroundTruthClassifierStub()
    run_seed = zlib.crc32(f"{seed}|{alpha}|{attack_type}|{RULE}".encode())
    np.random.seed(run_seed)
    learner = GRADFFederatedLearner(
        aggregation="fedavg", classifier=classifier, selector=selector,
        explainer=XAIExplainer(AuditTrail(os.devnull)),
        attack_labels=["none", attack_type], n_rounds=N_ROUNDS, n_classes=N_CLASSES,
        attack_type=attack_type, byzantine_ids=BYZANTINE_IDS, seed=seed,
        local_epochs=5, local_lr=0.01, server_steps="median", server_rng_seed=run_seed + 1,
    )
    old = learner.hardening
    learner.hardening = SplitPipeline(
        classifier, selector, magnitude_threshold=old.magnitude_threshold,
        accuracy_tolerance=old.accuracy_tolerance, dp_epsilon=old.dp_epsilon,
        dp_clipping=old.dp_clipping, clip=clip, noise=noise,
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
    with open(f"{ABLATION_RAW}/seed{seed}.json") as f:
        ref = {(a["attack_type"], a["arm"]): a["accuracy"] for a in json.load(f)["accuracy"] if a["alpha"] == alpha}
    lines, ok = [], True

    def report(name, passed, detail=""):
        nonlocal ok
        ok = ok and bool(passed)
        lines.append(f"{'PASS' if passed else 'FAIL'} {name} {detail}")

    for attack in ("sign_flipping", "gaussian_noise"):
        logs = {}
        for arm, spec in ARMS.items():
            t = time.time()
            acc, logs[arm] = run_arm(*spec, seed, alpha, attack, participants, root_data)
            if arm in ("F000", "F010"):
                report(f"check1 {arm} reproduces the ablation, {attack}", acc == ref[(attack, arm)],
                       f"{acc:.6f} vs {ref[(attack, arm)]:.6f}; {time.time() - t:.1f} s")
            lines.append(f"INFO {arm} {attack}: accuracy {acc:.4f}, g0 {logs[arm][-1]['g0_norm']:.3f}, "
                         f"client {logs[arm][-1]['client_norm_honest']:.3f}, delta {logs[arm][-1]['delta_norm']:.3f}, "
                         f"step along g0 {logs[arm][-1]['step_along_g0']:.3f}, "
                         f"clipped {np.mean([r['clipped_fraction'] for r in logs[arm]]):.2e}")
        report(f"check3 fields are written, {attack}", all(r["strategy"] == RULE and r["accepted"] for a in logs.values() for r in a))

    x = np.array([3.0, -3.0, 0.5])
    args = (None, None)
    state = np.random.get_state()[1].copy()
    a = SplitPipeline(*args, clip=True, noise=False).layer2_gradient_sanitization(x, 50.0, 2.0)
    report("check2 clipping alone clips and draws nothing", np.array_equal(a, [2.0, -2.0, 0.5])
           and np.array_equal(state, np.random.get_state()[1]))
    np.random.seed(0)
    b = SplitPipeline(*args, clip=False, noise=True).layer2_gradient_sanitization(x, 50.0, 2.0)
    np.random.seed(0)
    report("check2 noise alone adds the same draw and does not clip", np.array_equal(b, x + np.random.laplace(0, 1 / 50.0, size=3)))

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
