"""
Checks before the first confirmatory run of the P1 campaign (PREREGISTRATION, Sec. 8), on seed 0.

  probe --factor F --out JSON    runs small cases with the code of the current directory and records,
                                 per learner trained, its settings, the hash of its final params,
                                 its accuracy per round and the norm of its first step
  compare OLD NEW THIRD          checks 1, 3 and 4 from three probes
  pilot                          check 2: 1/3 reproduces cifar_step_pilot through global_lr
  guard                          check 5: wrong settings are refused
  dry                            one short job per kind of arm, through campaign_job.py
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

SEED, ALPHA = 0, 0.05
LOGISTIC = dict(dataset="mnist", model_type="logistic", local_epochs=5, n_rounds=3)
CNN = dict(dataset="cifar10", model_type="cnn", local_epochs=2, n_rounds=2)
CASES = [
    ("fixed", "logistic_fedavg", LOGISTIC, {"strategies": ["fedavg"]}, "sign_flipping"),
    ("fixed", "logistic_fltrust", LOGISTIC, {"strategies": ["fltrust"]}, "sign_flipping"),
    ("fixed", "logistic_fltrust_informed", LOGISTIC, {"strategies": ["fltrust"]}, "trim_attack"),
    ("fixed", "logistic_trimmed_mean", LOGISTIC, {"strategies": ["trimmed_mean"]}, "sign_flipping"),
    ("system", "logistic_random", LOGISTIC, {"systems": ["random"]}, "sign_flipping"),
    ("system", "logistic_fedstrategist", LOGISTIC, {"systems": ["fedstrategist"], "variant": "a"}, "sign_flipping"),
    ("system", "logistic_adaaggrl", LOGISTIC, {"systems": ["adaaggrl"]}, "sign_flipping"),
    ("system", "logistic_pretrained", LOGISTIC, {"systems": ["oracle", "gradf"]}, "sign_flipping"),
    ("fixed", "cnn_fedavg", CNN, {"strategies": ["fedavg"]}, "sign_flipping"),
    ("fixed", "cnn_fltrust", CNN, {"strategies": ["fltrust"]}, "sign_flipping"),
    ("fixed", "cnn_krum_informed", CNN, {"strategies": ["krum"]}, "trim_attack"),
    ("system", "cnn_random", CNN, {"systems": ["random"]}, "sign_flipping"),
    ("system", "cnn_fedstrategist", CNN, {"systems": ["fedstrategist"], "variant": "a"}, "sign_flipping"),
    ("system", "cnn_adaaggrl", CNN, {"systems": ["adaaggrl"]}, "sign_flipping"),
]


def probe(factor, out, only):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    import src.defense.aggregation_methods  # noqa: F401
    from src.experiments import exp9_dominance_grid as exp9
    from src.experiments import exp10_selector_comparison as exp10
    from src.fl.federated_learner import FederatedLearner
    from src.xai.audit_trail import AuditTrail

    AuditTrail.add = lambda self, *a, **k: None
    log = []
    train = FederatedLearner.train

    def recording_train(self, participants, root_data=None, verbose=True, **kw):
        self._first_step = None
        history = train(self, participants, root_data=root_data, verbose=verbose, **kw)
        log.append({"cls": type(self).__name__, "aggregation": self.aggregation,
                    "global_lr": getattr(self, "global_lr", None), "n_rounds": self.n_rounds,
                    "accuracy": [float(r.global_accuracy) for r in history], "first_step": self._first_step,
                    "params_sha256": hashlib.sha256(np.asarray(self._global_params).tobytes()).hexdigest()})
        return history
    FederatedLearner.train = recording_train

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    for cls in [FederatedLearner] + list(subclasses(FederatedLearner)):
        if "_run_round" in cls.__dict__:
            def wrap(inner):
                def run_round(self, *a, **k):
                    depth = getattr(self, "_round_depth", 0)
                    self._round_depth = depth + 1
                    before = None if self._global_params is None else np.array(self._global_params)
                    result = inner(self, *a, **k)
                    self._round_depth = depth
                    if depth == 0 and self._first_step is None and before is not None:
                        self._first_step = float(np.linalg.norm(np.asarray(self._global_params) - before))
                    return result
                return run_round
            cls._run_round = wrap(cls.__dict__["_run_round"])

    extra = {} if factor == 1.0 and only != "cnn" and not hasattr(FederatedLearner, "_global_step") else {"global_lr": factor}
    results = {}
    for kind, name, setting, arm, attack in CASES:
        if only and not name.startswith(only):
            continue
        del log[:]
        common = dict(alphas=[ALPHA], attack_types=[attack], n_clients=10, byzantine_fraction=0.2, n_classes=10,
                      seed=SEED, root_size=100, seed_rng=True, local_lr=0.01, server_steps="median",
                      **setting, **extra)
        if kind == "fixed":
            exp9.run_dominance_grid(**common, **arm)
        else:
            exp10.run_selector_comparison_grid(**common, **arm)
        results[name] = list(log)
        print(name, len(log), flush=True)
        with open(out, "w") as f:
            json.dump(results, f)


def compare(old_path, new_path, third_path):
    with open(old_path) as f:
        old = json.load(f)
    with open(new_path) as f:
        new = json.load(f)
    with open(third_path) as f:
        third = json.load(f)
    ok = True
    for name in old:
        a = [(r["cls"], r["params_sha256"], r["accuracy"]) for r in old[name]]
        b = [(r["cls"], r["params_sha256"], r["accuracy"]) for r in new[name]]
        same = a == b
        print(f"{'PASS' if same else 'DIFF'} check1 global_lr=1 reproduces the old code: {name} ({len(a)} learners)")
        if not same and name != "logistic_pretrained":
            ok = False
    for name in third:
        final1, final3 = new[name][-1], third[name][-1]
        ratio = final3["first_step"] / final1["first_step"] if final1["first_step"] else None
        passed = ratio is not None and abs(ratio - 1 / 3) < 1e-4
        print(f"{'PASS' if passed else 'DIFF'} check3 first step at 1/3: {name} ({final3['cls']}) ratio {ratio}")
        if not passed and name != "logistic_pretrained":
            ok = False
    for name in third:
        rates = {r["global_lr"] for r in third[name]}
        passed = rates == {1 / 3}
        print(f"{'PASS' if passed else 'FAIL'} check4 every learner at 1/3: {name} ({len(third[name])} learners)")
        ok = ok and passed
    n = len(third.get("logistic_pretrained", []))
    print(f"{'PASS' if n > 2 else 'FAIL'} check4 pretraining ran through these learners: {n}")
    ok = ok and n > 2
    print("CHECKS_PASS" if ok else "CHECKS_FAIL")


def pilot():
    sys.path.insert(0, "scripts/p1")
    from functools import partial
    import cifar_step_pilot as p

    p.AttackedFederatedLearner = partial(p.AttackedFederatedLearner, global_lr=1 / 3)
    ok = True
    for seed in (0, 1):
        with open(f"{p.OUT}/sign_flipping_r100_s{seed}.json") as f:
            ref = json.load(f)["accuracy"][:3]
        got = p.run("sign_flipping", 100, seed, rounds=3, factor=1.0)["accuracy"]
        passed = got == ref
        ok = ok and passed
        print(f"{'PASS' if passed else 'FAIL'} check2 seed {seed}: {got} vs {ref}", flush=True)
    print("CHECKS_PASS" if ok else "CHECKS_FAIL")


def guard():
    base = ["--block", "e1_main", "--arm", "fltrust_r100", "--alpha", "0.05", "--attack", "trim_attack"]
    cases = {
        "wrong factor": base + ["--seed", "42", "--global_lr", "1"],
        "wrong steps": base + ["--seed", "42", "--server_steps", "native"],
        "wrong root": base + ["--seed", "42", "--root", "2500"],
        "wrong rounds": base + ["--seed", "42", "--rounds", "50"],
        "wrong out": base + ["--seed", "42", "--out", "/tmp/x"],
        "arm not in block": ["--block", "e3_hardened", "--arm", "fedavg_r100", "--alpha", "0.05",
                             "--attack", "trim_attack", "--seed", "42"],
        "cell not in block": ["--block", "e3_main", "--arm", "random", "--alpha", "0.05",
                              "--attack", "gaussian_noise", "--seed", "42"],
        "job not in the block's list": ["--block", "e1_invariant", "--arm", "trimmed_mean_r2500", "--alpha", "0.05",
                                        "--attack", "trim_attack", "--seed", "43"],
        "not frozen or wrong hash or environment": base + ["--seed", "42"],
    }
    ok = True
    for name, argv in cases.items():
        run = subprocess.run([sys.executable, "scripts/p1/campaign_job.py", "run"] + argv,
                             capture_output=True, text=True)
        passed = run.returncode != 0 and "refused" in run.stderr
        ok = ok and passed
        print(f"{'PASS' if passed else 'FAIL'} check5 {name}: {run.stderr.strip().splitlines()[-1:]}", flush=True)
    print("CHECKS_PASS" if ok else "CHECKS_FAIL")


def dry(out):
    cases = [("e1_main", "fltrust_r100", "trim_attack"), ("e1_sens_steps", "fltrust_r2500", "sign_flipping"),
             ("e3_hardened", "krum_hardened", "trim_attack"), ("e3_main", "random", "label_flipping"),
             ("e3_main", "adaaggrl", "sign_flipping")]
    ok = True
    for block, arm, attack in cases:
        run = subprocess.run([sys.executable, "scripts/p1/campaign_job.py", "run", "--block", block, "--arm", arm,
                              "--alpha", "0.05", "--attack", attack, "--seed", "0", "--rounds", "2", "--out", out],
                             capture_output=True, text=True)
        path = f"{out}/{block}/raw/{arm}_a0.05_{attack}_s0.json"
        passed = run.returncode == 0 and os.path.exists(path)
        detail = run.stderr.strip().splitlines()[-1:] if not passed else ""
        if passed:
            with open(path) as f:
                d = json.load(f)
            detail = {k: d[k] for k in ("global_lr", "server_steps", "root", "accuracy", "machine")}
            detail["layers"] = len(d["layers"])
        ok = ok and passed
        print(f"{'PASS' if passed else 'FAIL'} dry {block}/{arm}: {detail}", flush=True)
    print("CHECKS_PASS" if ok else "CHECKS_FAIL")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--factor", type=lambda s: 1 / 3 if s == "1/3" else float(s), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="")
    c = sub.add_parser("compare")
    c.add_argument("paths", nargs=3)
    sub.add_parser("pilot")
    sub.add_parser("guard")
    d = sub.add_parser("dry")
    d.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "probe":
        probe(args.factor, args.out, args.only)
    elif args.cmd == "compare":
        compare(*args.paths)
    elif args.cmd == "pilot":
        pilot()
    elif args.cmd == "guard":
        guard()
    else:
        dry(args.out)


if __name__ == "__main__":
    main()
