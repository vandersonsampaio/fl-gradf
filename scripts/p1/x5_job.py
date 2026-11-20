"""
One job of X5, the action set against the reference set (results/p1_campaign/ADDENDUM_3.md).

Two kinds of job, both run through campaign_job.run_job unchanged:
  x5_fixed   FedAvg, FedProx and clustering as fixed rules at factor 1, root 100, on the 12 E3
             cells. These are the three rules of the 7-rule arsenal outside D_R.
  x5_replay  FedStrategist runs of the campaign (e3_main, factor 1/3) and of X1 (x1_systems,
             factor 1), run again with the rule chosen in each round recorded. The replay must
             reproduce the stored run: it records whether the final parameters' hash and the
             accuracy curve equal the stored ones.
On seeds 42-51 a job refuses a job outside this addendum's lists, an R other than 110, an
output other than the campaign's directory, a code hash other than X5_CODE.sha256 and an
unexpected numerical environment.

Usage: x5_job.py run --block x5_fixed --arm fedavg_r100 --alpha 0.05 --attack trim_attack --seed 42
       x5_job.py run --block x5_replay --arm e3_main --alpha 0.05 --attack trim_attack --seed 42
       x5_job.py list --block x5_replay
       x5_job.py hash
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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import campaign_job as cj  # noqa: E402
import x1_job  # noqa: E402

FROZEN_HASH = f"{cj.OUT}/X5_CODE.sha256"
OUTSIDE_DR = ("fedavg", "fedprox", "clustering")
SOURCES = ("e3_main", "x1_systems")   # the blocks whose FedStrategist runs are replayed

BLOCKS = {
    "x5_fixed": {"factor": 1.0, "steps": "median", "cells": cj.E3_CELLS,
                 "arms": {f"{r}_r100": cj.fixed(r, 100) for r in OUTSIDE_DR}},
    # each arm names the source block; the replayed arm is always fedstrategist
    "x5_replay": {"cells": cj.E3_CELLS, "arms": {s: {"source": s} for s in SOURCES}},
}


def job_keys(block):
    spec = BLOCKS[block]
    return [(arm, alpha, attack, seed) for arm in spec["arms"] for alpha, attack in spec["cells"]
            for seed in cj.SEEDS]


def code_hash():
    """The campaign's hash inputs plus x1_job.py (its blocks are replayed) and this file."""
    files = sorted(glob.glob("src/**/*.py", recursive=True)) + [os.path.abspath(cj.__file__),
                                                                 os.path.abspath(x1_job.__file__),
                                                                 os.path.abspath(__file__)]
    h = hashlib.sha256()
    for path in files:
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def guard(args):
    spec = BLOCKS[args.block]
    if args.arm not in spec["arms"]:
        raise SystemExit(f"refused: arm {args.arm!r} is not in block {args.block}")
    if (args.alpha, args.attack) not in spec["cells"]:
        raise SystemExit(f"refused: cell {(args.alpha, args.attack)} is not in block {args.block}")
    if args.seed in cj.CONFIRMATORY:
        if (args.arm, args.alpha, args.attack, args.seed) not in job_keys(args.block):
            raise SystemExit("refused: this job is not in the block's list")
        if args.rounds != cj.ROUNDS:
            raise SystemExit(f"refused: R={args.rounds} on seeds 42-51; X5 assigns {cj.ROUNDS}")
        if args.out != cj.OUT or args.source_out != cj.OUT:
            raise SystemExit("refused: X5 jobs on seeds 42-51 read and write only the campaign's directory")
        if not os.path.exists(FROZEN_HASH):
            raise SystemExit("refused: ADDENDUM_3 is not frozen")
        with open(FROZEN_HASH) as f:
            frozen = f.read().split()[0]
        if frozen != code_hash():
            raise SystemExit("refused: the code differs from the hash frozen for X5")
        cj.check_environment()


def run_replay(args):
    """Runs the stored FedStrategist job again, recording the bandit's choice in each round."""
    from src.defense.fedstrategist_selector import LinUCBAgent

    source = BLOCKS["x5_replay"]["arms"][args.arm]["source"]
    stored_path = f"{args.source_out}/{source}/raw/fedstrategist_a{args.alpha}_{args.attack}_s{args.seed}.json"
    if not os.path.exists(stored_path):
        raise SystemExit(f"refused: no stored run to replay at {stored_path}")
    with open(stored_path) as f:
        stored = json.load(f)
    if stored["rounds"] != args.rounds:
        raise SystemExit(f"refused: the stored run has R={stored['rounds']}, this job R={args.rounds}")

    actions = []
    select = LinUCBAgent.select_action

    def recording_select(self, state):
        action = select(self, state)
        actions.append(action)
        return action
    LinUCBAgent.select_action = recording_select

    last, pretraining, layers = cj.run_job(source, "fedstrategist", args.alpha, args.attack, args.seed, args.rounds)
    curve = last.pop("accuracy")
    if len(actions) != args.rounds:
        raise SystemExit(f"guard: recorded {len(actions)} choices for {args.rounds} rounds")
    return {"source_block": source, "source_path": stored_path, "accuracy": curve,
            "primary_trailing5": float(np.mean(curve[-5:])), "learner": last,
            "actions": actions, "action_counts": {a: actions.count(a) for a in sorted(set(actions))},
            "identical_params": last["params_sha256"] == stored["learner"]["params_sha256"],
            "identical_curve": curve == stored["accuracy"],
            "source_code_sha256": stored["code_sha256"], "source_machine": stored["machine"]}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--block", required=True, choices=sorted(BLOCKS))
    r.add_argument("--arm", required=True)
    r.add_argument("--alpha", type=float, required=True)
    r.add_argument("--attack", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--rounds", type=int, default=cj.ROUNDS)
    r.add_argument("--out", default=cj.OUT)
    r.add_argument("--source_out", default=cj.OUT, help="directory holding the stored runs to replay")
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

    guard(args)
    if args.block == "x5_fixed":
        path = f"{args.out}/x5_fixed/raw/{args.arm}_a{args.alpha}_{args.attack}_s{args.seed}.json"
    else:
        path = f"{args.out}/x5_replay/raw/fedstrategist_{args.arm}_a{args.alpha}_{args.attack}_s{args.seed}.json"
    if os.path.exists(path):
        print(f"{path} already exists, skipping")
        return
    cj.BLOCKS.update(x1_job.BLOCKS)
    cj.BLOCKS.update({"x5_fixed": BLOCKS["x5_fixed"]})
    start = time.time()
    head = {"block": args.block, "arm": args.arm, "alpha": args.alpha, "attack": args.attack, "seed": args.seed,
            "rounds": args.rounds}
    if args.block == "x5_fixed":
        spec = BLOCKS["x5_fixed"]
        last, pretraining, layers = cj.run_job("x5_fixed", args.arm, args.alpha, args.attack, args.seed, args.rounds)
        curve = last.pop("accuracy")
        body = {"global_lr": spec["factor"], "server_steps": spec["steps"], "root": spec["arms"][args.arm]["root"],
                "accuracy": curve, "final": curve[-1], "primary_trailing5": float(np.mean(curve[-5:])),
                "learner": last, "pretraining": pretraining, "layers": layers}
    else:
        body = run_replay(args)
    result = {**head, **body, "machine": socket.gethostname(), "platform": platform.processor(),
              "code_sha256": code_hash(), "campaign_code_sha256": cj.code_hash(), "seconds": time.time() - start}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
