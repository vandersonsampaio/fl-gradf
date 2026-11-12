"""
One job of X1, the exploratory runs at global factor 1 (results/p1_campaign/ADDENDUM_2.md).

Reuses campaign_job.py unchanged: its run_job runs the arm, and this file only adds the X1
blocks and their guard. Every setting comes from the job's block. On seeds 42-51 a job also
refuses a code hash other than the one frozen in X1_CODE.sha256, an R other than 110, an
output other than the campaign's directory and an unexpected numerical environment.

Usage: x1_job.py run --block x1_fixed --arm median_r100 --alpha 0.05 --attack trim_attack --seed 42
       x1_job.py list --block x1_systems
       x1_job.py hash
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

FROZEN_HASH = f"{cj.OUT}/X1_CODE.sha256"
FACTOR = 1.0

BLOCKS = {
    "x1_fixed": {"factor": FACTOR, "steps": "median", "cells": cj.ALL_CELLS,
                 "arms": {f"{r}_r100": cj.fixed(r, 100) for r in ("median", "krum")}},
    "x1_systems": {"factor": FACTOR, "steps": "median", "cells": cj.E3_CELLS,
                   "arms": {"random": cj.system("random"), "attack_aware": cj.system("oracle"),
                            "gradf": cj.system("gradf"), "fedstrategist": cj.system("fedstrategist"),
                            "adaaggrl": cj.system("adaaggrl")}},
    "x1_hardened": {"factor": FACTOR, "steps": "median", "cells": cj.E3_CELLS,
                    "arms": {f"{r}_hardened": {"kind": "hardened", "rule": r, "root": 100}
                             for r in ("fltrust", "median", "trimmed_mean", "krum")}},
}


def job_keys(block):
    """(arm, alpha, attack, seed) of an X1 block, as fixed by ADDENDUM_2."""
    spec = BLOCKS[block]
    return [(arm, alpha, attack, seed) for arm in spec["arms"] for alpha, attack in spec["cells"]
            for seed in cj.SEEDS]


def code_hash():
    """The campaign's hash inputs plus this file."""
    files = sorted(glob.glob("src/**/*.py", recursive=True)) + [os.path.abspath(cj.__file__),
                                                                 os.path.abspath(__file__)]
    h = hashlib.sha256()
    for path in files:
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def guard(args):
    if args.block not in BLOCKS:
        raise SystemExit(f"refused: {args.block!r} is not an X1 block")
    spec = BLOCKS[args.block]
    if args.arm not in spec["arms"]:
        raise SystemExit(f"refused: arm {args.arm!r} is not in block {args.block}")
    if (args.alpha, args.attack) not in spec["cells"]:
        raise SystemExit(f"refused: cell {(args.alpha, args.attack)} is not in block {args.block}")
    if args.seed in cj.CONFIRMATORY:
        if (args.arm, args.alpha, args.attack, args.seed) not in job_keys(args.block):
            raise SystemExit("refused: this job is not in the block's list")
        if args.rounds != cj.ROUNDS:
            raise SystemExit(f"refused: R={args.rounds} on seeds 42-51; X1 assigns {cj.ROUNDS}")
        if args.out != cj.OUT:
            raise SystemExit("refused: X1 jobs on seeds 42-51 write only to the campaign's directory")
        if not os.path.exists(FROZEN_HASH):
            raise SystemExit("refused: ADDENDUM_2 is not frozen")
        with open(FROZEN_HASH) as f:
            frozen = f.read().split()[0]
        if frozen != code_hash():
            raise SystemExit("refused: the code differs from the hash frozen for X1")
        cj.check_environment()


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
    spec = BLOCKS[args.block]
    path = f"{args.out}/{args.block}/raw/{args.arm}_a{args.alpha}_{args.attack}_s{args.seed}.json"
    if os.path.exists(path):
        print(f"{path} already exists, skipping")
        return
    cj.BLOCKS.update(BLOCKS)
    start = time.time()
    last, pretraining, layers = cj.run_job(args.block, args.arm, args.alpha, args.attack, args.seed, args.rounds)
    curve = last.pop("accuracy")
    result = {"block": args.block, "arm": args.arm, "alpha": args.alpha, "attack": args.attack, "seed": args.seed,
              "global_lr": spec["factor"], "server_steps": spec["steps"], "root": spec["arms"][args.arm]["root"],
              "rounds": args.rounds, "accuracy": curve, "final": curve[-1],
              "primary_trailing5": float(np.mean(curve[-5:])), "learner": last, "pretraining": pretraining,
              "layers": layers, "machine": socket.gethostname(), "platform": platform.processor(),
              "code_sha256": code_hash(), "campaign_code_sha256": cj.code_hash(), "seconds": time.time() - start}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
