"""
Writes X5's job list in the order of ADDENDUM_3.md: x5_fixed, then the x5_replay jobs of
x1_systems, then those of e3_main, each grouped by seed.

Usage: x5_jobs.py > results/p1_campaign/x5_jobs.txt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x5_job import job_keys  # noqa: E402

PY = "../fl-fedmed/venv/bin/python scripts/p1/x5_job.py run"
ORDER = (("x5_fixed", None), ("x5_replay", "x1_systems"), ("x5_replay", "e3_main"))


def main():
    for block, arm_only in ORDER:
        keys = [k for k in job_keys(block) if arm_only is None or k[0] == arm_only]
        for arm, alpha, attack, seed in sorted(keys, key=lambda k: k[3]):
            print(f"{PY} --block {block} --arm {arm} --alpha {alpha} --attack {attack} --seed {seed}")


if __name__ == "__main__":
    main()
