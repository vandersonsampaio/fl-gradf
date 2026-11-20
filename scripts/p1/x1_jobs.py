"""
Writes X1's job list in the order of ADDENDUM_2.md: x1_fixed, x1_systems, x1_hardened, each
grouped by seed, so that a partial block holds whole seeds.

Usage: x1_jobs.py > results/p1_campaign/x1_jobs.txt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x1_job import job_keys  # noqa: E402

PY = "../fl-fedmed/venv/bin/python scripts/p1/x1_job.py run"
ORDER = ("x1_fixed", "x1_systems", "x1_hardened")


def main():
    for block in ORDER:
        for arm, alpha, attack, seed in sorted(job_keys(block), key=lambda k: k[3]):
            print(f"{PY} --block {block} --arm {arm} --alpha {alpha} --attack {attack} --seed {seed}")


if __name__ == "__main__":
    main()
