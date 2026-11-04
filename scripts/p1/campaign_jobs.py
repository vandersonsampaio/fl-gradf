"""
Writes the campaign's job list in the order of PREREGISTRATION.md, Sec. 3: the pairs of the
invariance check first, the rest of the invariant block, E1 main, the sensitivity blocks, E3.
Inside a block jobs are grouped by seed, so that a partial block holds whole seeds.

Usage: campaign_jobs.py > results/p1_campaign/jobs.txt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from campaign_job import job_keys  # noqa: E402

PY = "../fl-fedmed/venv/bin/python scripts/p1/campaign_job.py run"
ORDER = ("e1_invariant", "e1_main", "e1_sens_factor", "e1_sens_steps", "e3_hardened", "e3_main", "e3_sens")


def main():
    first = [("e1_invariant", k) for k in job_keys("e1_invariant") if k[2] != "fltrust_aligned"]
    first += [("e1_main", k) for k in job_keys("e1_main")
              if k[0] == "trimmed_mean_r100" and k[2] != "fltrust_aligned" and k[3] == 42]
    rest = []
    for block in ORDER:
        keys = sorted(job_keys(block), key=lambda k: k[3])
        rest += [(block, k) for k in keys if (block, k) not in first]
    for block, (arm, alpha, attack, seed) in first + rest:
        print(f"{PY} --block {block} --arm {arm} --alpha {alpha} --attack {attack} --seed {seed}")


if __name__ == "__main__":
    main()
