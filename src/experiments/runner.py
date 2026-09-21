"""
CLI that runs the GRADF experiments (exp1, exp2, exp3, exp4, exp6, exp7) in
sequence, each as a subprocess `python -m src.experiments.<name>` —
delegates each module's own CLI/argparse and result-saving to results/.

Not included here:
  - exp5 (clinical validation): depends on human data collection. See
    `python -m src.experiments.exp5_clinical --help`.
  - exp8 (XAI examples): doesn't need human data, but depends on the audit
    trails that exp2 already saves to `results/audit_trail/` — run it
    manually afterward: `python -m src.experiments.exp8_xai_examples`.
"""

import argparse
import subprocess
import sys
from typing import List, Optional

EXPERIMENTS = [
    "exp1_baseline",
    "exp2_robustness",
    "exp3_scalability",
    "exp4_adaptive",
    "exp6_fairness",
    "exp7_modality_ablation",
]


def run_all(experiments: Optional[List[str]] = None, extra_args: Optional[List[str]] = None) -> None:
    experiments = experiments or EXPERIMENTS
    extra_args = extra_args or []
    for name in experiments:
        print(f"\n{'=' * 60}\nRunning src.experiments.{name}\n{'=' * 60}")
        subprocess.run(
            [sys.executable, "-m", f"src.experiments.{name}", *extra_args],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=None,
                         help="Subset of experiments to run (default: all)")
    args, extra = parser.parse_known_args()
    run_all(args.experiments, extra)
