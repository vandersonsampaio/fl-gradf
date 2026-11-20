"""
Applies the P1 confirmatory analyses (scripts/p1/inference.py) to the
MNIST data, as required by gate P-1. Writes
results/p1_audit/mnist_inference{suffix}.json and the E3 tables as CSV.

--source published: fixed rules from the published (unseeded) raw files.
--source seeded (canonical): fixed rules from the seeded rerun in
  results/p1_invariance/ (scripts/p1/run_invariance_mnist.sh), which fixes
  the copied median/TM/Krum rows under fltrust_aligned (AUDIT.md, B5).
The adaptive systems always come from the published exp10 runs.

Usage: venv/bin/python scripts/p1/run_mnist_inference.py [--source seeded|published]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import audit_mnist as audit  # noqa: E402
import inference as inf  # noqa: E402
from analyze_invariance import load_seeded  # noqa: E402

OUT = "results/p1_audit"
FA = "fltrust_aligned"
SYSTEMS = ["Random", "Oracle (decoupled)", "GRADF", "FedStrategist", "AdaAggRL"]
SENSITIVITY = ["AdaAggRL (pretrained)", "FedStrategist (GRADF detector)"]


def load_fixed(source):
    if source == "published":
        fixed = audit.load_fixed()
        return fixed["100"], fixed["3000_ten_seeds"], fixed["50_ten_seeds"]
    df = load_seeded()
    return tuple(df[df.root == r].drop(columns="root") for r in (100, 3000, 50))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("seeded", "published"), default="seeded")
    source = ap.parse_args().source
    sfx = "" if source == "published" else "_seeded"
    os.makedirs(OUT, exist_ok=True)
    sy = audit.load_systems()
    fx100, fx3000, fx50 = load_fixed(source)
    res = {"fixed_rules_source": source}

    # E1 (RQ1): FLTrust advantage over TM, enlarged vs canonical root
    for name, excl in (("all_21", ()), ("no_fltrust_aligned_18", (FA,))):
        r, per_seed = inf.e1_root_dependence(fx100, fx3000, exclude=excl)
        res[f"E1_3000_vs_100_{name}"] = {**r, "D_s": per_seed.round(5).to_dict()}
        r50, _ = inf.e1_root_dependence(fx50, fx100, exclude=excl)
        res[f"E1_100_vs_50_{name}"] = r50
    # published: the non-FLTrust rows of the 100/50 grids are copies of the
    # full-root runs, so this check is vacuous (AUDIT.md, B3).
    res["TM_invariance_on_existing_data"] = inf.tm_invariance(fx100, fx3000)

    # E2 (RQ2): in-sample vs cross-fitted headroom
    for name, fx in (("all_21", fx100), ("no_fltrust_aligned_18", fx100[fx100.attack_type != FA])):
        res[f"E2_{name}"] = {**inf.e2_headroom(fx), **inf.e2_selection_stability(fx)}

    # E3 (RQ3): confirmatory grid = 18 cells without FLTrust aligned
    fx18, sy18 = fx100[fx100.attack_type != FA], sy[sy.attack_type != FA]
    t, desc = inf.e3_capture(fx18, sy18, SYSTEMS)
    t.to_csv(f"{OUT}/mnist_e3_confirmatory_18cells{sfx}.csv", index=False)
    res["E3_descriptive_18"] = desc
    ts, _ = inf.e3_capture(fx18, sy18, SENSITIVITY + ["Random"])
    ts.to_csv(f"{OUT}/mnist_e3_sensitivity_18cells{sfx}.csv", index=False)
    t21, desc21 = inf.e3_capture(fx100, sy, SYSTEMS)
    t21.to_csv(f"{OUT}/mnist_e3_all_21cells{sfx}.csv", index=False)
    res["E3_descriptive_21"] = desc21

    with open(f"{OUT}/mnist_inference{sfx}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)

    cols = ["family", "system", "mean", "ci95_lo", "ci95_hi", "p_wilcoxon", "p_wilcoxon_holm",
            "p_t", "d_z", "p_tost", "category"]
    for k, v in res.items():
        if k.startswith(("E1", "E2", "TM")) and isinstance(v, dict):
            print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                      for kk, vv in v.items() if kk not in ("D_s", "H_loso_per_seed")})
    print("\nE3 confirmatory (18 cells)\n", t[cols].round(4).to_string(index=False))
    print("\nE3 sensitivity\n", ts[cols].round(4).to_string(index=False))
    print("\n", desc, desc21)


if __name__ == "__main__":
    main()
