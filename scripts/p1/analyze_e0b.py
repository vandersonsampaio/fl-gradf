"""
E0b analysis (results/p1_cifar10/e0b/PLAN.md). Applies the frozen rules:

  Sec. 1  regression check: the first 50 rounds of the 21 E0b jobs that repeat
          an E0 configuration must equal E0 bit for bit (stops otherwise)
  Sec. 2  rule for R (FedAvg no-attack curves, all three alpha)
  Sec. 3  GRADF Layer-1 threshold: 1.5 x the largest honest update L2 norm
  Sec. 4  sigma_E0 for the E1 family and the E3 approximation, at round R
  Sec. 5  compute budget: C(R, k) and the grid level (needs the part D costs;
          without them the plan's provisional costs are used and the level is
          marked provisional)
  Sec. 6  descriptive outputs

Usage:
  python scripts/p1/analyze_e0b.py [--raw DIR] [--e0 DIR] [--costs FILE] [--out DIR]
Defaults: --raw results/p1_cifar10/e0b/raw, --e0 results/p1_cifar10/e0,
--costs results/p1_cifar10/e0b/costs.json (optional), --out results/p1_cifar10/e0b.
Writes <out>/analysis.txt and <out>/decisions.json.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ALPHAS = (0.5, 0.1, 0.05)
ATTACKS = ("sign_flipping", "trim_attack")
SEEDS = (0, 1, 2)
ROUNDS = 180
EPOCHS = 2
WINDOW = 5                       # trailing mean over rounds r-4 .. r
GAIN_PP = 1.0
CANDIDATES = range(20, 151, 10)
R_MAX = 150
TAU_RATIO = 1.5
REGRESSION_ROUNDS = 50

BUDGET_H = 236.0
PROCS = 12
N_SEEDS = 10
CELLS_FULL = 18
LEVELS = [  # (level, description, k)
    (0, "3 alpha x 6 attacks (all but fltrust_aligned)", 18),
    (1, "3 alpha x {sign_flipping, label_flipping, trim_attack, krum_collusion}", 12),
    (2, "alpha in {0.5, 0.05} x {sign_flipping, label_flipping, trim_attack} (floor)", 6),
]
# Plan Sec. 5, provisional (s per round per process at 12 x 1; pretraining, h, full grid)
PROVISIONAL_COSTS = {
    "s_round": {"fixed": 21.0, "adaaggrl": 35.0, "random": 23.0, "attack_aware": 23.0,
                "gradf": 23.0, "fedstrategist": 23.0},
    "pretraining_h_full_grid": 20.0,
}


def load_raw(raw_dir):
    runs = {}
    for path in sorted(glob.glob(f"{raw_dir}/*.json")):
        with open(path) as f:
            d = json.load(f)
        if d["seed"] not in SEEDS:
            continue
        key = (d["alpha"], d["attack"], d["rule"], d["root_size"], d["seed"])
        runs[key] = d
    return runs


def expected_keys():
    keys = []
    for seed in SEEDS:
        for alpha in ALPHAS:
            keys.append((alpha, "none", "fedavg", 100, seed))
        for attack in ATTACKS:
            keys.append((0.05, attack, "fedavg", 100, seed))
            for root in (100, 2500):
                keys.append((0.05, attack, "fltrust", root, seed))
    return keys


def check_complete(runs, out):
    missing = [k for k in expected_keys() if k not in runs]
    short = [k for k, d in runs.items() if len(d["accuracy"]) != ROUNDS or d["local_epochs"] != EPOCHS]
    out.append(f"Runs found: {len(runs)}/27; missing {len(missing)}; wrong length/epochs {len(short)}")
    for k in missing + short:
        out.append(f"  {k}")
    return not missing and not short


def regression(runs, e0_dir, out):
    """Sec. 1: first 50 rounds equal to E0, bit for bit."""
    out += ["", "## Sec. 1 Regression check against E0 (first 50 rounds, exact)"]
    n, bad = 0, []
    for (alpha, attack, rule, root, seed), d in sorted(runs.items()):
        if root != 100:
            continue
        sub = "curves" if attack == "none" else "separation"
        path = f"{e0_dir}/{sub}/a{alpha}_{attack}_{rule}_e{EPOCHS}_s{seed}.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            ref = json.load(f)["accuracy"][:REGRESSION_ROUNDS]
        n += 1
        if d["accuracy"][:REGRESSION_ROUNDS] != ref:
            first = next(i for i, (a, b) in enumerate(zip(d["accuracy"], ref)) if a != b)
            bad.append(f"  MISMATCH {alpha} {attack} {rule} s{seed}: first differing round {first + 1}")
    out.append(f"Compared {n}/21 jobs; mismatches {len(bad)}")
    out += bad
    return n == 21 and not bad


def smooth(curve, r):
    return float(np.mean(curve[r - WINDOW:r]))


def rule_for_r(runs, out):
    """Sec. 2."""
    out += ["", "## Sec. 2 Rule for R (FedAvg, no attack; seed mean, 5-round trailing mean)"]
    mean_curve = {a: np.mean([runs[(a, "none", "fedavg", 100, s)]["accuracy"] for s in SEEDS], axis=0)
                  for a in ALPHAS}
    out.append("r     " + "  ".join(f"α={a}: S(r)  g20   g10-30" for a in ALPHAS))
    R = None
    for r in CANDIDATES:
        cells, ok = [], True
        for a in ALPHAS:
            c = mean_curve[a]
            g1 = 100 * (smooth(c, r + 20) - smooth(c, r))
            g2 = 100 * (smooth(c, r + 30) - smooth(c, r + 10))
            ok &= (g1 < GAIN_PP) and (g2 < GAIN_PP)
            cells.append(f"{100 * smooth(c, r):5.2f} {g1:+5.2f} {g2:+5.2f}")
        out.append(f"{r:<5} " + "   ".join(cells) + ("   <- rule met" if ok and R is None else ""))
        if ok and R is None:
            R = r
    stabilized = R is not None
    if not stabilized:
        R = R_MAX
        out.append(f"No candidate meets the rule: R = R_max = {R_MAX} ('not stabilized at R = 150').")
    else:
        out.append(f"R = {R}")
    return R, stabilized


def layer1(runs, out):
    """Sec. 3."""
    out += ["", "## Sec. 3 GRADF Layer-1 threshold (honest update L2 norms, all rounds)"]
    blocks = {"A (FedAvg, no attack)": [], "B (FedAvg, attack)": [], "C (FLTrust, attack)": []}
    by_alpha = {a: [] for a in ALPHAS}
    for (alpha, attack, rule, root, seed), d in runs.items():
        norms = np.array(d["update_l2"])                      # rounds x clients
        honest = [i for i in range(norms.shape[1]) if i not in d["byzantine_ids"]]
        h = norms[:, honest].ravel()
        block = "A (FedAvg, no attack)" if attack == "none" else (
            "B (FedAvg, attack)" if rule == "fedavg" else "C (FLTrust, attack)")
        blocks[block].append(h)
        by_alpha[alpha].append(h)
    allh = np.concatenate([np.concatenate(v) for v in blocks.values()])
    tau = TAU_RATIO * float(allh.max())
    for name, v in blocks.items():
        v = np.concatenate(v)
        out.append(f"{name:<22} n={v.size:6d}  median {np.median(v):.3f}  p99 {np.percentile(v, 99):.3f}  "
                   f"max {v.max():.3f}  share > tau {np.mean(v > tau):.4f}")
    for a in ALPHAS:
        v = np.concatenate(by_alpha[a])
        out.append(f"alpha {a:<5} max {v.max():.3f}")
    out.append(f"tau_CNN = {TAU_RATIO} x {allh.max():.4f} = {tau:.4f}  (MNIST tau = 5.0)")
    return tau, float(allh.max())


def acc_at(runs, key, R):
    return 100 * runs[key]["accuracy"][R - 1]


def sigma(runs, R, out):
    """Sec. 4."""
    out += ["", f"## Sec. 4 sigma_E0 at round R = {R} (final-round accuracy, unsmoothed, p.p.)"]
    d1 = [np.mean([acc_at(runs, (0.05, a, "fltrust", 2500, s), R) - acc_at(runs, (0.05, a, "fltrust", 100, s), R)
                   for a in ATTACKS]) for s in SEEDS]
    d3 = [np.mean([acc_at(runs, (0.05, a, "fltrust", 100, s), R) - acc_at(runs, (0.05, a, "fedavg", 100, s), R)
                   for a in ATTACKS]) for s in SEEDS]
    s1, s3 = float(np.std(d1, ddof=1)), float(np.std(d3, ddof=1))
    out.append(f"E1: D_s = mean over 2 cells of FLTrust(2500) - FLTrust(100): "
               f"{', '.join(f'{v:+.2f}' for v in d1)}; sigma_E0^E1 = {s1:.3f}; 0.5 sigma = {0.5 * s1:.3f}")
    out.append(f"E3 (approximation): D_s = mean over 2 cells of FLTrust(100) - FedAvg: "
               f"{', '.join(f'{v:+.2f}' for v in d3)}; sigma_E0^E3 = {s3:.3f}; 0.5 sigma = {0.5 * s3:.3f}")
    return s1, s3


def cost(R, k, costs):
    s = costs["s_round"]
    h = 1 / 3600 / PROCS
    e1 = (7 * 21 * N_SEEDS + 21 * N_SEEDS) * R * s["fixed"] * h
    runs_cell_grid = CELLS_FULL * N_SEEDS
    e3 = runs_cell_grid * R * sum(s[x] for x in ("adaaggrl", "random", "attack_aware", "gradf", "fedstrategist")) * h \
        + costs["pretraining_h_full_grid"]
    sens = runs_cell_grid * R * (s["adaaggrl"] + s["fedstrategist"]) * h
    return e1 + (e3 + sens) * k / CELLS_FULL


def budget(R, stabilized, costs_path, out):
    """Sec. 5."""
    measured = os.path.exists(costs_path)
    costs = json.load(open(costs_path)) if measured else PROVISIONAL_COSTS
    out += ["", f"## Sec. 5 Compute budget ({BUDGET_H:.0f} h; costs "
                f"{'measured: ' + costs_path if measured else 'PROVISIONAL (part D not run yet)'})"]
    for lvl, desc, k in LEVELS:
        out.append(f"level {lvl} (k = {k:2d}) C(R = {R}, k) = {cost(R, k, costs):6.1f} h   {desc}")
    chosen = next(((lvl, k) for lvl, _, k in LEVELS if cost(R, k, costs) <= BUDGET_H), None)
    R_final, truncated = R, False
    if chosen is None:
        chosen = (2, 6)
        fits = [r for r in range(10, R + 1, 10) if cost(r, 6, costs) <= BUDGET_H]
        R_final, truncated = (max(fits) if fits else 10), True
        out.append(f"Even the floor exceeds the budget: R truncated to {R_final} (deviation).")
    out.append(f"Grid level {chosen[0]} (k = {chosen[1]}); R = {R_final}"
               + ("" if measured else "  [provisional: re-run after part D]"))
    return {"grid_level": chosen[0], "k": chosen[1], "R_final": R_final, "R_truncated_by_budget": truncated,
            "costs_measured": measured, "costs": costs,
            "cost_h": {lvl: cost(R, k, costs) for lvl, _, k in LEVELS}}


def descriptive(runs, R, out):
    """Sec. 6 (descriptive only)."""
    out += ["", "## Sec. 6 Descriptive"]
    out.append("FedAvg no-attack accuracy (%, seed mean, 5-round trailing mean):")
    for a in ALPHAS:
        c = np.mean([runs[(a, "none", "fedavg", 100, s)]["accuracy"] for s in SEEDS], axis=0)
        out.append(f"  alpha {a}: " + "  ".join(f"r{r} {100 * smooth(c, r):.1f}" for r in (50, 100, 150, 180)))
    out.append(f"Ranking stability after R (limited: FedAvg - FLTrust(100) at alpha 0.05, seed mean, p.p.):")
    for a in ATTACKS:
        at = [acc_at(runs, (0.05, a, "fedavg", 100, s), r) - acc_at(runs, (0.05, a, "fltrust", 100, s), r)
              for r in (R, ROUNDS) for s in SEEDS]
        out.append(f"  {a}: at R {np.mean(at[:3]):+.2f} (per seed {', '.join(f'{v:+.2f}' for v in at[:3])}); "
                   f"at {ROUNDS} {np.mean(at[3:]):+.2f} (per seed {', '.join(f'{v:+.2f}' for v in at[3:])})")
    out.append(f"FLTrust root contrast per cell at R (2500 - 100, seed mean, p.p.; feeds sigma_E0^E1 only):")
    for a in ATTACKS:
        v = [acc_at(runs, (0.05, a, "fltrust", 2500, s), R) - acc_at(runs, (0.05, a, "fltrust", 100, s), R)
             for s in SEEDS]
        out.append(f"  {a}: {np.mean(v):+.2f} (per seed {', '.join(f'{x:+.2f}' for x in v)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/p1_cifar10/e0b/raw")
    ap.add_argument("--e0", default="results/p1_cifar10/e0")
    ap.add_argument("--costs", default="results/p1_cifar10/e0b/costs.json")
    ap.add_argument("--out", default="results/p1_cifar10/e0b")
    args = ap.parse_args()

    out = ["E0b analysis (results/p1_cifar10/e0b/PLAN.md), seeds 0-2, 180 rounds, 2 local epochs.", ""]
    runs = load_raw(args.raw)
    ok = check_complete(runs, out)
    ok = ok and regression(runs, args.e0, out)
    if not ok:
        out.append("\nSTOP: incomplete grid or regression mismatch (PLAN Sec. 1). No rule applied.")
        print("\n".join(out))
        sys.exit(1)

    R, stabilized = rule_for_r(runs, out)
    tau, max_honest = layer1(runs, out)
    plan = budget(R, stabilized, args.costs, out)
    s1, s3 = sigma(runs, plan["R_final"], out)          # R after any budget truncation
    descriptive(runs, plan["R_final"], out)

    decisions = {"R_rule": R, "stabilized": stabilized, **plan, "tau_cnn": tau, "max_honest_l2": max_honest,
                 "sigma_e0_e1": s1, "sigma_e0_e3": s3}
    text = "\n".join(out)
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/analysis.txt", "w") as f:
        f.write(text + "\n")
    with open(f"{args.out}/decisions.json", "w") as f:
        json.dump(decisions, f, indent=1)
    print(text)


if __name__ == "__main__":
    main()
