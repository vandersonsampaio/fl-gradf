"""
P-1: analysis of the seeded MNIST rerun (scripts/p1/run_invariance_mnist.sh).

1. Root invariant (plan Sec. 4.4): for median, trimmed mean and Krum,
   Acc(c, s, r=100) and Acc(c, s, r=50) must equal Acc(c, s, r=3000) exactly,
   except under fltrust_aligned, whose attacker copies the server's root update.
2. Reproducibility against the published (unseeded) data: per-cell mean
   differences, to show how much the published numbers move.
3. E1 recomputed on the seeded runs (V = FLTrust - TM), 21 and 18 cells.

Usage: venv/bin/python scripts/p1/analyze_invariance.py
"""

import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import audit_mnist as audit  # noqa: E402
import inference as inf  # noqa: E402

RAW = "results/p1_invariance/raw"
OUT = "results/p1_invariance"
FA = "fltrust_aligned"
RELATIONAL = ["median", "trimmed_mean", "krum"]


def load_seeded():
    frames = []
    for d in sorted(glob.glob(f"{RAW}/root*_seed*")):
        if not os.path.exists(f"{d}/done"):
            raise SystemExit(f"incomplete job: {d}")
        csv = glob.glob(f"{d}/exp9_dominance_grid*_seeded_raw.csv")[0]
        root = int(os.path.basename(d).split("_")[0].removeprefix("root"))
        frames.append(pd.read_csv(csv).assign(root=root))
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_seeded()
    key = ["alpha", "attack_type", "strategy", "seed"]
    wide = df.pivot_table(index=key, columns="root", values="accuracy")
    res = {"n_runs": int(len(df)), "runs_per_root": df.groupby("root").size().to_dict()}

    inv = []
    for r in (100, 50):
        d = (wide[r] - wide[3000]).rename("diff").reset_index()
        d = d[d.strategy.isin(RELATIONAL)]
        for fa, g in d.groupby(d.attack_type == FA):
            inv.append({"root": r, "subset": FA if fa else "other 6 attacks",
                        "n_pairs": len(g), "n_nonzero": int((g["diff"] != 0).sum()),
                        "max_abs": float(g["diff"].abs().max()),
                        "mean_diff": float(g["diff"].mean())})
    inv = pd.DataFrame(inv)
    inv.to_csv(f"{OUT}/invariance_check.csv", index=False)
    res["invariant_holds_outside_fltrust_aligned"] = bool(
        (inv[inv.subset != FA].n_nonzero == 0).all())

    # published (unseeded) vs seeded, cell means
    fixed = audit.load_fixed()
    pub = {3000: fixed["3000_ten_seeds"], 100: fixed["100"], 50: fixed["50_ten_seeds"]}
    drift = []
    for r, p in pub.items():
        a = p.groupby(["alpha", "attack_type", "strategy"]).accuracy.mean()
        b = df[df.root == r].groupby(["alpha", "attack_type", "strategy"]).accuracy.mean()
        d = (b - a).dropna()
        drift.append({"root": r, "max_abs_cell_mean_diff": float(d.abs().max()),
                      "mean_abs_cell_mean_diff": float(d.abs().mean())})
    res["published_vs_seeded"] = drift

    # Table 1 / Table 2 on seeded data
    seeded_fixed = {str(r): df[df.root == r].drop(columns="root") for r in (3000, 100, 50)}
    t1 = audit.table1(seeded_fixed)
    t1.to_csv(f"{OUT}/table1_seeded.csv", index=False)
    t2 = audit.table2(seeded_fixed, roots=("3000", "100"))
    t2.to_csv(f"{OUT}/table2_seeded.csv", index=False)

    # E1 on seeded data
    for name, excl in (("all_21", ()), ("no_fltrust_aligned_18", (FA,))):
        r, per_seed = inf.e1_root_dependence(seeded_fixed["100"], seeded_fixed["3000"], exclude=excl)
        res[f"E1_3000_vs_100_{name}"] = r
        r50, _ = inf.e1_root_dependence(seeded_fixed["50"], seeded_fixed["100"], exclude=excl)
        res[f"E1_100_vs_50_{name}"] = r50
    # E2 on seeded root-100 data
    for name, fx in (("all_21", seeded_fixed["100"]),
                     ("no_fltrust_aligned_18", seeded_fixed["100"].query("attack_type != @FA"))):
        res[f"E2_{name}"] = {**inf.e2_headroom(fx), **inf.e2_selection_stability(fx)}

    with open(f"{OUT}/analysis.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(inv.to_string(index=False))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("E2")}, indent=1, default=float))
    print(t1.to_string(index=False)); print(t2.to_string(index=False))
    for k in ("E2_all_21", "E2_no_fltrust_aligned_18"):
        print(k, {kk: res[k][kk] for kk in ("H_insample", "H_loso", "H_loso_ci95", "dstar_insample")})


if __name__ == "__main__":
    main()
