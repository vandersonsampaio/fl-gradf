"""
Analysis of the MNIST rerun that records the trailing mean (results/p1_parity/mnist_trailing/ADDENDUM.md).

  Sec. 1  completeness
  Sec. 2  regression: the final-round column against ../mnist_fixed and ../mnist_rq3
  Sec. 3  E1 on the trailing mean and on the final round
  Sec. 4  E2: d*, in-sample and leave-one-seed-out headroom
  Sec. 5  E3 on the 18 cells and on the 12 cells of level 1
  Sec. 6  win counts and how far the two metrics disagree

AdaAggRL was not rerun and has no trailing mean on MNIST, so it is left out of the E3 families here.

Usage: venv/bin/python scripts/p1/analyze_mnist_trailing.py
"""

import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import inference as inf  # noqa: E402

BASE = "results/p1_parity"
RAW = f"{BASE}/mnist_trailing"
OUT = f"{RAW}/analysis"
FA = "fltrust_aligned"
LEVEL1 = ("sign_flipping", "label_flipping", "trim_attack", "krum_collusion")
SYSTEMS = ["Random", "Oracle (decoupled)", "GRADF", "FedStrategist"]
SENSITIVITY = ["FedStrategist (GRADF detector)", "Random"]
ROOTS = (100, 3000, 50)
METRICS = {"trailing": "accuracy_trailing5", "final": "accuracy"}
E3_COLS = ["family", "system", "mean", "ci95_lo", "ci95_hi", "p_wilcoxon", "p_wilcoxon_holm", "p_t", "d_z", "category"]


def load_fixed(base):
    frames = []
    for d in sorted(glob.glob(f"{base}/mnist_fixed/raw/root*_seed*")):
        if not os.path.exists(f"{d}/done"):
            raise SystemExit(f"incomplete job: {d}")
        root = int(os.path.basename(d).split("_")[0].removeprefix("root"))
        frames.append(pd.read_csv(glob.glob(f"{d}/*_stepsmedian_raw.csv")[0]).assign(root=root))
    return pd.concat(frames, ignore_index=True)


def load_systems(base):
    frames = []
    for v in "ba":
        for d in sorted(glob.glob(f"{base}/mnist_rq3/raw/variant{v}_seed*")):
            if not os.path.exists(f"{d}/done"):
                raise SystemExit(f"incomplete job: {d}")
            df = pd.read_csv(glob.glob(f"{d}/*_stepsmedian_raw.csv")[0])
            # Variant a is the arm's own diagnostic; variant b uses GRADF's detector.
            frames.append(df.replace({"system": {"FedStrategist": "FedStrategist (GRADF detector)"}}) if v == "b" else df)
    return pd.concat(frames, ignore_index=True)


def fmt(r):
    return (f"mean {100 * r['mean']:+6.2f} p.p., CI95 ({100 * r['ci95_lo']:+.2f}; {100 * r['ci95_hi']:+.2f}), "
            f"Wilcoxon p {r['p_wilcoxon']:.4f}, t p {r['p_t']:.4f}, d_z {r['d_z']:.2f}, {r['category']}")


def regression(fx, sy, out):
    out += ["", "## Sec. 2 Regression of the final-round column (must be exact, except GRADF's DQN)"]
    rows = []
    key = ["alpha", "attack_type", "strategy", "seed", "root"]
    m = fx.merge(load_fixed(BASE), on=key, suffixes=("_new", "_old"), validate="one_to_one")
    for name, g in m.groupby("strategy"):
        rows.append({"source": "mnist_fixed", "arm": name, "n_pairs": len(g),
                     "n_nonzero": int((g.accuracy_new != g.accuracy_old).sum()),
                     "max_abs": float((g.accuracy_new - g.accuracy_old).abs().max())})
    key = ["alpha", "attack_type", "system", "seed"]
    m = sy.merge(load_systems(BASE), on=key, suffixes=("_new", "_old"), validate="one_to_one")
    for name, g in m.groupby("system"):
        rows.append({"source": "mnist_rq3", "arm": name, "n_pairs": len(g),
                     "n_nonzero": int((g.accuracy_new != g.accuracy_old).sum()),
                     "max_abs": float((g.accuracy_new - g.accuracy_old).abs().max())})
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/regression.csv", index=False)
    out.append(t.to_string(index=False))


def e1(fx, res, out):
    out += ["", "## Sec. 3 E1 (V = FLTrust - TM; D_s over seeds)"]
    for metric, col in METRICS.items():
        f = fx.assign(accuracy=fx[col])
        src = {r: f[f.root == r].drop(columns="root") for r in ROOTS}
        for name, excl in (("21 cells", ()), ("18 cells", (FA,))):
            a, _ = inf.e1_root_dependence(src[100], src[3000], exclude=excl)
            b, _ = inf.e1_root_dependence(src[50], src[100], exclude=excl)
            res[f"E1 3000 vs 100 {metric} {name}"], res[f"E1 100 vs 50 {metric} {name}"] = a, b
            out.append(f"{metric:<9} {name:<9} 3000 vs 100: {fmt(a)}")
            out.append(f"{metric:<9} {name:<9}  100 vs  50: {fmt(b)}")


def e2(fx, res, out):
    out += ["", "## Sec. 4 E2 (root 100; D_R = FLTrust, median, TM, Krum)"]
    for metric, col in METRICS.items():
        f = fx[fx.root == 100].drop(columns="root").assign(accuracy=lambda d: d[col])
        for name, g in (("21 cells", f), ("18 cells", f[f.attack_type != FA])):
            r = {**inf.e2_headroom(g), **inf.e2_selection_stability(g)}
            res[f"E2 {metric} {name}"] = r
            means = g[g.strategy.isin(inf.D_R)].groupby("strategy").accuracy.mean().mul(100).round(2).to_dict()
            out.append(f"{metric} {name}: d* = {r['dstar_insample']} (LOSO agreement {r['dstar_agreement']:.2f}), "
                       f"cell-winner agreement {r['cell_winner_agreement']:.2f}")
            out.append(f"  global means %: {means}")
            out.append(f"  H in-sample {100 * r['H_insample']:.2f} p.p.; H_LOSO {100 * r['H_loso']:.2f} p.p., "
                       f"CI95 ({100 * r['H_loso_ci95'][0]:.2f}; {100 * r['H_loso_ci95'][1]:.2f}), {r['category']}")


def e3(fx, sy, res, out):
    out += ["", "## Sec. 5 E3 (accuracy differences as fractions; AdaAggRL has no trailing mean on MNIST)"]
    r100 = fx[fx.root == 100].drop(columns="root")
    grids = {"18 cells": lambda d: d[d.attack_type != FA], "12 cells, level 1": lambda d: d[d.attack_type.isin(LEVEL1)]}
    for metric, col in METRICS.items():
        for name, pick in grids.items():
            f, s = pick(r100).assign(accuracy=lambda d: d[col]), pick(sy).assign(accuracy=lambda d: d[col])
            t, desc = inf.e3_capture(f, s, SYSTEMS)
            ts, _ = inf.e3_capture(f, s, SENSITIVITY)
            stem = f"e3_{metric}_{name.split()[0]}cells"
            t.to_csv(f"{OUT}/{stem}.csv", index=False)
            ts.to_csv(f"{OUT}/{stem}_sensitivity.csv", index=False)
            res[f"E3 {metric} {name}"] = desc
            out += ["", f"{metric}, {name}:", t[E3_COLS].round(4).to_string(index=False), "Sensitivity:",
                    ts[E3_COLS].round(4).to_string(index=False), f"Descriptive: {desc}"]


def wins(fx, sy, out):
    out += ["", "## Sec. 6 Win counts by cell (D_R) and agreement between the metrics"]
    rows = []
    winners = {}
    for metric, col in METRICS.items():
        for root in ROOTS:
            f = fx[(fx.root == root) & fx.strategy.isin(inf.D_R)]
            w = f.groupby(["alpha", "attack_type", "strategy"])[col].mean().unstack().idxmax(axis=1)
            winners[metric, root] = w
            rows.append({"metric": metric, "root": root, **w.value_counts().reindex(inf.D_R, fill_value=0).to_dict()})
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/wins.csv", index=False)
    out.append(t.to_string(index=False))
    for root in ROOTS:
        same = int((winners["trailing", root] == winners["final", root]).sum())
        out.append(f"root {root}: same cell winner under both metrics in {same} of 21 cells")
    d = (sy.accuracy_trailing5 - sy.accuracy).abs() * 100
    out.append(f"|trailing - final| per run, p.p.: systems mean {d.mean():.2f}, max {d.max():.2f}")
    for rule in ("fltrust", "trimmed_mean", "median", "krum"):
        g = fx[(fx.root == 100) & (fx.strategy == rule)]
        d = (g.accuracy_trailing5 - g.accuracy) * 100
        out.append(f"{rule} root 100, trailing - final, p.p.: mean {d.mean():+.2f}, mean abs {d.abs().mean():.2f}, max abs {d.abs().max():.2f}")
    means = fx[fx.root == 100].groupby(["alpha", "attack_type", "strategy"])[list(METRICS.values())].mean().mul(100).round(2)
    means.to_csv(f"{OUT}/grid_root100_rules.csv")
    sy.groupby(["alpha", "attack_type", "system"])[list(METRICS.values())].mean().mul(100).round(2).to_csv(f"{OUT}/grid_systems.csv")


def main():
    os.makedirs(OUT, exist_ok=True)
    out = ["MNIST trailing-mean analysis (mnist_trailing/ADDENDUM.md). Seeds 42-51. Not blind: see the addendum.", ""]
    res = {}
    fx, sy = load_fixed(RAW), load_systems(RAW)
    n, m = fx.groupby("root").size().to_dict(), sy.groupby("system").size().to_dict()
    ok = n == {100: 1470, 3000: 840, 50: 840} and all(v == 210 for v in m.values())
    out += ["## Sec. 1 Completeness", f"rows per root: {n}; rows per system: {m}; "
            f"NaN in trailing column: {int(fx.accuracy_trailing5.isna().sum() + sy.accuracy_trailing5.isna().sum())}",
            "OK" if ok else "NOT OK"]
    if not ok:
        print("\n".join(out))
        sys.exit(1)
    regression(fx, sy, out)
    e1(fx, res, out)
    e2(fx, res, out)
    e3(fx, sy, res, out)
    wins(fx, sy, out)
    text = "\n".join(out)
    with open(f"{OUT}/analysis.txt", "w") as f:
        f.write(text + "\n")
    with open(f"{OUT}/analysis.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(text)


if __name__ == "__main__":
    main()
