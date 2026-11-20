"""
Analysis of E4a (results/p1_e4a/PLAN.md, Sec. 5). Per α, not pooled; per-seed differences, no Holm
family and no verdict category, since there is no confirmatory test.

Usage: venv/bin/python scripts/p1/analyze_e4a.py [--base results/p1_e4a] [--seeds 42 ...] [--phase_len 15]
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import inference as inf  # noqa: E402

FIXED = ["fltrust", "median", "trimmed_mean", "krum"]
GREEDY = "greedy"
ALPHAS = (0.1, 0.5)
ARMS = [GREEDY] + FIXED
PHASES = 3
WINDOW = 5


def load(base, seeds, phase_len):
    frames = []
    for s in seeds:
        d = f"{base}/raw/seed{s}"
        if not os.path.exists(f"{d}/done"):
            raise SystemExit(f"incomplete job: {d}")
        frames.append(pd.read_csv(f"{d}/per_round.csv"))
    df = pd.concat(frames, ignore_index=True)
    expected = len(seeds) * len(ALPHAS) * len(ARMS) * PHASES * phase_len
    if len(df) != expected or df.accuracy.isna().any():
        raise SystemExit(f"grid incomplete: {len(df)} rows, expected {expected}")
    return df


def phase_means(df, phase_len):
    """Mean accuracy in the last WINDOW rounds of each phase, per (alpha, arm, seed, phase)."""
    last = df[df["round"] > (df.phase - 1) * phase_len + phase_len - WINDOW]
    return last.groupby(["alpha", "arm", "seed", "phase"]).accuracy.mean().reset_index()


def report(x):
    d = inf.describe(x.values)
    return {"n": d["n"], "mean_pp": 100 * d["mean"], "ci95_lo_pp": 100 * d["ci95_lo"], "ci95_hi_pp": 100 * d["ci95_hi"],
            "p_wilcoxon": d["p_wilcoxon"], "p_t": d["p_t"], "d_z": d["d_z"]}


def headroom(score, alpha):
    """H_T of a per-seed score against the in-sample best fixed rule and the rule chosen on the other seeds."""
    a = score[score.alpha == alpha].pivot(index="seed", columns="arm", values="value")
    best = a[FIXED].mean().idxmax()
    cross = pd.Series({s: a.loc[s, a[FIXED].drop(s).mean().idxmax()] for s in a.index})
    return (a[GREEDY] - a[best]), best, (a[GREEDY] - cross), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/p1_e4a")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--phase_len", type=int, default=15)
    args = ap.parse_args()
    out_dir = f"{args.base}/analysis"
    os.makedirs(out_dir, exist_ok=True)
    df = load(args.base, args.seeds, args.phase_len)
    pm = phase_means(df, args.phase_len)
    r_end = PHASES * args.phase_len

    scores = {
        "phase score A (primary)": pm.groupby(["alpha", "arm", "seed"]).accuracy.mean().reset_index(name="value"),
        f"trailing, last {WINDOW} rounds": df[df["round"] > r_end - WINDOW].groupby(
            ["alpha", "arm", "seed"]).accuracy.mean().reset_index(name="value"),
        "final round": df[df["round"] == r_end][["alpha", "arm", "seed", "accuracy"]].rename(columns={"accuracy": "value"}),
    }
    out, rows = [f"E4a analysis (results/p1_e4a/PLAN.md). Seeds {args.seeds}. Constructed headroom, not confirmatory."], []
    for name, score in scores.items():
        out += ["", f"## {name}"]
        for alpha in ALPHAS:
            insample, best, cross, a = headroom(score, alpha)
            for ref, x in ((f"in-sample best fixed ({best})", insample), ("cross-fitted fixed", cross)):
                r = report(x)
                rows.append({"metric": name, "alpha": alpha, "reference": ref, **r})
                out.append(f"alpha {alpha}, H_T against {ref}: {r['mean_pp']:+.2f} p.p., CI95 ({r['ci95_lo_pp']:+.2f}; "
                           f"{r['ci95_hi_pp']:+.2f}), Wilcoxon p {r['p_wilcoxon']:.4f}, t p {r['p_t']:.4f}, d_z {r['d_z']:.2f}")
            means = a.mean().mul(100).round(2).to_dict()
            out.append(f"  arm means %: {means}")
    pd.DataFrame(rows).to_csv(f"{out_dir}/headroom.csv", index=False)

    out += ["", "## Per phase: last 5 rounds, arm means %, and whether G's rule still wins its phase among the fixed rules"]
    table = pm.groupby(["alpha", "phase", "arm"]).accuracy.mean().mul(100).round(2).unstack()[ARMS]
    table.to_csv(f"{out_dir}/phases.csv")
    out.append(table.to_string())
    rules = df[df.arm == GREEDY].groupby(["alpha", "phase"]).rule.first()
    for (alpha, phase), rule in rules.items():
        row = table.loc[(alpha, phase)]
        winner = row[FIXED].idxmax()
        out.append(f"alpha {alpha}, phase {phase}: G's rule {rule}; best fixed in this phase {winner}"
                   f" ({'same' if winner == rule else 'different'}); G {row[GREEDY]}, fixed {rule} {row[rule]}")

    curves = df.groupby(["alpha", "arm", "round"]).accuracy.mean().mul(100).round(3).reset_index()
    curves.to_csv(f"{out_dir}/curves.csv", index=False)
    text = "\n".join(out)
    with open(f"{out_dir}/analysis.txt", "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
