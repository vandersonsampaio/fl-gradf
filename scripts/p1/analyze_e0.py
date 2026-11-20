"""
E0 analysis (results/p1_plan/PLAN.md, Sec. 5, E0 and gate P-0), CIFAR-10.

  curves:     FedAvg without attack, alpha in {0.5, 0.1, 0.05}, local epochs
              1 and 2, seeds 0-2. Where does accuracy stabilize?
  separation: the 7 fixed rules under sign_flipping and trim_attack at
              alpha = 0.05, local epochs 2, seeds 0-2. Do the rules separate
              beyond the across-seed noise?

Descriptive only (3 seeds, outside the confirmatory block 42-51). The
stabilization rule and the separation check are defined here, before the
pre-registration, and are not confirmatory tests.

Usage: python scripts/p1/analyze_e0.py
Writes results/p1_cifar10/e0/{analysis.txt, curves_summary.csv,
separation_summary.csv, separation_pairs.csv, curves.png}
"""

import glob
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

E0 = "results/p1_cifar10/e0"
WINDOW = 5          # trailing window (rounds) for the smoothed accuracy
STABLE_PP = 1.0     # stabilized: within 1 p.p. of the final window
READ_AT = (20, 30, 40, 50)
CHANCE = 0.10


def load(mode):
    rows = []
    for path in sorted(glob.glob(f"{E0}/{mode}/*.json")):
        with open(path) as f:
            d = json.load(f)
        rows.append(d)
    return rows


def smooth(acc, r):
    """Mean accuracy over rounds r-WINDOW+1 .. r (1-based)."""
    return float(np.mean(acc[r - WINDOW:r]))


def stabilization_round(mean_curve):
    final = smooth(mean_curve, len(mean_curve))
    for r in range(WINDOW, len(mean_curve) + 1):
        if all(abs(smooth(mean_curve, q) - final) * 100 <= STABLE_PP
               for q in range(r, len(mean_curve) + 1)):
            return r
    return len(mean_curve)


def curves(out):
    runs = load("curves")
    lines, rows = ["## E0.1 FedAvg without attack (accuracy, %; mean ± SD over seeds)", ""], []
    by = {}
    for d in runs:
        by.setdefault((d["alpha"], d["local_epochs"]), []).append(d)
    for (alpha, ep), ds in sorted(by.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        acc = np.array([d["accuracy"] for d in ds])           # seeds x rounds
        mean = acc.mean(axis=0)
        row = {"alpha": alpha, "local_epochs": ep, "n_seeds": len(ds),
               "stable_round": stabilization_round(mean),
               "s_per_round": float(np.mean([d["seconds"] for d in ds]) / acc.shape[1])}
        for r in READ_AT:
            sm = [smooth(a, r) for a in acc]
            row[f"acc_r{r}"] = 100 * float(np.mean(sm))
            row[f"sd_r{r}"] = 100 * float(np.std(sm, ddof=1))
        row["gain_30_50_pp"] = row["acc_r50"] - row["acc_r30"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{E0}/curves_summary.csv", index=False)
    lines.append(f"Smoothed accuracy = mean of the last {WINDOW} rounds up to round r. "
                 f"Stable round = first r after which the smoothed seed-mean stays within "
                 f"{STABLE_PP} p.p. of its value at round 50.")
    lines.append("")
    hdr = "alpha  ep  " + "  ".join(f"r{r:<11}" for r in READ_AT) + "  gain30-50  stable  s/round"
    lines.append(hdr)
    for _, r in df.iterrows():
        cells = "  ".join(f"{r[f'acc_r{q}']:5.1f}±{r[f'sd_r{q}']:4.1f}  " for q in READ_AT)
        lines.append(f"{r.alpha:<5}  {int(r.local_epochs)}   {cells}  {r.gain_30_50_pp:+6.2f}    "
                     f"{int(r.stable_round):>4}    {r.s_per_round:5.1f}")
    out += lines + [""]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        for ax, ep in zip(axes, (1, 2)):
            for (alpha, e), ds in sorted(by.items()):
                if e != ep:
                    continue
                acc = 100 * np.array([d["accuracy"] for d in ds])
                x = np.arange(1, acc.shape[1] + 1)
                line, = ax.plot(x, acc.mean(axis=0), label=f"α = {alpha}")
                ax.fill_between(x, acc.min(axis=0), acc.max(axis=0), alpha=0.2, color=line.get_color())
            ax.axhline(100 * CHANCE, color="grey", lw=0.8, ls=":")
            ax.set_title(f"FedAvg, no attack, {ep} local epoch{'s' if ep > 1 else ''}")
            ax.set_xlabel("round")
        axes[0].set_ylabel("global accuracy (%)")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(f"{E0}/curves.png", dpi=150)
    except ImportError:
        out.append("(matplotlib not available: curves.png not written)")
    return df


def separation(out, read_at):
    runs = load("separation")
    rows = []
    for d in runs:
        acc = d["accuracy"]
        rows.append({"attack": d["attack"], "rule": d["rule"], "seed": d["seed"],
                     **{f"acc_r{r}": 100 * smooth(acc, r) for r in read_at}})
    raw = pd.DataFrame(rows)
    summ, pairs = [], []
    t_crit = stats.t.ppf(0.975, df=2)
    for r in read_at:
        col = f"acc_r{r}"
        out += [f"## E0.2 Separation at alpha = 0.05, round {r} (accuracy %, smoothed over "
                f"{WINDOW} rounds; 3 seeds)", ""]
        for attack, g in raw.groupby("attack"):
            wide = g.pivot(index="seed", columns="rule", values=col)
            means = wide.mean().sort_values(ascending=False)
            sds = wide.std(ddof=1)
            pooled_sd = float(np.sqrt(np.mean(sds ** 2)))
            out.append(f"{attack}: pooled across-seed SD = {pooled_sd:.2f} p.p.; "
                       f"range of rule means = {means.max() - means.min():.2f} p.p.")
            for rule in means.index:
                out.append(f"  {rule:<13} {means[rule]:6.2f} ± {sds[rule]:5.2f}")
                summ.append({"round": r, "attack": attack, "rule": rule, "mean": means[rule],
                             "sd": sds[rule], "pooled_sd": pooled_sd})
            n_sep = 0
            for a, b in combinations(means.index, 2):
                diff = wide[a] - wide[b]                          # paired by seed (same partition)
                m, s = float(diff.mean()), float(diff.std(ddof=1))
                half = t_crit * s / np.sqrt(len(diff))
                sep = (m - half > 0) or (m + half < 0)
                n_sep += sep
                pairs.append({"round": r, "attack": attack, "rule_a": a, "rule_b": b,
                              "mean_diff": m, "sd_diff": s, "ci95_lo": m - half,
                              "ci95_hi": m + half, "separated": sep,
                              "same_sign_all_seeds": bool((diff > 0).all() or (diff < 0).all())})
            n_pairs = len(means) * (len(means) - 1) // 2
            best, second = means.index[0], means.index[1]
            d_best = wide[best] - wide[second]
            out.append(f"  pairs separated (paired 95% CI excludes 0, t with 2 df): {n_sep}/{n_pairs}; "
                       f"best vs second ({best} − {second}): {d_best.mean():+.2f} p.p., "
                       f"per seed {', '.join(f'{v:+.2f}' for v in d_best)}")
            above = int((means > 100 * CHANCE + 5).sum())
            out.append(f"  rules more than 5 p.p. above chance (10%): {above}/{len(means)}")
            out.append("")
    pd.DataFrame(summ).to_csv(f"{E0}/separation_summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(f"{E0}/separation_pairs.csv", index=False)
    return raw


def sigma_e0(out, curves_df, sep_raw, read_at):
    out += ["## σ_E0 (across-seed SD, p.p.; root mean square over cells)", ""]
    for r in read_at:
        col = f"acc_r{r}"
        sds = sep_raw.groupby(["attack", "rule"])[col].std(ddof=1)
        line = f"round {r}: separation cells (n = {len(sds)}) {np.sqrt(np.mean(sds ** 2)):.2f}"
        cs = curves_df[curves_df.local_epochs == 2][f"sd_r{r}"] if f"sd_r{r}" in curves_df else None
        if cs is not None and len(cs):
            line += f"; FedAvg no-attack cells, 2 local epochs (n = {len(cs)}) {np.sqrt(np.mean(cs ** 2)):.2f}"
        out.append(line)
    out.append("")


def main():
    out = ["E0 analysis (CIFAR-10, small CNN, CPU, seeds 0-2). Descriptive; not confirmatory.", ""]
    cdf = curves(out)
    read_at = (30, 50)
    raw = separation(out, read_at)
    sigma_e0(out, cdf, raw, read_at)
    text = "\n".join(out)
    with open(f"{E0}/analysis.txt", "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
