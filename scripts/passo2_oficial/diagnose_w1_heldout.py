"""
Post hoc diagnostic of W1's held-out series (results/w1_protocolo_independente_oficial/RESULTS.md).
Reads the raw JSONs only; no training. Run with the official venv:
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/diagnose_w1_heldout.py

HeldOutAdapter.step evaluates inner.net after the environment's step, when inner.net holds the update
of the last client of the NEXT round's sample, not the global model. Test 1: if so, on MNIST rounds
where the global model has converged, P(held-out acc < 50% at round i) = n/10, with n the real attackers
among the 10 clients aggregated at round i+1. Test 2: the same comparison at round i (no relation expected).
Also: fixed − td3 on the validation accuracy, which comes from the global model (descriptive).
"""

import glob
import json

import numpy as np
from scipy import stats

OUT = "results/w1_protocolo_independente_oficial"
ATTACKS = ("LMP", "EB")


def series(r):
    st = r["steps"][:500]
    return (np.array([x["test_acc"] for x in r["held_out"][:500]]), np.array([x["acc"] for x in st]),
            np.array([x["n_att_real"] for x in st]))


def main():
    lines, runs = [], [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/raw/*_R500.json"))]
    lines.append(f"{len(runs)} runs; clients per round: {sorted({int(r['num_clients'] * r['subsample_rate']) for r in runs})}")

    next_round, same_round, fedavg_low, sample = {}, {}, [0, 0], None
    for r in runs:
        if r["dataset"] != "MNIST":
            continue
        t, v, n = series(r)
        if r["attack"] == "none":
            fedavg_low[0] += int((t[100:] < 0.5).sum())
            fedavg_low[1] += len(t[100:])
            continue
        for i in range(100, 499):
            if v[i] > 0.9 and v[i + 1] > 0.9:
                for tab, k in ((next_round, int(n[i + 1])), (same_round, int(n[i]))):
                    c = tab.setdefault(k, [0, 0])
                    c[0] += int(t[i] < 0.5)
                    c[1] += 1
        if (r["attack"], r["condition"], r["seed"]) == ("EB", "fixed", 160):
            sample = (t[470:500], v[470:500])

    lines.append("\nMNIST, attacked runs, rounds >= 100, validation acc > 90% at i and i+1:")
    lines.append("  n   rounds  P(held-out<50% | n at i+1)  P(held-out<50% | n at i)")
    for k in sorted(next_round):
        a, b = next_round[k], same_round.get(k, [0, 0])
        lines.append(f"  {k}   {a[1]:6d}  {a[0] / a[1]:26.3f}  {b[0] / b[1] if b[1] else float('nan'):24.3f}")
    lines.append(f"MNIST, fedavg/none, rounds >= 100: held-out<50% in {fedavg_low[0]}/{fedavg_low[1]} rounds")
    lines.append("\nMNIST EB fixed seed 160, rounds 471-500:")
    lines.append("  held-out   " + " ".join(f"{x:.2f}" for x in sample[0]))
    lines.append("  validation " + " ".join(f"{x:.2f}" for x in sample[1]))

    rows = []
    for r in runs:
        t, v, _ = series(r)
        rows.append((r["dataset"], r["attack"], r["condition"], r["seed"], float(np.median(v[400:500])), float(v.mean()),
                     float(np.median(t[400:500]))))
    lines.append("\nMeans per dataset x attack x condition (validation = global model; held-out = last client's model):")
    lines.append("  dataset     attack cond    val_median_401_500  val_mean_1_500  held_out_median_401_500")
    for key in sorted({x[:3] for x in rows}):
        g = np.array([x[4:] for x in rows if x[:3] == key])
        lines.append(f"  {key[0]:11s} {key[1]:6s} {key[2]:7s} {g[:, 0].mean():18.4f} {g[:, 1].mean():15.4f} {g[:, 2].mean():24.4f}")

    lines.append("\nfixed - td3 on validation, global model, n = 20 per dataset (p.p.; descriptive, post hoc):")
    for d in ("MNIST", "BloodMNIST"):
        for col, name in ((4, "median 401-500"), (5, "mean 1-500")):
            val = {(x[1], x[2], x[3]): x[col] for x in rows if x[0] == d}
            x = 100 * np.array([val[(a, "fixed", s)] - val[(a, "td3", s)] for (a, c, s) in val if c == "fixed" and a in ATTACKS])
            se = x.std(ddof=1) / np.sqrt(len(x))
            t90, t95 = stats.t.ppf(0.95, len(x) - 1) * se, stats.t.ppf(0.975, len(x) - 1) * se
            lines.append(f"  {d} {name}: D={x.mean():+.2f} CI90=({x.mean() - t90:+.2f}; {x.mean() + t90:+.2f}) "
                         f"CI95=({x.mean() - t95:+.2f}; {x.mean() + t95:+.2f}) t p={stats.ttest_1samp(x, 0).pvalue:.4f}")
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/heldout_diagnosis.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
