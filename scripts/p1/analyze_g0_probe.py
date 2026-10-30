"""
Reads the FLTrust server-step diagnostic (results/p1_cifar10/e0b/ADDENDUM3.md,
"How it is read"); descriptive only. Writes results/p1_g0_probe/analysis.txt
and per_cell.csv.

Usage: python scripts/p1/analyze_g0_probe.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd

RAW = "results/p1_g0_probe/raw"
E0B = "results/p1_cifar10/e0b/raw"
OUT = "results/p1_g0_probe"
STEPS = ["native", "20", "470", "median"]


def load():
    rows = []
    for p in glob.glob(f"{RAW}/*.json"):
        d = json.load(open(p))
        rows.append({**{k: d[k] for k in ("dataset", "alpha", "attack", "root_size", "server_steps",
                                          "seed", "server_steps_n", "rounds")},
                     "acc": 100 * d["accuracy"][-1], "accuracy": d["accuracy"],
                     "g0": float(np.median(d["g0_l2"])), "g0_first": d["g0_l2"][0],
                     "agg_l2": float(np.median(d["agg_l2"])),
                     "client_med": float(np.median(d["update_l2"]))})
    return pd.DataFrame(rows)


def cifar(df, out):
    c = df[df.dataset == "cifar10"]
    out.append(f"## CIFAR-10 (alpha 0.05, seed 0, native, {len(c)} jobs)")
    for _, r in c.sort_values(["attack", "root_size"]).iterrows():
        ref = json.load(open(f"{E0B}/a0.05_{r.attack}_fltrust_e2_s0_r{r.root_size}.json"))["accuracy"]
        same = ref[:r.rounds] == r.accuracy
        out.append(f"{r.attack:14s} root {r.root_size:5d} steps {r.server_steps_n:4d}  "
                   f"||g0|| median {r.g0:.3f} (r1 {r.g0_first:.3f})  ||agg|| median {r.agg_l2:.3f}  "
                   f"client median {r.client_med:.3f}  regression vs E0b r1-{r.rounds}: "
                   f"{'equal' if same else 'DIFFERENT'}")


def mnist(df, out):
    m = df[df.dataset == "mnist"]
    out.append(f"\n## MNIST ({len(m)} jobs; final accuracy at round 15, %)")
    out.append("Server steps and norms (median over cells, seeds and rounds):")
    for (r, st), g in m.groupby(["root_size", "server_steps"]):
        out.append(f"  root {r:5d} steps {st:>6s} (n = {int(g.server_steps_n.median()):4d})  "
                   f"||g0|| {g.g0.median():7.3f}  ||agg|| {g.agg_l2.median():7.3f}  "
                   f"client {g.client_med.median():.3f}  acc mean {g.acc.mean():.2f}")
    acc = m.pivot_table(index=["alpha", "attack", "seed"], columns=["root_size", "server_steps"],
                        values="acc")
    cell = pd.DataFrame(index=acc.index)
    for st in STEPS:
        cell[f"D_root@{st}"] = acc[(3000, st)] - acc[(100, st)]
    cell["D_native"] = cell.pop("D_root@native")
    for r in (100, 3000):
        cell[f"step470-20@root{r}"] = acc[(r, "470")] - acc[(r, "20")]
        cell[f"median-native@root{r}"] = acc[(r, "median")] - acc[(r, "native")]
    # 2x2 {root 100, 3000} x {20, 470 steps} (ADDENDUM4): main effects and interaction
    cell["main_step"] = (cell["step470-20@root100"] + cell["step470-20@root3000"]) / 2
    cell["main_root"] = (cell["D_root@20"] + cell["D_root@470"]) / 2
    cell["interaction"] = cell["D_root@470"] - cell["D_root@20"]
    per_cell = cell.groupby(["alpha", "attack"]).mean()
    per_cell.round(2).to_csv(f"{OUT}/per_cell.csv")
    cols = ["D_native", "D_root@20", "D_root@470", "D_root@median",
            "step470-20@root100", "step470-20@root3000", "median-native@root100",
            "median-native@root3000", "main_step", "main_root", "interaction"]
    out.append("\nContrasts (p.p.; D_root@N = FLTrust(3000, N) - FLTrust(100, N); seed mean):")
    out.append(per_cell[cols].round(2).to_string())
    for label, sub in (("all 21 cells", per_cell),
                       ("18 cells without fltrust_aligned",
                        per_cell[per_cell.index.get_level_values("attack") != "fltrust_aligned"])):
        out.append(f"\nMean over {label}:")
        out.append(sub[cols].mean().round(2).to_string())
        out.append("by alpha:")
        out.append(sub[cols].groupby(level="alpha").mean().round(2).to_string())
    pos = (per_cell[cols[:4]] > 0).sum()
    out.append("\nCells with a positive root contrast (of 21):\n" + pos.to_string())


def main():
    df = load()
    out = [f"FLTrust server-step diagnostic (results/p1_cifar10/e0b/ADDENDUM3.md); {len(df)} jobs\n"]
    cifar(df, out)
    mnist(df, out)
    text = "\n".join(out)
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/analysis.txt", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
