"""
P1, Fig. 2 (revision): counts recomputed with the corrected metric.

  O(c) = max_d A(d, c)    best fixed rule of the cell in hindsight (FLTrust, Krum, Median,
                           Trimmed mean; exp9, root 100, 10 seeds)
  B(c) = A(d*, c)         d* = fixed rule with the highest global mean (--dstar argument for sensitivity)
  R_s(c) = [A_s(c) − B(c)] / [O(c) − B(c)]
  Ceiling  = number of cells with O(c) > B(c)
  capture  = R_s(c) > 0, counted only in the cells with headroom
  Discrete = GRADF or FedStrategist; Continuous = AdaAggRL; Captured = any deployable system

Data: results/tables/exp9_dominance_grid_10seeds_ALL_root100_raw.csv (fixed rules) and
results/tables/exp10_FULL_variantb_raw.csv from tag p1.0.0 (systems).

Usage: venv/bin/python scripts/p1_fig2_recalculo.py [--dstar trimmed_mean|fltrust|...] [--sem_fltrust_aligned]
"""

import argparse
import io
import subprocess

import pandas as pd

FIXOS = "results/tables/exp9_dominance_grid_10seeds_ALL_root100_raw.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dstar", default=None, help="global reference rule (default: highest global mean)")
    ap.add_argument("--sem_fltrust_aligned", action="store_true")
    a = ap.parse_args()
    fx = pd.read_csv(FIXOS)
    sy = pd.read_csv(io.StringIO(subprocess.check_output(
        ["git", "show", "p1.0.0:results/tables/exp10_FULL_variantb_raw.csv"]).decode()))
    if a.sem_fltrust_aligned:
        fx, sy = fx[fx.attack_type != "fltrust_aligned"], sy[sy.attack_type != "fltrust_aligned"]
    A = fx.groupby(["alpha", "attack_type", "strategy"]).accuracy.mean().unstack()
    S = sy.groupby(["alpha", "attack_type", "system"]).accuracy.mean().unstack()
    medias = fx.groupby("strategy").accuracy.mean()
    dstar = a.dstar or medias.idxmax()
    O, B = A.max(axis=1), A[dstar]
    head = [c for c in A.index if O[c] - B[c] > 1e-12]
    cap = {s: {c for c in head if S.loc[c, s] > B[c]} for s in ["GRADF", "FedStrategist", "AdaAggRL"]}
    disc, cont = cap["GRADF"] | cap["FedStrategist"], cap["AdaAggRL"]
    n = len(A)
    print("global means of the fixed rules:", medias.round(4).to_dict(), "-> d* =", dstar)
    print(f"Ceiling {len(head)}/{n} | Captured {len(disc | cont)}/{n} | Discrete {len(disc)}/{n} "
          f"(GRADF {len(cap['GRADF'])}, FedStrat {len(cap['FedStrategist'])}) | Continuous {len(cont)}/{n}")
    for c in head:
        r = {s: (S.loc[c, s] - B[c]) / (O[c] - B[c]) for s in ["GRADF", "FedStrategist", "AdaAggRL"]}
        print(f"  α={c[0]:<4} {c[1]:17s} O−B={100 * (O[c] - B[c]):5.1f} p.p. | "
              + "  ".join(f"{s} R={v:+.2f}" for s, v in r.items()))


if __name__ == "__main__":
    main()
