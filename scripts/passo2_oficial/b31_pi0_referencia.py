"""
B3.2, post-hoc reference (descriptive, NOT pre-registered): sd_estados, S_swap and S_shuffle
of the INITIAL actor π₀ on the same states used for π₅₀₀ in analisar_b31.py::b32.
Used to compare the sensitivity of the final policy with that of a freshly initialized network
(like B2.2's "S_swap of the initial actor" reference on MNIST).

Usage (official venv):
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/b31_pi0_referencia.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analisar_b21 import make_actor_evaluator  # noqa: E402

OUT = "results/b31_medmnist_oficial"
RAW = f"{OUT}/raw"
SEEDS = list(range(135, 145))
ATAQUES = ["LMP", "EB"]
ROUNDS = 500


def _tag(att, s):
    return f"BloodMNIST_{att}_q0.5_td3_seed{s}_R{ROUNDS}"


def main():
    policy = make_actor_evaluator()
    rows = []
    for att in ATAQUES:
        other = [a for a in ATAQUES if a != att][0]
        for s in SEEDS:
            tag = _tag(att, s)
            p0, p5 = f"{RAW}/actors/{tag}_step000.pt", f"{RAW}/actors/{tag}_step500.pt"
            full = np.load(f"{RAW}/obs/{tag}.npy")[:ROUNDS]
            S = full[400:500]
            S2 = np.load(f"{RAW}/obs/{_tag(other, s)}.npy")[400:500]
            idx = np.random.RandomState(0).randint(0, len(full), size=len(S))  # same draw as analisar_b31
            row = {"attack": att, "seed": s}
            for nome, p in (("pi0", p0), ("pi500", p5)):
                a = policy(p, S)
                row[f"sd_estados_{nome}"] = float(np.mean(a.std(axis=0)))
                row[f"S_swap_{nome}"] = float(np.mean(np.abs(a - policy(p, S2))))
                row[f"S_shuffle_{nome}"] = float(np.mean(np.abs(a - policy(p, full[idx]))))
            rows.append(row)
    m = pd.DataFrame(rows)
    m.to_csv(f"{OUT}/pi0_referencia.csv", index=False)
    cols = [c for c in m.columns if c not in ("attack", "seed")]
    lines = ["B3.2 — post-hoc reference (descriptive): π₀ vs. π₅₀₀ on the states of rounds 401–500",
             "", "Median per attack:", m.groupby("attack")[cols].median().round(4).T.to_string(),
             "", "Overall median (n=20):", m[cols].median().round(4).to_string(),
             "", "Ratio π₅₀₀/π₀ per run (median):"]
    for k in ("sd_estados", "S_swap", "S_shuffle"):
        r = m[f"{k}_pi500"] / m[f"{k}_pi0"]
        lines.append(f"  {k}: {r.median():.2f} (min {r.min():.2f}, max {r.max():.2f})")
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/pi0_referencia.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
