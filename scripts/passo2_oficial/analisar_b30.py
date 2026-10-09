"""
B3.0 analysis (sanity), following results/b31_medmnist_oficial/PREREGISTRO.md §3.
Mechanically applies the rules fixed before any run:
  T = median accuracy over rounds 401-500 (uniform FedAvg), seeds 130-132.
  Convergence: mean T (BloodMNIST, no attack) ≥ 50% AND |mean 451-500 − mean
    401-450| < 2 p.p. (mean across seeds); otherwise: stop and consult the author.
  Inclusion: an attack is included if mean T under attack ≤ mean T without attack − 5 p.p.
  Margin: M = 1.0 × (100 − T_Blood) / (100 − T_MNIST), rounded up to the next
    multiple of 0.25 p.p., bounded to [1.0; 3.0].

Usage (official venv): external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_b30.py
"""

import json
import math
import os

import numpy as np

OUT = "results/b30_medmnist_sanity"
RAW = f"{OUT}/raw"
SEEDS = [130, 131, 132]
ROUNDS = 500


def _load(dataset, attack, seed):
    p = f"{RAW}/{dataset}_{attack}_q0.5_fedavg_seed{seed}_R{ROUNDS}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    acc = np.array([s["acc"] for s in d["steps"][:ROUNDS]])
    return {"T": float(np.median(acc[400:500])), "m401_450": float(acc[400:450].mean()),
            "m451_500": float(acc[450:500].mean()), "resets": len(d["resets"]) - 1,
            "sec_por_rodada": float(np.mean([s["sec"] for s in d["steps"]]))}


def main():
    lines = ["B3.0 — BloodMNIST sanity (uniform FedAvg, 500 rounds, seeds 130–132)", ""]
    res = {}
    for ds, att in [("BloodMNIST", "none"), ("BloodMNIST", "LMP"), ("BloodMNIST", "EB"), ("MNIST", "none")]:
        rs = [_load(ds, att, s) for s in SEEDS]
        if any(r is None for r in rs):
            raise SystemExit(f"missing runs for {ds}/{att}")
        res[(ds, att)] = rs
        lines.append(f"{ds:10s} {att:4s}: T per seed = {[round(100 * r['T'], 2) for r in rs]}%, "
                     f"mean {100 * np.mean([r['T'] for r in rs]):.2f}%; resets {[r['resets'] for r in rs]}; "
                     f"s/round {np.mean([r['sec_por_rodada'] for r in rs]):.1f}")
    T_blood = float(np.mean([r["T"] for r in res[("BloodMNIST", "none")]]))
    T_mnist = float(np.mean([r["T"] for r in res[("MNIST", "none")]]))
    drift = float(np.mean([r["m451_500"] - r["m401_450"] for r in res[("BloodMNIST", "none")]]))
    conv = T_blood >= 0.50 and abs(100 * drift) < 2.0
    lines += ["", f"Convergence: T_Blood = {100 * T_blood:.2f}% (≥ 50%?) and mean 451–500 − 401–450 = "
                  f"{100 * drift:+.2f} p.p. (|·| < 2?) -> " + ("CONVERGES" if conv else "does NOT converge: stop and consult the author")]
    incl = []
    for att in ["LMP", "EB"]:
        Ta = float(np.mean([r["T"] for r in res[("BloodMNIST", att)]]))
        ok = (T_blood - Ta) >= 0.05
        incl.append(att) if ok else None
        lines.append(f"Inclusion {att}: drop = {100 * (T_blood - Ta):+.2f} p.p. (≥ 5?) -> " + ("INCLUDED" if ok else "NOT included"))
    raw_m = 1.0 * (1 - T_blood) / (1 - T_mnist)
    M = min(3.0, max(1.0, math.ceil(raw_m / 0.25 - 1e-9) * 0.25))
    lines += [f"Margin: T_MNIST = {100 * T_mnist:.2f}%; raw M = {raw_m:.3f} p.p. -> M = {M:.2f} p.p."
              + (" (capped at 3.0: weak equivalence)" if raw_m > 3.0 else ""), "",
              "DECISION: " + ("B3.1 with attacks " + str(incl) + f" and margin ±{M:.2f} p.p." if conv and incl
                             else "stop and consult the author (gate not met)")]
    text = "\n".join(lines) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/analise.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
