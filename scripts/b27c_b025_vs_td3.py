"""
B2.7c (reformulated, CONFIRMATORY): does the fixed threshold b = 0.25 beat td3_ref at H = 150?
Plan in results/b27c_b025_vs_td3/PLANO.md (b fixed beforehand from B2.6; fresh seeds 82-91).

Systems (3 cells × seeds 82-91, H = 150, the ablation learner, 60 runs):
  td3_ref     ablation learner in td3 mode, as in B2.7/B2.7a (no patch)
  fixed_b025  a = [0.5]*4, b = 0.25 through the same constant path as B2.7b
              (scripts/b27b_explicativo.constant_for)

Primary test: D = fixed_b025 − td3_ref paired by seed, one-sided t per cell, Holm over
the 3 cells (family α = 0.05); confirmed if Holm rejects in ≥ 2/3 and no mean D < 0.
Secondary: one-sided t on the per-seed mean of D over the 3 cells.

Usage (CPU):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.b27c_b025_vs_td3 run --alpha 0.05 --attack label_flipping --seed 82 --system td3_ref
  venv/bin/python -m scripts.b27c_b025_vs_td3 verificar   # 15-round smoke on seed 42 vs B2.7 / B2.7b
  venv/bin/python -m scripts.b27c_b025_vs_td3 analisar
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/b27c_b025_vs_td3"
RAW = f"{OUT}/raw"
CELLS = [(0.05, "label_flipping"), (0.1, "label_flipping"), (0.05, "low_mag_backdoor")]
SEEDS = list(range(82, 92))
SYSTEMS = ["td3_ref", "fixed_b025"]
HORIZONS = [15, 50, 150]
H_MAX = 150
ALPHA_FAMILY = 0.05
B27_RAW = "results/b27_horizonte/grade_raw.csv"
B27B_RAW = "results/b27b_td3_constante/b27b_raw"


def run(alpha, attack, seed, system, n_rounds=H_MAX, out_dir=RAW):
    from scripts.b27_horizonte import BYZ, INPUT_SHAPE, _reseed
    from scripts.b27b_explicativo import constant_for
    from scripts.frente1_ablacao_adaaggrl import build_learner_cls
    from src.experiments.exp9_dominance_grid import _split_name
    from src.utils.data_loader import load_dataset_participants

    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{system}_{attack}_a{alpha}_seed{seed}_R{n_rounds}.csv"
    if os.path.exists(path):
        print(f"already exists: {path}", flush=True)
        return path
    Ablation = build_learner_cls()
    participants, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    _reseed(seed)  # as in B2.7, before each system
    learner = Ablation(input_shape=INPUT_SHAPE, dataset="mnist", mode="td3", n_rounds=n_rounds, n_classes=10,
                       attack_type=attack, byzantine_ids=BYZ, seed=seed)
    if system == "fixed_b025":
        const = constant_for("fixed_b025", alpha, attack)
        learner.agent.select_action = lambda state, *a, **k: const.copy()
        learner.agent.train_step = lambda *a, **k: None
    elif system != "td3_ref":
        raise ValueError(system)
    t0 = time.time()
    res = learner.train(participants, root_data=root, verbose=False)
    rows = [{"alpha": alpha, "attack_type": attack, "seed": seed, "system": system, "H": H,
             "accuracy": float(res[H - 1].global_accuracy), "sec": time.time() - t0}
            for H in HORIZONS if len(res) >= H]
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    print(f"END {system} {attack} a{alpha} seed{seed} ({time.time() - t0:.0f}s)", flush=True)
    return path


def verificar():
    """15-round smoke, seed 42, label_flipping α 0.05: must reproduce B2.7 td3_ref and B2.7b fixed_b025 at H = 15."""
    alpha, attack, seed = 0.05, "label_flipping", 42
    ref = {"td3_ref": pd.read_csv(B27_RAW),
           "fixed_b025": pd.read_csv(f"{B27B_RAW}/fixed_b025_{attack}_a{alpha}_seed{seed}_R150.csv")}
    lines = []
    for system in SYSTEMS:
        p = run(alpha, attack, seed, system, n_rounds=15, out_dir=f"{OUT}/verificacao")
        new = float(pd.read_csv(p).query("H == 15").accuracy.iloc[0])
        r = ref[system]
        old = float(r[(r.alpha == alpha) & (r.attack_type == attack) & (r.seed == seed)
                      & (r.system == system) & (r.H == 15)].accuracy.iloc[0])
        lines.append(f"{system}: new {new:.6f} vs reference {old:.6f} |Δ| = {abs(new - old):.2e} "
                     + ("OK" if abs(new - old) < 1e-9 else "FAIL"))
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/verificacao.txt", "w").write(text)
    print(text)


def _holm(pvals, alpha):
    order = np.argsort(pvals)
    reject = np.zeros(len(pvals), bool)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (len(pvals) - rank):
            reject[i] = True
        else:
            break
    return reject


def analisar():
    allr = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RAW}/*_R{H_MAX}.csv"))])
    allr.to_csv(f"{OUT}/grade_raw.csv", index=False)
    n_runs = len(allr[allr.H == H_MAX])

    rows, d_by_cell = [], {}
    for H in HORIZONS:
        for alpha, attack in CELLS:
            g = allr[(allr.H == H) & (allr.alpha == alpha) & (allr.attack_type == attack)]
            w = g.pivot_table(index="seed", columns="system", values="accuracy")
            d = (100 * (w["fixed_b025"] - w["td3_ref"])).dropna()
            n, m, s = len(d), d.mean(), d.std(ddof=1)
            se = s / np.sqrt(n)
            row = {"H": H, "alpha": alpha, "attack_type": attack, "n": n,
                   "td3_ref": 100 * w["td3_ref"].mean(), "fixed_b025": 100 * w["fixed_b025"].mean(),
                   "D_pp": m, "ic95_lo": m - stats.t.ppf(0.975, n - 1) * se,
                   "ic95_hi": m + stats.t.ppf(0.975, n - 1) * se,
                   "ic90_unilat_lo": m - stats.t.ppf(0.95, n - 1) * se, "d_z": m / s,
                   "p_t_unilat": float(stats.ttest_1samp(d, 0, alternative="greater").pvalue),
                   "p_wilcoxon_unilat": float(stats.wilcoxon(d, alternative="greater").pvalue)}
            rows.append(row)
            if H == H_MAX:
                d_by_cell[(alpha, attack)] = d
    r = pd.DataFrame(rows)
    prim = r[r.H == H_MAX].copy()
    prim["holm_rejeita"] = _holm(prim.p_t_unilat.to_numpy(), ALPHA_FAMILY)
    r = r.merge(prim[["H", "alpha", "attack_type", "holm_rejeita"]], how="left")
    r.to_csv(f"{OUT}/deltas.csv", index=False)

    n_rej = int(prim.holm_rejeita.sum())
    any_neg = bool((prim.D_pp < 0).any())
    if n_rej == 3 and not any_neg:
        verdict = "H1 CONFIRMED in all cells (b = 0.25 beats td3_ref)"
    elif n_rej >= 2 and not any_neg:
        verdict = "H1 CONFIRMED (b = 0.25 beats td3_ref)"
    else:
        verdict = "H1 NOT confirmed (not evidence of equivalence)"

    dbar = pd.concat(d_by_cell.values(), axis=1).dropna().mean(axis=1)
    p_agg = float(stats.ttest_1samp(dbar, 0, alternative="greater").pvalue)

    # descriptive: same D in B2.7b (seeds 42-51)
    old = pd.read_csv("results/b27b_td3_constante/b27b_grade_raw.csv")
    old = old[old.H == H_MAX]
    prev = []
    for (alpha, attack), g in old.groupby(["alpha", "attack_type"]):
        w = g.pivot_table(index="seed", columns="system", values="accuracy")
        prev.append({"alpha": alpha, "attack_type": attack,
                     "D_pp_b27b_seeds42_51": 100 * (w["fixed_b025"] - w["td3_ref"]).mean()})
    cmp_ = prim[["alpha", "attack_type", "D_pp"]].merge(pd.DataFrame(prev)).rename(columns={"D_pp": "D_pp_seeds82_91"})

    pd.set_option("display.width", 220)
    lines = ["B2.7c reformulated (CONFIRMATORY) — fixed_b025 vs td3_ref, seeds 82–91, 3 cells",
             f"runs at H = {H_MAX}: {n_runs} (expected 60)", "",
             "D = fixed_b025 − td3_ref (p.p.), paired by seed; one-sided t; Holm over 3 cells at H = 150:",
             r.round(4).to_string(index=False), "",
             f"PRIMARY (H1): Holm rejects in {n_rej}/3 cells; any mean D < 0: {any_neg} -> {verdict}",
             f"SECONDARY (H1b): mean D over cells per seed = {dbar.mean():.3f} p.p. (n = {len(dbar)}), "
             f"one-sided p = {p_agg:.4g} -> " + ("CONFIRMED" if p_agg <= 0.05 else "NOT confirmed"), "",
             "Descriptive: D at H = 150 vs B2.7b (seeds 42–51; b chosen in hindsight there):",
             cmp_.round(3).to_string(index=False)]
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/analise.txt", "w").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--alpha", type=float, required=True)
    r.add_argument("--attack", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--system", required=True, choices=SYSTEMS)
    r.add_argument("--n_rounds", type=int, default=H_MAX)
    r.add_argument("--out_dir", default=RAW)
    sub.add_parser("verificar")
    sub.add_parser("analisar")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.alpha, a.attack, a.seed, a.system, a.n_rounds, a.out_dir)
    elif a.cmd == "verificar":
        verificar()
    else:
        analisar()


if __name__ == "__main__":
    main()
