"""
A0: cheap analyses that close loose ends of B2.6, B2.8 and C0.
Plan in results/a0_analises/PLANO.md.
Does not change `src/`, the ablation or B2.6 (reuses the B2.6 learner by inheritance).

Subcommands (project venv, CPU):
  massa       (a) runs sr_only and sr_bin on the 8 cells with a negative gap in C0,
                  seeds 52-54, recording each client's final weight per round
                  and is_byz; checks that the accuracy reproduces B2.6.
  analisar    (a) mass on attackers; (b) signal-selection ceiling (B2.6);
              (c) per-client agreement S_R x cos_server (ablation log).
Item (d) (three TD3 regimes) is in scripts/passo2_oficial/a0_regimes_td3.py
(official venv).

Usage:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.a0_analises massa --seed 52
  venv/bin/python -m scripts.a0_analises analisar
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/a0_analises"
B26_RAW = "results/b26_decomposicao/raw"
DETECT = "results/frente1_ablacao_adaaggrl/detectores"
CELLS = [(0.05, "gaussian_noise"), (0.05, "krum_collusion"), (0.05, "trim_attack"),
         (0.1, "gaussian_noise"), (0.1, "krum_collusion"), (0.1, "trim_attack"),
         (0.1, "sign_flipping"), (0.1, "low_mag_backdoor")]  # negative gap with CI95 < 0 in C0
SEEDS_MASSA = [52, 53, 54]
VARIANTS_MASSA = ["sr_only", "sr_bin"]
EXCLUDED = {(0.05, "fltrust_aligned"), (0.1, "fltrust_aligned")}
MASS_THRESHOLD = 0.05  # PLANO §3: real exploitation if the mean mass >= 0.05 (FedAvg would give ~0.2)


def build_recording_learner():
    from scripts import b26_decomposicao as B
    from src.defense.adaaggrl_agent import reconstruct_client_distribution
    from src.fl.attacked_learner import compute_param_updates_auto
    from src.fl.federated_learner import RoundResult

    Base = B.build_learner_cls()

    class RecordingDecompLearner(Base):
        """Same _run_round as B2.6 (literal copy of the logic, same RNG order),
        only adding the logging of the normalized weights and of is_byz."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rows = []

        def _run_round(self, round_num, participants, root_data):
            active = self._is_active(round_num)
            param_updates, is_byz = compute_param_updates_auto(self, participants, active, root_data)
            n_feat = participants[0].n_features
            K = 1 if self.n_classes == 2 else self.n_classes
            theta_list = [self._global_params + u for u in param_updates]
            n = len(participants)
            s_r = np.full(n, np.nan)
            if self.variant in B.NEEDS_SR:
                for i, (theta_k, u) in enumerate(zip(theta_list, param_updates)):
                    W = theta_k[: n_feat * K].reshape(n_feat, K).astype(np.float32)
                    b = theta_k[n_feat * K:].astype(np.float32)
                    _, s_r[i] = reconstruct_client_distribution(
                        W, b, u, self.local_lr, n_feat, K,
                        num_images=self.num_images, max_iters=self.max_iters, seed=round_num)
            if self.variant not in ("sr_only", "sr_bin"):
                raise ValueError("A0 only uses sr_only and sr_bin")
            if self._h is None:
                self._h = np.zeros(n)
            w, self._h = B.weights_for(self.variant, s_r, self._h)
            tw = float(w.sum())
            nw = np.ones(n) / n if tw < 1e-8 else w / tw
            self._global_params = sum(wi * th for wi, th in zip(nw, theta_list))
            for i in range(n):
                self.rows.append({"round": round_num, "client": i, "is_byz": bool(is_byz[i]),
                                  "S_R": float(s_r[i]), "weight": float(nw[i])})
            per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
            return RoundResult(round_num, float(np.mean(list(per_acc.values()))), per_acc,
                               trust_scores=None, n_accepted=int((w > 0).sum()))

    return RecordingDecompLearner


def massa(seed: int):
    from src.experiments.exp9_dominance_grid import _split_name
    from src.utils.data_loader import load_dataset_participants

    os.makedirs(f"{OUT}/massa", exist_ok=True)
    L = build_recording_learner()
    for alpha in sorted({a for a, _ in CELLS}):
        participants, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
        for a2, attack in CELLS:
            if a2 != alpha:
                continue
            for variant in VARIANTS_MASSA:
                path = f"{OUT}/massa/{variant}_{attack}_a{alpha}_seed{seed}.csv"
                if os.path.exists(path):
                    continue
                t0 = time.time()
                lr = L(n_rounds=15, n_classes=10, attack_type=attack, byzantine_ids=[0, 1], seed=seed,
                       input_shape=(28, 28, 1), dataset="mnist", variant=variant)
                acc = lr.train(participants, root_data=root, verbose=False)[-1].global_accuracy
                df = pd.DataFrame(lr.rows).assign(alpha=alpha, attack_type=attack, variant=variant, seed=seed,
                                                  final_accuracy=acc)
                df.to_csv(path + ".tmp", index=False)
                os.replace(path + ".tmp", path)
                print(f"{variant} {attack} a={alpha} seed={seed} ({time.time() - t0:.0f}s)", flush=True)


def _ci95(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, m - h, m + h


def analisar():
    lines = ["A0 — cheap analyses", ""]

    # (a) mass on attackers
    files = sorted(glob.glob(f"{OUT}/massa/*.csv"))
    if files:
        m = pd.concat([pd.read_csv(f) for f in files])
        b26 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{B26_RAW}/*_seed*.csv")])
        fin = m.groupby(["variant", "attack_type", "alpha", "seed"])["final_accuracy"].first().reset_index()
        chk = fin.merge(b26, on=["variant", "attack_type", "alpha", "seed"], how="left", suffixes=("", "_b26"))
        maxdiff = float((chk["final_accuracy"] - chk["accuracy"]).abs().max())
        lines += [f"(a) Check: {len(chk)} runs; max |acc A0 − acc B2.6| = {maxdiff:.2e} "
                  f"({'REPRODUCES' if maxdiff < 1e-9 else 'DOES NOT REPRODUCE'} B2.6)", ""]
        mass = (m[m.is_byz].groupby(["variant", "attack_type", "alpha", "seed", "round"])["weight"].sum()
                .groupby(["variant", "attack_type", "alpha", "seed"]).mean().reset_index(name="massa"))
        tab = mass.groupby(["variant", "alpha", "attack_type"])["massa"].agg(["mean", "min", "max"]).round(4)
        lines += ["(a) Mean mass on attackers per round (uniform FedAvg would give 0.2):", tab.to_string(), ""]
        for v in VARIANTS_MASSA:
            per_cell = mass[mass.variant == v].groupby(["alpha", "attack_type"])["massa"].mean()
            n_ok = int((per_cell >= MASS_THRESHOLD).sum())
            lines.append(f"    {v}: cells with mass ≥ {MASS_THRESHOLD}: {n_ok}/{len(per_cell)} -> "
                         + ("REAL EXPLOITATION (the attacker-exploitation direction holds)" if n_ok >= len(per_cell) / 2
                            else "no relevant exploitation (the attacker-exploitation direction is not explained by exploitation)"))
        lines.append("")

    # (b) signal-selection ceiling (B2.6 data, seeds 52-61, 19 valid cells)
    b26 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{B26_RAW}/*_seed*.csv")])
    b26 = b26[[(a, t) not in EXCLUDED for a, t in zip(b26.alpha, b26.attack_type)]]
    w = b26.pivot_table(index=["alpha", "attack_type", "seed"], columns="variant", values="accuracy")
    sr, cs = w["sr_only"], w["cosserver_only"]
    by_seed_max = np.maximum(sr, cs).groupby("seed").mean()                         # per-seed oracle (biased)
    cell_mean = w[["sr_only", "cosserver_only"]].groupby(["alpha", "attack_type"]).mean()
    pick = cell_mean.idxmax(axis=1)                                                  # per-cell choice, all seeds
    chosen = pd.Series([w.loc[i, pick[i[:2]]] for i in w.index], index=w.index).groupby("seed").mean()
    loso = []                                                                        # per-cell choice, leaving the seed out
    for s in sorted(w.index.get_level_values("seed").unique()):
        tr = w[w.index.get_level_values("seed") != s]
        pk = tr[["sr_only", "cosserver_only"]].groupby(["alpha", "attack_type"]).mean().idxmax(axis=1)
        te = w[w.index.get_level_values("seed") == s]
        loso.append(np.mean([te.loc[i, pk[i[:2]]] for i in te.index]))
    ref = {k: v.groupby("seed").mean() for k, v in [("sr_only", sr), ("cosserver_only", cs),
                                                    ("sr_cosserver", w["sr_cosserver"]), ("sr_bin", w["sr_bin"]),
                                                    ("sr_b025", w["sr_b025"])]}
    lines.append("(b) Signal-selection ceiling (B2.6, 19 cells, mean per seed; %):")
    for name, ser in [("max per seed (biased)", by_seed_max), ("per-cell choice (in-sample)", chosen),
                      ("per-cell choice (leave the seed out)", pd.Series(loso))] + list(ref.items()):
        mm, lo, hi = _ci95(100 * np.asarray(ser))
        lines.append(f"    {name:<46} {mm:6.2f}  CI95=({lo:.2f}, {hi:.2f})")
    best_single = np.maximum(ref["sr_only"], ref["cosserver_only"])
    for name, ser in [("max per seed", by_seed_max), ("per-cell choice (LOSO)", pd.Series(loso, index=best_single.index))]:
        d = 100 * (np.asarray(ser) - np.asarray(best_single))
        mm, lo, hi = _ci95(d)
        lines.append(f"    gain of {name} over the best single signal per seed: {mm:+.2f} p.p. CI95=({lo:+.2f}, {hi:+.2f})")
    lines.append("    Signal chosen per cell (in-sample): " + "; ".join(f"{t} α={a}: {pick[(a, t)]}" for a, t in pick.index))
    lines.append("")

    # (c) per-client agreement S_R x cos_server (ablation log: td3 trajectory, seed 42)
    rows = []
    for f in sorted(glob.glob(f"{DETECT}/td3_*_seed42.csv")):
        base = os.path.basename(f)[4:-11]
        attack, alpha = base.rsplit("_a", 1)
        d = pd.read_csv(f).dropna(subset=["S_R", "cos_server"])
        rho = d.groupby("round").apply(lambda g: stats.spearmanr(g["S_R"], g["cos_server"]).correlation
                                       if g["S_R"].nunique() > 1 and g["cos_server"].nunique() > 1 else np.nan)
        def auc(col):
            y, s = d["is_byz"].astype(int), d[col]
            if y.nunique() < 2:
                return np.nan
            r = stats.rankdata(-s)  # high score = honest -> an attacker should have a low score
            pos = r[y == 1]
            return float((pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * (len(y) - len(pos))))
        rows.append({"alpha": float(alpha), "attack_type": attack, "spearman_medio": float(np.nanmean(rho)),
                     "AUC_S_R": auc("S_R"), "AUC_cos_server": auc("cos_server")})
    if rows:
        c = pd.DataFrame(rows).sort_values(["alpha", "attack_type"])
        lines += ["(c) Per-client agreement between S_R and cos_server (ablation log: td3 trajectory, seed 42 only; "
                  "PRELIMINARY) and AUC of each signal at separating attackers (1 = perfect separation):",
                  c.round(3).to_string(index=False), ""]
        c.to_csv(f"{OUT}/concordancia_sinais.csv", index=False)

    os.makedirs(OUT, exist_ok=True)
    text = "\n".join(lines)
    open(f"{OUT}/analise.txt", "w").write(text + "\n")
    print(text)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("massa")
    r.add_argument("--seed", type=int, required=True, choices=SEEDS_MASSA)
    sub.add_parser("analisar")
    a = p.parse_args()
    massa(a.seed) if a.cmd == "massa" else analisar()


if __name__ == "__main__":
    main()
