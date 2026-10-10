"""
C0b (results/c0b_espaco_h150/PLANO.md + ADENDO1.md): remaining headroom at H = 150
against the best existing method. 19 cells, seeds 72-81, nested horizons
15/50/150.

Systems (8), one job per cell × seed, `_reseed(seed)` before each one:
  skeleton variants (the B2.6 learner): sr_only, sr_bin, sr_b025, cosserver_only
  static rules (B2.7's PlainRuleLearner): fltrust, trimmed_mean, clustering, median
Ceilings per α × seed (B2.7's run_ceiling, already with the FedAvg-8 oracle fix).

Per-client logging in the 4 variants (for the A0 agreement analysis and the signal-selection
ceiling without re-running): per round × client, is_byz, S_R (in the variants that already
compute the inversion; NaN in cosserver_only, to avoid paying for the inversion),
cos_server (ALWAYS computed, with save/restore of np.random, as in B2.6, so the
trajectory does not change), mask (client included: weight > 0) and normalized weight.

Usage (CPU, no GPU):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.c0b_espaco_h150 celula --alpha 0.05 --attack label_flipping --seed 72
  ... -m scripts.c0b_espaco_h150 teto --alpha 0.05 --seed 72
  ... -m scripts.c0b_espaco_h150 analisar
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/c0b_espaco_h150"
RAW = f"{OUT}/raw"
SEEDS = list(range(72, 82))
ALPHAS = [0.5, 0.1, 0.05]
EXCLUDED = {(0.05, "fltrust_aligned"), (0.1, "fltrust_aligned")}
VARIANTS = ["sr_only", "sr_bin", "sr_b025", "cosserver_only"]
RULES = ["fltrust", "trimmed_mean", "clustering", "median"]
SYSTEMS = VARIANTS + RULES
HORIZONS = [15, 50, 150]
H_MAX = 150
THRESH_PP = 2.0


def cells():
    from src.experiments.exp9_dominance_grid import BLIND_ATTACKS, INFORMED_ATTACKS
    return [(a, t) for a in ALPHAS for t in INFORMED_ATTACKS + BLIND_ATTACKS if (a, t) not in EXCLUDED]


def build_logging_learner():
    """B2.6 DecompLearner with _run_round copied and only logging added."""
    from scripts.b26_decomposicao import NEEDS_SR, _cos, build_learner_cls, weights_for
    from src.defense.adaaggrl_agent import reconstruct_client_distribution
    from src.fl.attacked_learner import compute_param_updates_auto
    from src.fl.federated_learner import RoundResult

    Decomp = build_learner_cls()

    class LoggingDecompLearner(Decomp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.log = []

        def _run_round(self, round_num, participants, root_data):
            active = self._is_active(round_num)
            param_updates, is_byz = compute_param_updates_auto(self, participants, active, root_data)
            n_feat = participants[0].n_features
            K = 1 if self.n_classes == 2 else self.n_classes
            theta_list = [self._global_params + u for u in param_updates]
            n = len(participants)

            s_r = np.full(n, np.nan)
            if self.variant in NEEDS_SR:
                for i, (theta_k, u) in enumerate(zip(theta_list, param_updates)):
                    W = theta_k[: n_feat * K].reshape(n_feat, K).astype(np.float32)
                    b = theta_k[n_feat * K:].astype(np.float32)
                    _, s_r[i] = reconstruct_client_distribution(
                        W, b, u, self.local_lr, n_feat, K,
                        num_images=self.num_images, max_iters=self.max_iters, seed=round_num,
                    )
            # cos_server always (logging); save/restore of np.random as in B2.6
            rng_state = np.random.get_state()
            root = root_data or self._carve_root(participants[0])
            srv_update = self._make_model(n_feat).fit(root["X"], root["y"])
            np.random.set_state(rng_state)
            cos_srv = np.array([_cos(u, srv_update) for u in param_updates])

            if self.variant == "cosserver_only":
                score = cos_srv
            elif self.variant == "sr_cosserver":
                score = 0.5 * s_r + 0.5 * cos_srv
            else:
                score = s_r

            if self._h is None:
                self._h = np.zeros(n)
            w, self._h = weights_for(self.variant, score, self._h)
            tw = float(w.sum())
            nw = np.ones(n) / n if tw < 1e-8 else w / tw
            self._global_params = sum(wi * th for wi, th in zip(nw, theta_list))

            for i, p in enumerate(participants):
                self.log.append({"round": round_num, "client": p.id, "is_byz": bool(is_byz[i]),
                                 "S_R": float(s_r[i]), "cos_server": float(cos_srv[i]),
                                 "incluido": bool(w[i] > 0), "peso": float(nw[i])})

            per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
            return RoundResult(round_num, float(np.mean(list(per_acc.values()))), per_acc,
                               trust_scores=None, n_accepted=int((w > 0).sum()))

    return LoggingDecompLearner


def run_cell(alpha, attack, seed, n_rounds=H_MAX, systems=None, out_dir=RAW, log=True):
    from scripts.b27_horizonte import BYZ, INPUT_SHAPE, _reseed, build_plain_rule_learner
    from scripts.b26_decomposicao import build_learner_cls
    from src.experiments.exp9_dominance_grid import _split_name
    from src.utils.data_loader import load_dataset_participants

    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/celula_{attack}_a{alpha}_seed{seed}_R{n_rounds}.csv"
    if os.path.exists(path):
        print(f"already exists: {path}", flush=True)
        return path
    Variant = build_logging_learner() if log else build_learner_cls()
    PlainRule, _ = build_plain_rule_learner()
    participants, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    common = dict(n_rounds=n_rounds, n_classes=10, attack_type=attack, byzantine_ids=BYZ, seed=seed)
    rows, logs = [], []
    for system in systems or SYSTEMS:
        t0 = time.time()
        _reseed(seed)
        if system in VARIANTS:
            lr = Variant(input_shape=INPUT_SHAPE, dataset="mnist", variant=system, **common)
        else:
            lr = PlainRule(rule=system, **common)
        res = lr.train(participants, root_data=root, verbose=False)
        dt = time.time() - t0
        for H in HORIZONS:
            if len(res) >= H:
                rows.append({"alpha": alpha, "attack_type": attack, "seed": seed, "system": system, "H": H,
                             "accuracy": float(res[H - 1].global_accuracy), "sec": dt})
        if log and system in VARIANTS:
            logs.append(pd.DataFrame(lr.log).assign(system=system))
        print(f"alpha={alpha} attack={attack} seed={seed} {system} ({dt:.0f}s)", flush=True)
    if logs:
        lp = f"{out_dir}/scores_{attack}_a{alpha}_seed{seed}_R{n_rounds}.csv.gz"
        pd.concat(logs).to_csv(lp + ".tmp.gz", index=False)
        os.replace(lp + ".tmp.gz", lp)
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    print(f"END {path}", flush=True)
    return path


def run_teto(alpha, seed):
    from scripts.b27_horizonte import run_ceiling
    return run_ceiling(alpha, seed, n_rounds=H_MAX, out_dir=RAW)


# ---------------------------------------------------------------------------
# Pre-written analysis
# ---------------------------------------------------------------------------

def _ci95(x):
    x = np.asarray(x, float)
    n, m = len(x), x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def _auc(y, s):
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return np.nan
    r = stats.rankdata(-np.asarray(s))  # high score = honest
    pos = r[y == 1]
    return float((pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * (len(y) - len(pos))))


def analisar():
    raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RAW}/celula_*_R{H_MAX}.csv"))])
    tet = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RAW}/teto_*_R{H_MAX}.csv"))])
    raw.to_csv(f"{OUT}/grade_raw.csv", index=False)
    tet.to_csv(f"{OUT}/tetos_raw.csv", index=False)
    pd.set_option("display.width", 250)
    lines = ["C0b — remaining headroom at H = 150 (descriptive; C0 criterion); seeds 72–81; 19 cells",
             f"rows: {len(raw)} (expected {19 * 10 * 8 * 3}); ceilings: {len(tet)} (expected {3 * 10 * 2 * 3})", ""]

    # (i) remaining-headroom map
    rows = []
    for (alpha, attack), g in raw.groupby(["alpha", "attack_type"]):
        for H, gh in g.groupby("H"):
            w = gh.pivot_table(index="seed", columns="system", values="accuracy")
            best = w.mean().idxmax()
            best_var = w[[v for v in VARIANTS if v in w]].mean().idxmax()
            best_rule = w[[r for r in RULES if r in w]].mean().idxmax()
            t = tet[(tet.alpha == alpha) & (tet.H == H)].pivot_table(index="seed", columns="system", values="accuracy")
            row = {"alpha": alpha, "attack_type": attack, "H": H, "melhor_existente": best,
                   "acc_melhor": w[best].mean(), "melhor_variante": best_var, "melhor_regra": best_rule}
            for tname, lab in [("teto_oraculo_fedavg8", "oraculo8"), ("teto_fedavg10", "fedavg10")]:
                for ref, rl in [(best, "global"), ("sr_only", "esqueleto_ref")]:
                    d = (t[tname] - w[ref]).dropna()
                    m, lo, hi = _ci95(d)
                    row.update({f"gap_{rl}_{lab}_pp": 100 * m, f"gap_{rl}_{lab}_lo": 100 * lo,
                                f"gap_{rl}_{lab}_hi": 100 * hi})
            row["espaco_global"] = bool(row["gap_global_oraculo8_pp"] > THRESH_PP and row["gap_global_oraculo8_lo"] > 0)
            row["espaco_vs_esqueleto_ref"] = bool(row["gap_esqueleto_ref_oraculo8_pp"] > THRESH_PP
                                                  and row["gap_esqueleto_ref_oraculo8_lo"] > 0)
            d = (w[best_var] - w[best_rule]).dropna()
            m, lo, hi = _ci95(d)
            row.update({"var_menos_regra_pp": 100 * m, "var_menos_regra_lo": 100 * lo, "var_menos_regra_hi": 100 * hi})
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/espaco_restante.csv", index=False)
    for H in HORIZONS:
        r = res[res.H == H]
        lines.append(f"H={H:3d}: cells with headroom vs. the BEST EXISTING (FedAvg-8 oracle, gap > {THRESH_PP} p.p., "
                     f"CI95 > 0) = {int(r.espaco_global.sum())}/{len(r)}; vs. the reference skeleton (sr_only) = "
                     f"{int(r.espaco_vs_esqueleto_ref.sum())}/{len(r)}")
    lines += ["", "(i) Map at H = 150:",
              res[res.H == H_MAX][["alpha", "attack_type", "melhor_existente", "acc_melhor", "gap_global_oraculo8_pp",
                                   "gap_global_oraculo8_lo", "gap_global_oraculo8_hi", "espaco_global",
                                   "gap_global_fedavg10_pp", "gap_esqueleto_ref_oraculo8_pp", "espaco_vs_esqueleto_ref"]]
              .round(3).to_string(index=False), "",
              "(iii) Best skeleton variant − best static rule (p.p., CI95), H = 150:",
              res[res.H == H_MAX][["alpha", "attack_type", "melhor_variante", "melhor_regra", "var_menos_regra_pp",
                                   "var_menos_regra_lo", "var_menos_regra_hi"]].round(3).to_string(index=False), ""]

    # (ii) signal-selection ceiling at H = 150 (same logic as A0 b)
    w = raw[raw.H == H_MAX].pivot_table(index=["alpha", "attack_type", "seed"], columns="system", values="accuracy")
    sr, cs = w["sr_only"], w["cosserver_only"]
    by_seed_max = np.maximum(sr, cs).groupby("seed").mean()
    pick = w[["sr_only", "cosserver_only"]].groupby(["alpha", "attack_type"]).mean().idxmax(axis=1)
    chosen = pd.Series([w.loc[i, pick[i[:2]]] for i in w.index], index=w.index).groupby("seed").mean()
    loso = []
    for s in SEEDS:
        tr = w[w.index.get_level_values("seed") != s]
        pk = tr[["sr_only", "cosserver_only"]].groupby(["alpha", "attack_type"]).mean().idxmax(axis=1)
        te = w[w.index.get_level_values("seed") == s]
        loso.append(np.mean([te.loc[i, pk[i[:2]]] for i in te.index]))
    loso = pd.Series(loso, index=SEEDS)
    lines.append("(ii) Signal-selection ceiling at H = 150 (19 cells, mean per seed; %):")
    for name, ser in ([("max per seed (biased)", by_seed_max), ("per-cell choice (in-sample)", chosen),
                       ("per-cell choice (LOSO, reference)", loso)]
                      + [(sname, w[sname].groupby("seed").mean()) for sname in SYSTEMS if sname in w]):
        m, lo, hi = _ci95(100 * ser.to_numpy())
        lines.append(f"    {name:<42} {m:6.2f}  CI95=({lo:.2f}, {hi:.2f})")
    best_single = np.maximum(sr.groupby("seed").mean(), cs.groupby("seed").mean())
    for name, ser in [("max per seed", by_seed_max), ("per-cell choice (LOSO)", loso)]:
        m, lo, hi = _ci95(100 * (ser.to_numpy() - best_single.to_numpy()))
        lines.append(f"    gain of {name} over the best single signal per seed: {m:+.2f} p.p. CI95=({lo:+.2f}, {hi:+.2f})")
    lines.append("    Signal chosen per cell (in-sample): " + "; ".join(f"{t} α={a}: {pick[(a, t)]}" for a, t in pick.index))
    lines.append("")

    # S_R × cos_server agreement and AUC per signal (sr_only log, 10 seeds) — descriptive
    rows = []
    for f in sorted(glob.glob(f"{RAW}/scores_*_R{H_MAX}.csv.gz")):
        base = os.path.basename(f)[len("scores_"):-len(f"_R{H_MAX}.csv.gz")]
        attack, rest = base.rsplit("_a", 1)
        alpha, seed = rest.split("_seed")
        d = pd.read_csv(f)
        d = d[(d.system == "sr_only") & (d["round"] >= 2)].dropna(subset=["S_R", "cos_server"])
        rho = d.groupby("round").apply(lambda g: stats.spearmanr(g.S_R, g.cos_server).correlation
                                       if g.S_R.nunique() > 1 and g.cos_server.nunique() > 1 else np.nan)
        rows.append({"alpha": float(alpha), "attack_type": attack, "seed": int(seed),
                     "spearman_medio": float(np.nanmean(rho)), "AUC_S_R": _auc(d.is_byz, d.S_R),
                     "AUC_cos_server": _auc(d.is_byz, d.cos_server)})
    if rows:
        c = pd.DataFrame(rows)
        c.to_csv(f"{OUT}/concordancia_sinais.csv", index=False)
        lines += ["Per-client agreement (sr_only trajectory, rounds ≥ 2, mean of 10 seeds; descriptive):",
                  c.groupby(["alpha", "attack_type"])[["spearman_medio", "AUC_S_R", "AUC_cos_server"]].mean()
                  .round(3).to_string(), ""]
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/analise.txt", "w").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("celula")
    c.add_argument("--alpha", type=float, required=True)
    c.add_argument("--attack", required=True)
    c.add_argument("--seed", type=int, required=True)
    t = sub.add_parser("teto")
    t.add_argument("--alpha", type=float, required=True)
    t.add_argument("--seed", type=int, required=True)
    sub.add_parser("analisar")
    a = ap.parse_args()
    if a.cmd == "celula":
        run_cell(a.alpha, a.attack, a.seed)
    elif a.cmd == "teto":
        run_teto(a.alpha, a.seed)
    else:
        analisar()


if __name__ == "__main__":
    main()
