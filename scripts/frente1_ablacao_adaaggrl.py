"""
Front 1 — AdaAggRL ablation (pre-registered in `results/frente1_ablacao_adaaggrl/PREREGISTRO.md`).
Changes nothing in `src/`.

`AdaAggRLAblationLearner` replicates `AdaAggRLGridLearner._run_round`
(src/experiments/exp10_selector_comparison.py) line by line, with `mode`:

  td3          identical to the reference AdaAggRL (TD3 chooses A=(a,b)).
  fixed        same detector (inversion + 4 MMD cues, random extractor), but
               FIXED action a=[.5,.5,.5,.5], b=.5 and no TD3 (neither select nor train).
  sr_only      only S_R from gradient inversion: a=[1,0,0,0], b=.5.
               Skips the extractor/MMD.
  cosmed_only  no inversion: the per-client score is the cosine of the update to the
               coordinate-wise median of the round's updates;
               a=[1,0,0,0], b=.5.
  cosserver_only  EXPLORATORY (chosen after seeing the step-1 AUCs): score =
               cosine of the update to the server update on the root (FLTrust
               style), computed with save/restore of np.random; a=[1,0,0,0], b=.5.

In the 4 modes the rest of the mechanism is identical: min-max of w_hat, threshold
delta=max(w_tilde)*b, counter h and penalty lam**h (lam=2), weighted
aggregation of the full parameters theta_k = global + update_k.

The `RandomCNNFeatureExtractor` is ALWAYS built (even when not
used), because its __init__ calls keras.utils.set_random_seed(seed), which
re-seeds the global np.random — omitting it would change local training and break the
pairing by seed with the reference.

`record=True` records, per round x client: is_byz, S_R and cues (if computed)
and trivial detectors — update norm, cosine and L2 distance to the
median, cosine to the server update (root, FLTrust style; computed
with save/restore of the np.random state so as not to perturb the trajectory).

Usage:
  # step 1: AUC of the trivial detectors along the reference trajectory
  python -m scripts.frente1_ablacao_adaaggrl run --mode td3 --record --seeds 42 --attack_types sign_flipping
  python -m scripts.frente1_ablacao_adaaggrl auc
  # step 3: grid
  python -m scripts.frente1_ablacao_adaaggrl run --mode fixed --seeds 42
  python -m scripts.frente1_ablacao_adaaggrl analyze
"""

import argparse
import glob
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

OUT = "results/frente1_ablacao_adaaggrl"
INPUT_SHAPE = (28, 28, 1)
MODES = ["td3", "fixed", "sr_only", "cosmed_only", "cosserver_only"]
FIXED_B = 0.5


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_learner_cls():
    from src.defense.adaaggrl_agent import (
        compute_weights_and_penalty, cue_similarity, mmd_rbf, reconstruct_client_distribution,
    )
    from src.defense.tars_selector import cross_entropy_loss
    from src.experiments.exp10_selector_comparison import AdaAggRLGridLearner
    from src.fl.attacked_learner import compute_param_updates_auto
    from src.fl.federated_learner import RoundResult

    class AdaAggRLAblationLearner(AdaAggRLGridLearner):
        def __init__(self, *args, mode: str = "td3", record: bool = False, **kwargs):
            kwargs["feature_extractor"] = "random"  # always built — see the module docstring
            super().__init__(*args, **kwargs)
            if mode not in MODES:
                raise ValueError(mode)
            self.mode = mode
            self.record = record
            self.rows: List[Dict] = []

        def _run_round(self, round_num, participants, root_data):
            active = self._is_active(round_num)
            param_updates, is_byz = compute_param_updates_auto(self, participants, active, root_data)
            n_feat = participants[0].n_features
            K = 1 if self.n_classes == 2 else self.n_classes

            old_model = self._make_model(n_feat)
            old_model.set_params(self._global_params)
            eval_data = root_data or {"X": participants[0].X_test, "y": participants[0].y_test}
            old_loss = cross_entropy_loss(old_model, eval_data["X"], eval_data["y"])

            theta_list = [self._global_params + u for u in param_updates]
            n = len(participants)
            U = np.stack(param_updates)
            med = np.median(U, axis=0)
            cos_med = np.array([_cos(u, med) for u in param_updates])

            s_r = np.full(n, np.nan)
            cues = np.full((n, 3), np.nan)
            if self.mode in ("td3", "fixed", "sr_only"):
                v_current = {}
                for i, (p, theta_k, u) in enumerate(zip(participants, theta_list, param_updates)):
                    W = theta_k[: n_feat * K].reshape(n_feat, K).astype(np.float32)
                    b = theta_k[n_feat * K:].astype(np.float32)
                    D_rec, s_r[i] = reconstruct_client_distribution(
                        W, b, u, self.local_lr, n_feat, K,
                        num_images=self.num_images, max_iters=self.max_iters, seed=round_num,
                    )
                    if self.mode != "sr_only":
                        v_current[p.id] = self.feature_extractor.extract(D_rec)
                if self.mode != "sr_only":
                    v_g = np.concatenate(list(v_current.values()), axis=0)
                    for i, p in enumerate(participants):
                        v_cur = v_current[p.id]
                        v_hist = self._v_history.get(p.id, v_cur)
                        cues[i] = [cue_similarity(mmd_rbf(v_cur, v_hist)),
                                   cue_similarity(mmd_rbf(v_cur, v_g)),
                                   cue_similarity(mmd_rbf(v_hist, v_g))]
                    for p in participants:
                        self._v_history[p.id] = v_current[p.id]

            cos_srv = None
            if self.mode == "cosserver_only":
                rng_state = np.random.get_state()
                root = root_data or self._carve_root(participants[0])
                srv_update = self._make_model(n_feat).fit(root["X"], root["y"])
                np.random.set_state(rng_state)
                cos_srv = np.array([_cos(u, srv_update) for u in param_updates])

            if self._h is None:
                self._h = np.zeros(n)

            if self.mode in ("td3", "fixed"):
                state_matrix = np.column_stack([s_r, cues])
            elif self.mode == "sr_only":
                state_matrix = np.column_stack([s_r, np.zeros((n, 3))])
            elif self.mode == "cosmed_only":
                state_matrix = np.column_stack([cos_med, np.zeros((n, 3))])
            else:
                state_matrix = np.column_stack([cos_srv, np.zeros((n, 3))])
            mean_state = state_matrix.mean(axis=0)

            if self.mode == "td3":
                action = self.agent.select_action(mean_state)
            elif self.mode == "fixed":
                action = np.array([0.5, 0.5, 0.5, 0.5, FIXED_B])
            else:
                action = np.array([1.0, 0.0, 0.0, 0.0, FIXED_B])

            weights, new_h, _ = compute_weights_and_penalty(state_matrix, action, self._h, lam=self.lam)
            self._h = new_h
            tw = float(weights.sum())
            nw = np.ones(n) / n if tw < 1e-8 else weights / tw
            self._global_params = sum(w * th for w, th in zip(nw, theta_list))

            new_model = self._make_model(n_feat)
            new_model.set_params(self._global_params)
            reward = old_loss - cross_entropy_loss(new_model, eval_data["X"], eval_data["y"])
            if self.mode == "td3":
                self.agent.train_step(mean_state, action, reward, mean_state)

            per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
            global_acc = float(np.mean(list(per_acc.values())))

            if self.record:
                rng_state = np.random.get_state()
                root = root_data or self._carve_root(participants[0])
                srv_model = self._instantiate_model(n_feat)
                srv_model.set_params(self._global_params - sum(w * u for w, u in zip(nw, param_updates)))
                server_update = srv_model.fit(root["X"], root["y"])
                np.random.set_state(rng_state)
                for i, p in enumerate(participants):
                    self.rows.append({
                        "round": round_num, "client": p.id, "is_byz": bool(is_byz[i]),
                        "S_R": s_r[i], "S_cl": cues[i, 0], "S_cg": cues[i, 1], "S_lg": cues[i, 2],
                        "norm": float(np.linalg.norm(param_updates[i])),
                        "cos_median": float(cos_med[i]),
                        "l2_median": float(np.linalg.norm(param_updates[i] - med)),
                        "cos_server": _cos(param_updates[i], server_update),
                        "final_weight": float(nw[i]),
                    })

            return RoundResult(round_num, global_acc, per_acc, trust_scores=None,
                               n_accepted=int((weights > 0).sum()))

    return AdaAggRLAblationLearner


def run(mode: str, seeds: List[int], alphas: List[float], attack_types: Optional[List[str]],
        record: bool, n_rounds: int = 15, tag: str = "") -> None:
    from src.experiments.exp9_dominance_grid import BLIND_ATTACKS, INFORMED_ATTACKS, _split_name
    from src.utils.data_loader import load_dataset_participants

    Learner = build_learner_cls()
    attack_types = attack_types or (INFORMED_ATTACKS + BLIND_ATTACKS)
    os.makedirs(f"{OUT}/raw", exist_ok=True)
    os.makedirs(f"{OUT}/detectores", exist_ok=True)
    for seed in seeds:
        rows = []
        for alpha in alphas:
            participants, root = load_dataset_participants(
                "mnist", _split_name(alpha), 10, root_size=100, root_seed=seed,
            )
            for attack in attack_types:
                t0 = time.time()
                learner = Learner(
                    n_rounds=n_rounds, n_classes=10, attack_type=attack,
                    byzantine_ids=[0, 1], seed=seed, input_shape=INPUT_SHAPE,
                    dataset="mnist", mode=mode, record=record,
                )
                acc = learner.train(participants, root_data=root, verbose=False)[-1].global_accuracy
                rows.append({"alpha": alpha, "attack_type": attack, "mode": mode,
                             "accuracy": acc, "seed": seed})
                print(f"mode={mode} seed={seed} alpha={alpha} attack={attack} acc={acc:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                if record:
                    pd.DataFrame(learner.rows).to_csv(
                        f"{OUT}/detectores/{mode}_{attack}_a{alpha}_seed{seed}.csv", index=False)
        suffix = f"_{tag}" if tag else ""
        pd.DataFrame(rows).to_csv(f"{OUT}/raw/{mode}_seed{seed}{suffix}.csv", index=False)


# ---------------------------------------------------------------------------
# Step 1 — detector AUC
# ---------------------------------------------------------------------------

# pre-declared direction: +1 = Byzantine has a HIGHER value; -1 = LOWER
DETECTORS = {"S_R": -1, "S_cl": -1, "S_cg": -1, "S_lg": -1,
             "norm": +1, "cos_median": -1, "l2_median": +1, "cos_server": -1}


def auc() -> None:
    from sklearn.metrics import roc_auc_score

    rows = []
    for f in sorted(glob.glob(f"{OUT}/detectores/td3_*.csv")):
        name = os.path.basename(f)[4:-4]
        attack, rest = name.rsplit("_a", 1)
        alpha = float(rest.split("_seed")[0])
        d = pd.read_csv(f)
        d = d[d["round"] >= 2]
        y = d["is_byz"].astype(int).values
        for det, sign in DETECTORS.items():
            x = d[det].values * sign
            ok = 0 < y.sum() < len(y) and np.nanstd(x) > 0
            rows.append({"attack_type": attack, "alpha": alpha, "detector": det,
                         "auc": roc_auc_score(y, x) if ok else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/passo1_auc_detectores.csv", index=False)
    piv = df.pivot_table(index=["attack_type", "alpha"], columns="detector", values="auc")
    piv = piv[list(DETECTORS)]
    pd.set_option("display.width", 200)
    print("=== AUC per cell (pre-declared direction; rounds >= 2; seed 42) ===")
    print(piv.round(2).to_string())
    print("\n=== mean over the cells ===")
    print(piv.mean().round(3).to_string())
    print("\n=== number of cells with AUC >= 0.70 ===")
    print((piv >= 0.70).sum().to_string())


# ---------------------------------------------------------------------------
# Step 3 — pre-registered analysis
# ---------------------------------------------------------------------------

MARGIN = 0.01
ALPHA_TEST = 0.05


def _load_reference() -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(glob.glob(
        "results/tables/exp10_selector_comparison_variantb_full_seed*_raw.csv"))]
    ref = pd.concat(frames)
    ref = ref[ref.system == "AdaAggRL"][["alpha", "attack_type", "seed", "accuracy"]]
    return ref.assign(mode="td3_ref")


def _load_random() -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(glob.glob(
        "results/tables/exp10_selector_comparison_variantb_full_seed*_raw.csv"))]
    r = pd.concat(frames)
    return r[r.system == "Random"][["alpha", "attack_type", "seed", "accuracy"]].assign(mode="random")


def _tost_paired(diff: np.ndarray, margin: float):
    """Paired TOST (t) on per-seed differences: equivalent if the 90% CI
    of the mean lies within ±margin. Returns (p_tost, ic90_lo, ic90_hi)."""
    from scipy import stats
    n = len(diff)
    m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return (0.0 if abs(m) < margin else 1.0), m, m
    p_lo = 1 - stats.t.cdf((m + margin) / se, n - 1)
    p_hi = stats.t.cdf((m - margin) / se, n - 1)
    t90 = stats.t.ppf(0.95, n - 1)
    return max(p_lo, p_hi), m - t90 * se, m + t90 * se


def _wilcoxon(diff: np.ndarray) -> float:
    from scipy import stats
    if np.allclose(diff, 0):
        return 1.0
    return float(stats.wilcoxon(diff).pvalue)


def _holm(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run_max = 0.0
    for k, idx in enumerate(order):
        run_max = max(run_max, (m - k) * p[idx])
        adj[idx] = min(run_max, 1.0)
    return adj.tolist()


def _cohen_d(diff: np.ndarray) -> float:
    s = diff.std(ddof=1)
    return float(diff.mean() / s) if s > 0 else float("nan")


def _headroom_count(sys_df: pd.DataFrame, fixed_raw: pd.DataFrame) -> int:
    """Same criterion as exp9/Step Zero: best fixed rule per cell (highest
    mean), headroom if (system_mean - best_fixed_mean) > system_std + fixed_std."""
    fx = fixed_raw.groupby(["alpha", "attack_type", "strategy"])["accuracy"].agg(["mean", "std"]).reset_index()
    best = fx.loc[fx.groupby(["alpha", "attack_type"])["mean"].idxmax()]
    s = sys_df.groupby(["alpha", "attack_type"])["accuracy"].agg(["mean", "std"]).reset_index()
    m = s.merge(best, on=["alpha", "attack_type"], suffixes=("_sys", "_fix"))
    return int(((m.mean_sys - m.mean_fix) > (m.std_sys + m.std_fix)).sum())


def analyze() -> None:
    raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/raw/*_seed*.csv"))
                     if "_smoke" not in f])
    raw = raw[raw["mode"] != "td3"]
    ref = _load_reference()
    rnd = _load_random()
    fixed_raw = pd.read_csv("results/tables/exp9_dominance_grid_10seeds_ALL_root100_raw.csv")
    allsys = pd.concat([ref, rnd, raw[["alpha", "attack_type", "seed", "accuracy", "mode"]]])
    allsys.to_csv(f"{OUT}/grade_combined_raw.csv", index=False)

    lines = []
    summ = allsys.groupby("mode")["accuracy"].mean()
    lines.append("=== mean accuracy (210 cells×seed) ===\n" + summ.round(4).to_string())
    hr = {m: _headroom_count(allsys[allsys["mode"] == m], fixed_raw) for m in allsys["mode"].unique()}
    lines.append("\n=== cells with headroom over the best fixed rule (out of 21) ===\n" +
                 "\n".join(f"{k}: {v}" for k, v in hr.items()))

    pooled_rows, per_attack_rows = [], []
    comparisons = [("fixed", "td3_ref"), ("sr_only", "td3_ref"), ("cosmed_only", "td3_ref"),
                   ("cosmed_only", "sr_only"), ("fixed", "random"), ("sr_only", "random"),
                   ("cosmed_only", "random"),
                   # exploratory
                   ("cosserver_only", "td3_ref"), ("cosserver_only", "sr_only"),
                   ("cosserver_only", "random")]
    wide = allsys.pivot_table(index=["alpha", "attack_type", "seed"], columns="mode", values="accuracy")
    for a, b in comparisons:
        if a not in wide or b not in wide:
            continue
        w = wide[[a, b]].dropna()
        by_seed = w.groupby("seed").mean()
        diff = (by_seed[a] - by_seed[b]).values
        p_tost, lo, hi = _tost_paired(diff, MARGIN)
        pooled_rows.append({"comparison": f"{a} - {b}", "n_seeds": len(diff), "delta": diff.mean(),
                            "ci90_lo": lo, "ci90_hi": hi, "p_tost": p_tost,
                            "equivalent_pm0.01": p_tost < ALPHA_TEST,
                            "p_wilcoxon": _wilcoxon(diff), "cohen_d": _cohen_d(diff)})
        att = []
        for attack, g in w.groupby("attack_type"):
            bs = g.groupby("seed").mean()
            d = (bs[a] - bs[b]).values
            pt, lo, hi = _tost_paired(d, MARGIN)
            att.append({"comparison": f"{a} - {b}", "attack_type": attack, "delta": d.mean(),
                        "ci90_lo": lo, "ci90_hi": hi, "p_tost": pt, "p_wilcoxon": _wilcoxon(d),
                        "cohen_d": _cohen_d(d)})
        pw = _holm([r["p_wilcoxon"] for r in att])
        pt = _holm([r["p_tost"] for r in att])
        for r, x, y in zip(att, pw, pt):
            r["p_wilcoxon_holm"] = x
            r["p_tost_holm"] = y
            r["sig_diff"] = x < ALPHA_TEST
            r["equivalent"] = y < ALPHA_TEST
        per_attack_rows += att

    pooled = pd.DataFrame(pooled_rows)
    per_attack = pd.DataFrame(per_attack_rows)
    pooled.to_csv(f"{OUT}/grade_pooled.csv", index=False)
    per_attack.to_csv(f"{OUT}/grade_por_ataque.csv", index=False)
    pd.set_option("display.width", 220)
    lines.append("\n=== pooled (mean per seed over the 21 cells; TOST ±0.01 + Wilcoxon) ===\n" +
                 pooled.round(4).to_string(index=False))
    lines.append("\n=== per attack (mean per seed over 3 alphas; Holm over 7) ===\n" +
                 per_attack.round(4).to_string(index=False))
    text = "\n".join(lines)
    with open(f"{OUT}/grade_analise.txt", "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=MODES, required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    r.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.1, 0.05])
    r.add_argument("--attack_types", nargs="+", default=None)
    r.add_argument("--n_rounds", type=int, default=15)
    r.add_argument("--record", action="store_true")
    r.add_argument("--tag", default="")
    sub.add_parser("auc")
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.mode, args.seeds, args.alphas, args.attack_types, args.record, args.n_rounds, args.tag)
    elif args.cmd == "auc":
        auc()
    else:
        analyze()
