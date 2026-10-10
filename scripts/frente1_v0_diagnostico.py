"""
Front 1, V0: sanity diagnostic of the AdaAggRL detector (gradient inversion + 4 MMD cues)
and of the pretrained extractor. Does NOT change any code in `src/`; it only observes.

Part A — static checks:
  A1. pretrained extractor weights: the file exists, loads, differs from a fresh
      initialization, and the full classifier is accurate on data it
      never saw (train pool); the feature sub-model reproduces the classifier's
      intermediate layer.
  A2. inference mode: layer types (Dropout/BatchNorm would differ between
      training and inference).
  A3. input range of the real data (the extractor was trained on [0,1]).
  A4. overlap between `data/raw/mnist/X_test.npy` (where the extractor was
      pretrained) and the clients' `X_test` (where the FL accuracy is measured).

Part B — real AdaAggRL trajectory (`random` extractor, the reference
floor), and every round, for every client, recomputes the detector in
variants that isolate each suspicion:
  recon ∈ {orig: target=+update/lr, W=theta_k (as in the code),
           flip: target=−update/lr, W=theta_k,
           flip_global: target=−update/lr, W=global (the point where the gradient
           was actually computed)}
  extractor ∈ {random, pretrained};  prep ∈ {raw, clip01}
  and for each combination: MMD² (median per-pair bandwidth, as in the
  code), S by Eq. 1 on MMD² (as in the code), S by Eq. 1 on
  sqrt(MMD²), and S with a fixed bandwidth per round.
  Also records, per client: cosine between the update and the true −∇ (on the client's
  clean data, at the global model), range statistics of D_rec, and the
  MMD² between features of D_rec and of the client's real images (fidelity).

Part C — AUC of each cue (benign × Byzantine) per variant and cell, and
distribution figures.

Usage:
    python -m scripts.frente1_v0_diagnostico --cells sign_flipping:0.5 \
        gaussian_noise:0.1 label_flipping:0.5 --seed 42
    python -m scripts.frente1_v0_diagnostico --analyze_only
"""

import argparse
import glob
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

OUT_TABLES = "results/tables"
OUT_FIGS = "results/figures"
WEIGHTS = "results/models/adaaggrl_pretrained_extractor_mnist.weights.h5"
INPUT_SHAPE = (28, 28, 1)


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------

def static_checks() -> Dict:
    from tensorflow import keras

    from scripts.pretrain_adaaggrl_extractor import _build_classifier
    from src.defense.adaaggrl_agent import PretrainedCNNFeatureExtractor
    from src.utils.data_loader import load_dataset_participants

    out: Dict = {}
    out["A1_weights_exist"] = os.path.exists(WEIGHTS)

    keras.utils.set_random_seed(123)
    fresh = _build_classifier(INPUT_SHAPE, 10, 32)
    loaded = _build_classifier(INPUT_SHAPE, 10, 32)
    loaded.load_weights(WEIGHTS)
    diffs = [float(np.abs(a - b).max()) for a, b in zip(fresh.get_weights(), loaded.get_weights())]
    out["A1_max_abs_diff_vs_fresh_init_per_tensor"] = diffs

    participants, root = load_dataset_participants("mnist", "non_iid", 10)
    X = np.concatenate([p.X_train[:300] for p in participants])
    y = np.concatenate([p.y_train[:300] for p in participants])
    Xi = X.reshape((-1,) + INPUT_SHAPE).astype("float32")
    out["A1_classifier_acc_on_train_pool_sample"] = float(
        np.mean(loaded.predict(Xi, verbose=0).argmax(1) == y)
    )

    ext = PretrainedCNNFeatureExtractor(INPUT_SHAPE, weights_path=WEIGHTS)
    f_ext = ext.extract(X[:200].reshape(200, -1))
    f_ref = keras.Model(loaded.inputs, loaded.get_layer("features").output).predict(Xi[:200], verbose=0)
    out["A1_feature_submodel_matches_classifier"] = bool(np.allclose(f_ext, f_ref, atol=1e-5))
    out["A1_feature_stats_real_images"] = {
        "mean": float(f_ext.mean()), "frac_zero_relu": float((f_ext == 0).mean()),
    }

    out["A2_layer_types"] = [type(l).__name__ for l in ext._model.layers]
    out["A2_has_dropout_or_bn"] = any(
        n in ("Dropout", "BatchNormalization") for n in out["A2_layer_types"]
    )

    out["A3_client_X_train_range"] = [float(X.min()), float(X.max())]
    out["A3_client_X_shape_flat"] = int(participants[0].n_features)

    raw_test = np.load("data/raw/mnist/X_test.npy").reshape(10000, -1)
    raw_hash = {hash(r.tobytes()) for r in raw_test.astype(np.float32)}
    n_tot = n_hit = 0
    for p in participants:
        for r in p.X_test.reshape(len(p.X_test), -1).astype(np.float32):
            n_tot += 1
            n_hit += hash(r.tobytes()) in raw_hash
    out["A4_client_test_rows_found_in_raw_X_test"] = f"{n_hit}/{n_tot}"
    return out


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------

def _mmd2(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    def k(A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-d2 / (2.0 * sigma ** 2))
    return float(k(X, X).mean() + k(Y, Y).mean() - 2.0 * k(X, Y).mean())


def _true_grad(W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    z = X @ W + b
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    p[np.arange(len(y)), y] -= 1.0
    gW = X.T @ p / len(y)
    gb = p.mean(axis=0)
    return np.concatenate([gW.reshape(-1), gb])


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_diag_learner_cls():
    from src.defense.adaaggrl_agent import (
        PretrainedCNNFeatureExtractor, RandomCNNFeatureExtractor,
        compute_weights_and_penalty, cue_similarity, mmd_rbf,
        reconstruct_client_distribution,
    )
    from src.defense.tars_selector import cross_entropy_loss
    from src.experiments.exp10_selector_comparison import AdaAggRLGridLearner
    from src.fl.attacked_learner import compute_param_updates_auto
    from src.fl.federated_learner import RoundResult

    RECONS = ["orig", "flip", "flip_global"]
    EXTRACTORS = ["random", "pretrained"]
    PREPS = ["raw", "clip01"]

    class DiagAdaAggRL(AdaAggRLGridLearner):
        """Follows exactly the `AdaAggRLGridLearner` trajectory (recon orig,
        constructor extractor, raw input) and records side diagnostics
        that do not influence the aggregation."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.extractors = {
                "random": RandomCNNFeatureExtractor(self.input_shape, feature_dim=16, seed=kwargs["seed"]),
                "pretrained": PretrainedCNNFeatureExtractor(self.input_shape, feature_dim=32, weights_path=WEIGHTS),
            }
            self._hist: Dict[tuple, np.ndarray] = {}
            self.rows: List[Dict] = []
            self.traj_rows: List[Dict] = []

        def _run_round(self, round_num, participants, root_data):
            active = self._is_active(round_num)
            param_updates, is_byz_list = compute_param_updates_auto(self, participants, active, root_data)
            n_feat = participants[0].n_features
            K = 1 if self.n_classes == 2 else self.n_classes

            old_model = self._make_model(n_feat)
            old_model.set_params(self._global_params)
            eval_data = root_data or {"X": participants[0].X_test, "y": participants[0].y_test}
            old_loss = cross_entropy_loss(old_model, eval_data["X"], eval_data["y"])

            g = self._global_params
            Wg = g[: n_feat * K].reshape(n_feat, K)
            bg = g[n_feat * K:]
            theta_list = [g + u for u in param_updates]

            # --- side diagnostic: 3 reconstructions per client ---
            recs: Dict[str, List[np.ndarray]] = {r: [] for r in RECONS}
            srs: Dict[str, List[float]] = {r: [] for r in RECONS}
            client_info = []
            for i, (p, theta_k, u) in enumerate(zip(participants, theta_list, param_updates)):
                Wk = theta_k[: n_feat * K].reshape(n_feat, K).astype(np.float32)
                bk = theta_k[n_feat * K:].astype(np.float32)
                for r, (W, b, tgt) in {
                    "orig": (Wk, bk, u),
                    "flip": (Wk, bk, -u),
                    "flip_global": (Wg.astype(np.float32), bg.astype(np.float32), -u),
                }.items():
                    D, S_R = reconstruct_client_distribution(
                        W, b, tgt, self.local_lr, n_feat, K,
                        num_images=self.num_images, max_iters=self.max_iters, seed=round_num,
                    )
                    recs[r].append(D); srs[r].append(S_R)
                n_s = min(200, p.n_train)
                idx = np.random.RandomState(round_num * 100 + i).choice(p.n_train, n_s, replace=False)
                Xc = p.X_train[idx].reshape(n_s, -1).astype(np.float64)
                gt = _true_grad(Wg, bg, Xc, p.y_train[idx].astype(int), K)
                client_info.append({
                    "cos_update_vs_neg_true_grad": _cos(u, -gt),
                    "real_imgs": p.X_train[idx[:self.num_images]].reshape(min(self.num_images, n_s), -1),
                })

            # batched features: 1 predict per (recon, extractor, prep)
            n_img = self.num_images
            for r in RECONS:
                D_all = np.concatenate(recs[r], axis=0)
                for prep in PREPS:
                    D_in = D_all if prep == "raw" else np.clip(D_all, 0.0, 1.0)
                    for ex in EXTRACTORS:
                        F_all = self.extractors[ex].extract(D_in)
                        F = [F_all[i * n_img:(i + 1) * n_img] for i in range(len(participants))]
                        v_g = F_all
                        d = np.linalg.norm(v_g[:, None] - v_g[None], axis=-1)
                        nz = d[d > 0]
                        sigma_fix = max(float(np.median(nz)) if nz.size else 1.0, 1e-6)
                        for i, p in enumerate(participants):
                            key = (r, prep, ex, p.id)
                            v_cur = F[i]
                            v_hist = self._hist.get(key, v_cur)
                            m_cl, m_cg, m_lg = mmd_rbf(v_cur, v_hist), mmd_rbf(v_cur, v_g), mmd_rbf(v_hist, v_g)
                            f_cl, f_cg, f_lg = (_mmd2(v_cur, v_hist, sigma_fix), _mmd2(v_cur, v_g, sigma_fix),
                                                _mmd2(v_hist, v_g, sigma_fix))
                            row = {
                                "round": round_num, "client": p.id, "is_byz": bool(is_byz_list[i]),
                                "recon": r, "prep": prep, "extractor": ex, "S_R": srs[r][i],
                                "mmd2_cl": m_cl, "mmd2_cg": m_cg, "mmd2_lg": m_lg,
                                "S_cl": cue_similarity(m_cl), "S_cg": cue_similarity(m_cg), "S_lg": cue_similarity(m_lg),
                                "Ssqrt_cl": cue_similarity(np.sqrt(max(m_cl, 0))),
                                "Ssqrt_cg": cue_similarity(np.sqrt(max(m_cg, 0))),
                                "Ssqrt_lg": cue_similarity(np.sqrt(max(m_lg, 0))),
                                "mmd2fix_cl": f_cl, "mmd2fix_cg": f_cg, "mmd2fix_lg": f_lg,
                                "Sfix_cl": cue_similarity(f_cl), "Sfix_cg": cue_similarity(f_cg),
                                "Sfix_lg": cue_similarity(f_lg),
                                "drec_min": float(recs[r][i].min()), "drec_max": float(recs[r][i].max()),
                                "drec_mean": float(recs[r][i].mean()),
                                "drec_frac_outside_01": float(((recs[r][i] < 0) | (recs[r][i] > 1)).mean()),
                                "cos_update_vs_neg_true_grad": client_info[i]["cos_update_vs_neg_true_grad"],
                            }
                            if ex == "pretrained" and prep == "raw":
                                F_real = self.extractors[ex].extract(client_info[i]["real_imgs"])
                                row["mmd2_rec_vs_real"] = mmd_rbf(v_cur, F_real)
                            self.rows.append(row)
                            self._hist[key] = v_cur

            # --- official trajectory, identical to AdaAggRLGridLearner._run_round ---
            v_current = {p.id: self.feature_extractor.extract(D) for p, D in zip(participants, recs["orig"])}
            v_g = np.concatenate(list(v_current.values()), axis=0)
            if self._h is None:
                self._h = np.zeros(len(participants))
            state_rows = []
            for p, S_R in zip(participants, srs["orig"]):
                v_cur = v_current[p.id]
                v_hist = self._v_history.get(p.id, v_cur)
                state_rows.append([S_R, cue_similarity(mmd_rbf(v_cur, v_hist)),
                                   cue_similarity(mmd_rbf(v_cur, v_g)), cue_similarity(mmd_rbf(v_hist, v_g))])
            state_matrix = np.array(state_rows)
            mean_state = state_matrix.mean(axis=0)
            action = self.agent.select_action(mean_state)
            weights, new_h, _ = compute_weights_and_penalty(state_matrix, action, self._h, lam=self.lam)
            self._h = new_h
            tw = float(weights.sum())
            nw = np.ones(len(participants)) / len(participants) if tw < 1e-8 else weights / tw
            self._global_params = sum(w * th for w, th in zip(nw, theta_list))
            for p in participants:
                self._v_history[p.id] = v_current[p.id]
            new_model = self._make_model(n_feat)
            new_model.set_params(self._global_params)
            reward = old_loss - cross_entropy_loss(new_model, eval_data["X"], eval_data["y"])
            self.agent.train_step(mean_state, action, reward, mean_state)

            per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
            global_acc = float(np.mean(list(per_acc.values())))
            self.traj_rows.append({
                "round": round_num, "global_acc": global_acc,
                "weight_on_byz": float(sum(w for w, b in zip(nw, is_byz_list) if b)),
                "action": json.dumps([round(float(a), 4) for a in action]),
            })
            return RoundResult(round_num, global_acc, per_acc, trust_scores=None,
                               n_accepted=int((weights > 0).sum()))

    return DiagAdaAggRL


def run_cell(attack: str, alpha: float, seed: int, n_rounds: int) -> None:
    from src.experiments.exp9_dominance_grid import _split_name
    from src.utils.data_loader import load_dataset_participants

    np.random.seed(seed)
    DiagAdaAggRL = build_diag_learner_cls()
    participants, root = load_dataset_participants(
        "mnist", _split_name(alpha), 10, root_size=100, root_seed=seed,
    )
    learner = DiagAdaAggRL(
        n_rounds=n_rounds, n_classes=10, attack_type=attack,
        byzantine_ids=[0, 1], seed=seed, input_shape=INPUT_SHAPE,
        feature_extractor="random", dataset="mnist",
    )
    t0 = time.time()
    res = learner.train(participants, root_data=root, verbose=False)
    tag = f"{attack}_a{alpha}_seed{seed}_r{n_rounds}"
    os.makedirs(OUT_TABLES, exist_ok=True)
    pd.DataFrame(learner.rows).to_csv(f"{OUT_TABLES}/frente1_v0_cues_{tag}.csv", index=False)
    pd.DataFrame(learner.traj_rows).to_csv(f"{OUT_TABLES}/frente1_v0_traj_{tag}.csv", index=False)
    print(f"[{tag}] final acc={res[-1].global_accuracy:.4f}  ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# Part C
# ---------------------------------------------------------------------------

CUE_COLS = ["S_R", "S_cl", "S_cg", "S_lg", "Ssqrt_cl", "Ssqrt_cg", "Ssqrt_lg", "Sfix_cl", "Sfix_cg", "Sfix_lg"]


def analyze() -> None:
    from sklearn.metrics import roc_auc_score

    files = sorted(glob.glob(f"{OUT_TABLES}/frente1_v0_cues_*.csv"))
    if not files:
        print("no cue CSV found"); return
    auc_rows, spread_rows = [], []
    for f in files:
        cell = os.path.basename(f)[len("frente1_v0_cues_"):-4]
        df = pd.read_csv(f)
        df = df[df["round"] >= 2]  # round 1: v_hist = v_cur by construction
        for (r, prep, ex), g in df.groupby(["recon", "prep", "extractor"]):
            y = g["is_byz"].astype(int).values
            for c in CUE_COLS:
                x = g[c].values
                # cues are similarities: a Byzantine client should have a LOWER value -> AUC of -x
                auc = roc_auc_score(y, -x) if 0 < y.sum() < len(y) and np.std(x) > 0 else np.nan
                auc_rows.append({"cell": cell, "recon": r, "prep": prep, "extractor": ex,
                                 "cue": c, "auc_byz_lower": auc})
                spread_rows.append({"cell": cell, "recon": r, "prep": prep, "extractor": ex, "cue": c,
                                    "min": x.min(), "p50": np.median(x), "max": x.max(), "std": x.std()})
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(f"{OUT_TABLES}/frente1_v0_auc.csv", index=False)
    pd.DataFrame(spread_rows).to_csv(f"{OUT_TABLES}/frente1_v0_cue_spread.csv", index=False)

    pd.set_option("display.width", 250)
    base = auc[(auc.prep == "raw")]
    piv = base.pivot_table(index=["cell", "recon", "extractor"], columns="cue", values="auc_byz_lower")
    print("=== AUC (Byzantine < benign), prep=raw, rounds >= 2 ===")
    print(piv.round(2).to_string())
    piv2 = auc[auc.recon == "flip_global"].pivot_table(
        index=["cell", "prep", "extractor"], columns="cue", values="auc_byz_lower")
    print("\n=== AUC, recon=flip_global, raw vs clip01 ===")
    print(piv2.round(2).to_string())

    all_df = pd.concat([pd.read_csv(f).assign(cell=os.path.basename(f)[16:-4]) for f in files])
    o = all_df[(all_df.recon == "orig") & (all_df.prep == "raw")]
    print("\n=== cue range (current code: recon=orig, raw), per extractor ===")
    print(o.groupby("extractor")[["S_R", "S_cl", "S_cg", "S_lg", "mmd2_cl", "mmd2_cg", "mmd2_lg"]]
          .describe().T.round(4).to_string())
    print("\n=== range of D_rec and cos(update, true −∇), per recon ===")
    print(all_df[(all_df.prep == "raw") & (all_df.extractor == "random")].groupby(["recon", "is_byz"])[
        ["S_R", "drec_min", "drec_max", "drec_frac_outside_01", "cos_update_vs_neg_true_grad"]
    ].mean().round(3).to_string())
    if "mmd2_rec_vs_real" in all_df:
        print("\n=== fidelity: MMD²(D_rec features, features of the client's real images), pretrained extractor ===")
        print(all_df.dropna(subset=["mmd2_rec_vs_real"]).groupby("recon")["mmd2_rec_vs_real"]
              .describe().round(4).to_string())

    _plots(all_df)


def _plots(all_df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_FIGS, exist_ok=True)
    cells = sorted(all_df["cell"].unique())
    cues = ["S_R", "S_cl", "S_cg", "S_lg"]
    for recon in ["orig", "flip_global"]:
        fig, axes = plt.subplots(len(cells) * 2, 4, figsize=(15, 3.0 * len(cells) * 2), squeeze=False)
        for ci, cell in enumerate(cells):
            for ei, ex in enumerate(["random", "pretrained"]):
                g = all_df[(all_df.cell == cell) & (all_df.recon == recon) & (all_df.prep == "raw")
                           & (all_df.extractor == ex) & (all_df["round"] >= 2)]
                for k, c in enumerate(cues):
                    ax = axes[ci * 2 + ei, k]
                    for byz, color, lab in [(False, "#4C78A8", "benign"), (True, "#E45756", "Byzantine")]:
                        ax.hist(g.loc[g.is_byz == byz, c], bins=30, alpha=0.6, color=color, label=lab, density=True)
                    ax.set_title(f"{cell} | {ex} | {c}", fontsize=8)
                    ax.tick_params(labelsize=7)
                    if k == 0 and ci == 0 and ei == 0:
                        ax.legend(fontsize=7)
        fig.suptitle(f"Front 1 V0 — per-client cues (recon={recon}, rounds ≥ 2)")
        fig.tight_layout()
        path = f"{OUT_FIGS}/frente1_v0_cues_{recon}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"figure: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=["sign_flipping:0.5", "gaussian_noise:0.1", "label_flipping:0.5"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_rounds", type=int, default=15)
    ap.add_argument("--static_only", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    ap.add_argument("--skip_static", action="store_true")
    args = ap.parse_args()

    if args.analyze_only:
        analyze()
    else:
        if not args.skip_static:
            res = static_checks()
            os.makedirs(OUT_TABLES, exist_ok=True)
            with open(f"{OUT_TABLES}/frente1_v0_static_checks.json", "w") as fh:
                json.dump(res, fh, indent=2)
            print(json.dumps(res, indent=2))
        if not args.static_only:
            for cell in args.cells:
                a, al = cell.split(":")
                run_cell(a, float(al), args.seed, args.n_rounds)
