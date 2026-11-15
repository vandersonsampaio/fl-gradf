"""
Front 1 under FLTrust step parity (results/f1_server_parity/PLAN.md). Changes nothing in src/
and nothing in scripts/frente1_ablacao_adaaggrl.py.

Front 1 computed the server's root update (cos_server, and the reference the fltrust_aligned
attacker clones) with the native fit: the clients' epochs over the 100-sample root, 20 SGD steps
against roughly 750-920 for the clients. P1 later matched the server's step count to the
clients' median (`server_steps="median"` on the P1 branches). This script measures what that
changes in Front 1, with the parity fit copied from P1: `steps` mini-batches drawn with
replacement from the root, one pass, `steps` = median over clients of
ceil(n_train / batch) * epochs.

Variants (the native trajectory is never perturbed: every parity fit saves and restores the
global NumPy RNG, and the native fits still run where Front 1 ran them):
  detector  step 1 along the stored td3 trajectory, recording cos_server_parity next to
            Front 1's columns, which must equal the stored CSVs.
  attack    the 3 fltrust_aligned cells, td3 with record, the attacker cloning the parity
            server update instead of the native one; records both cos_server columns.
  grid      cosserver_only, Front 1's step-3 grid (21 cells, seeds 42-51), with the detector's
            server update fitted with parity; the attack keeps the native reference.

Usage:
  python -m scripts.f1_server_parity run --variant detector [--workers 7]
  python -m scripts.f1_server_parity run --variant attack
  python -m scripts.f1_server_parity run --variant grid
  python -m scripts.f1_server_parity analyze
"""

import argparse
import glob
import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np
import pandas as pd

OUT = "results/f1_server_parity"
F1 = "results/frente1_ablacao_adaaggrl"
SEEDS_GRID = list(range(42, 52))
ALPHAS = [0.05, 0.1, 0.5]
FA = "fltrust_aligned"


def parity_steps(learner, participants) -> int:
    per = [int(np.ceil(p.n_train / learner.local_batch_size)) * learner.local_epochs for p in participants]
    return int(np.median(per))


def parity_fit(learner, participants, root: Dict, params: np.ndarray) -> np.ndarray:
    """P1's parity fit, from `params`, with the global NumPy RNG saved and restored."""
    state = np.random.get_state()
    steps = parity_steps(learner, participants)
    model = learner._instantiate_model(participants[0].n_features)
    model.set_params(params.copy())
    idx = np.random.randint(0, len(root["y"]), steps * learner.local_batch_size)
    model.n_epochs = 1
    delta = model.fit(root["X"][idx], root["y"][idx])
    np.random.set_state(state)
    return delta


def build(variant: str):
    import src.fl.attacked_learner as al
    import scripts.frente1_ablacao_adaaggrl as f1

    original = al.compute_param_updates_auto

    def recording(learner, participants, active, root_data):
        ups, byz = original(learner, participants, active, root_data)
        learner._last_updates = ups
        return ups, byz
    al.compute_param_updates_auto = recording   # bound by build_learner_cls below
    Base = f1.build_learner_cls()
    al.compute_param_updates_auto = original

    class ParityLearner(Base):
        def _run_round(self, round_num, participants, root_data):
            root = root_data or self._carve_root(participants[0])
            self._parity_ref = parity_fit(self, participants, root, self._global_params)
            self._parity_steps = parity_steps(self, participants)
            n_rows = len(self.rows)
            if variant == "attack":
                sim, poison = self._sim, self._sim.poison_update

                def poison_parity(update, attack_type, reference=None, **kw):
                    return poison(update, attack_type, reference=self._parity_ref, **kw)
                sim.poison_update = poison_parity
            try:
                res = super()._run_round(round_num, participants, root_data)
            finally:
                if variant == "attack":
                    self._sim.poison_update = poison
            for k, row in enumerate(self.rows[n_rows:]):
                u = self._last_updates[k]
                row["cos_server_parity"] = float(u @ self._parity_ref /
                                                 (np.linalg.norm(u) * np.linalg.norm(self._parity_ref) + 1e-12))
                row["norm_parity_ref"] = float(np.linalg.norm(self._parity_ref))
                row["parity_steps"] = self._parity_steps
            return res
    return ParityLearner


def run_cell(args):
    variant, seed, alpha, attack = args
    from src.experiments.exp9_dominance_grid import _split_name
    from src.utils.data_loader import load_dataset_participants
    import scripts.frente1_ablacao_adaaggrl as f1

    Learner = build_grid() if variant == "grid" else build(variant)
    participants, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    mode = "cosserver_only" if variant == "grid" else "td3"
    record = variant != "grid"
    t0 = time.time()
    learner = Learner(n_rounds=15, n_classes=10, attack_type=attack, byzantine_ids=[0, 1], seed=seed,
                      input_shape=f1.INPUT_SHAPE, dataset="mnist", mode=mode, record=record)
    acc = learner.train(participants, root_data=root, verbose=False)[-1].global_accuracy
    if record:
        os.makedirs(f"{OUT}/{variant}", exist_ok=True)
        pd.DataFrame(learner.rows).to_csv(f"{OUT}/{variant}/td3_{attack}_a{alpha}_seed{seed}.csv", index=False)
    return {"variant": variant, "alpha": alpha, "attack_type": attack, "seed": seed, "accuracy": acc,
            "seconds": time.time() - t0}


def build_grid():
    """cosserver_only with the score's server update fitted with parity."""
    import scripts.frente1_ablacao_adaaggrl as f1
    from src.classification.attack_simulator import AttackSimulator

    informed = set(AttackSimulator.INFORMED_ATTACK_TYPES)
    Base = f1.build_learner_cls()

    class GridParityLearner(Base):
        replaced = 0

        def _run_round(self, round_num, participants, root_data):
            root = root_data or self._carve_root(participants[0])
            ref = parity_fit(self, participants, root, self._global_params)
            native_make = self._make_model
            calls = {"root_fits": 0}
            # Root fits in a cosserver_only round: the informed attacker's reference first
            # (informed attacks only), then the score's server update, which Front 1 fits with
            # the RNG saved before and restored after. Only that last one is replaced.
            target = 2 if self.attack_type in informed else 1

            def make_model(n_feat):
                m = native_make(n_feat)
                fit = m.fit

                def fit_hook(X, y):
                    if X is root["X"]:
                        calls["root_fits"] += 1
                        if calls["root_fits"] == target:
                            self.replaced += 1
                            return ref
                    return fit(X, y)
                m.fit = fit_hook
                return m
            self._make_model = make_model
            try:
                res = super()._run_round(round_num, participants, root_data)
            finally:
                self._make_model = native_make
            if calls["root_fits"] != target:
                raise SystemExit(f"guard: {calls['root_fits']} root fits in round {round_num}, expected {target}")
            return res
    return GridParityLearner


def cells(variant):
    from src.experiments.exp9_dominance_grid import BLIND_ATTACKS, INFORMED_ATTACKS
    attacks = INFORMED_ATTACKS + BLIND_ATTACKS
    if variant == "detector":
        return [(variant, 42, a, t) for a in ALPHAS for t in attacks]
    if variant == "attack":
        return [(variant, 42, a, FA) for a in ALPHAS]
    return [(variant, s, a, t) for s in SEEDS_GRID for a in ALPHAS for t in attacks]


def run(variant, workers):
    os.makedirs(OUT, exist_ok=True)
    todo = cells(variant)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(run_cell, todo))
    pd.DataFrame(rows).to_csv(f"{OUT}/{variant}_runs.csv", index=False)
    print(pd.DataFrame(rows).groupby("variant").seconds.describe())


def analyze():
    from sklearn.metrics import roc_auc_score
    import scripts.frente1_ablacao_adaaggrl as f1

    out = []
    # check: Front 1's columns along the detector variant equal the stored CSVs
    cols = ["S_R", "S_cl", "S_cg", "S_lg", "norm", "cos_median", "l2_median", "cos_server", "final_weight"]
    maxdiff = 0.0
    for f in sorted(glob.glob(f"{OUT}/detector/td3_*.csv")):
        a, b = pd.read_csv(f), pd.read_csv(f"{F1}/detectores/{os.path.basename(f)}")
        maxdiff = max(maxdiff, float((a[cols] - b[cols]).abs().max().max()))
    out.append(f"CHECK detector variant reproduces Front 1's stored step-1 CSVs: max |diff| = {maxdiff:.3g}")

    dets = dict(f1.DETECTORS, cos_server_parity=-1)
    rows = []
    for variant in ("detector", "attack"):
        for f in sorted(glob.glob(f"{OUT}/{variant}/td3_*.csv")):
            name = os.path.basename(f)[4:-4]
            attack, rest = name.rsplit("_a", 1)
            alpha = float(rest.split("_seed")[0])
            d = pd.read_csv(f)
            d = d[d["round"] >= 2]
            y = d["is_byz"].astype(int).values
            for det, sign in dets.items():
                x = d[det].values * sign
                ok = 0 < y.sum() < len(y) and np.nanstd(x) > 0
                rows.append({"variant": variant, "attack_type": attack, "alpha": alpha, "detector": det,
                             "auc": roc_auc_score(y, x) if ok else np.nan})
            if attack == FA:
                g = d.groupby("is_byz")["norm"].median()
                out.append(f"{variant} {attack} a{alpha}: median norm byz/honest = {g[True] / g[False]:.3f}, "
                           f"parity steps {int(d.parity_steps.iloc[0])}")
    auc = pd.DataFrame(rows)
    auc.to_csv(f"{OUT}/auc.csv", index=False)
    piv = auc[auc.variant == "detector"].pivot_table(index=["attack_type", "alpha"], columns="detector", values="auc")
    pd.set_option("display.width", 220)
    out.append("=== detector variant: AUC per cell (rounds >= 2, seed 42) ===\n" + piv.round(2).to_string())
    m = piv.mean()
    out.append("=== mean over 21 cells ===\n" + m.round(3).sort_values(ascending=False).to_string())
    no_fa = piv.drop(index=FA, level="attack_type").mean()
    out.append("=== mean over 18 cells (no fltrust_aligned) ===\n" + no_fa.round(3).sort_values(ascending=False).to_string())
    out.append("=== cells with AUC >= 0.70 ===\n" + (piv >= 0.70).sum().to_string())
    piv_a = auc[auc.variant == "attack"].pivot_table(index=["attack_type", "alpha"], columns="detector", values="auc")
    out.append("=== attack variant (fltrust_aligned cloning the parity server update) ===\n" + piv_a.round(2).to_string())
    # mixed 21-cell means: 18 cells from the detector variant + the 3 fltrust_aligned cells from the attack variant
    mixed = pd.concat([piv.drop(index=FA, level="attack_type"), piv_a]).mean()
    out.append("=== 21-cell mean with fltrust_aligned under the parity attack ===\n"
               + mixed.round(3).sort_values(ascending=False).to_string())

    # grid: cosserver_only with parity vs Front 1's cosserver_only and sr_only
    if os.path.exists(f"{OUT}/grid_runs.csv"):
        g = pd.read_csv(f"{OUT}/grid_runs.csv")[["alpha", "attack_type", "seed", "accuracy"]]
        ref = pd.concat([pd.read_csv(f) for f in glob.glob(f"{F1}/raw/*_seed*.csv")])
        key = ["alpha", "attack_type", "seed"]
        w = ref.pivot_table(index=key, columns="mode", values="accuracy")
        w = w.join(g.set_index(key).accuracy.rename("cosserver_parity"), how="inner")
        out.append(f"grid cells matched: {len(w)}")
        for a, b in (("cosserver_parity", "cosserver_only"), ("cosserver_parity", "sr_only"),
                     ("cosserver_only", "sr_only")):
            d = (w[a] - w[b]).groupby(level="seed").mean()
            lo, hi = np.percentile([d.sample(len(d), replace=True, random_state=i).mean() for i in range(20000)],
                                   [5, 95])
            out.append(f"{a} - {b}: mean {d.mean():+.4f}, 90% bootstrap CI ({lo:+.4f}; {hi:+.4f}), n_seeds {len(d)}")
        per = (w["cosserver_parity"] - w["sr_only"]).groupby(level="attack_type").mean()
        per0 = (w["cosserver_only"] - w["sr_only"]).groupby(level="attack_type").mean()
        out.append("=== per attack, minus sr_only: native | parity ===\n"
                   + pd.DataFrame({"native": per0, "parity": per}).round(4).to_string())
    text = "\n\n".join(out)
    with open(f"{OUT}/analysis.txt", "w") as f:
        f.write(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--variant", required=True, choices=["detector", "attack", "grid"])
    r.add_argument("--workers", type=int, default=7)
    sub.add_parser("analyze")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.variant, a.workers)
    else:
        analyze()


if __name__ == "__main__":
    main()
