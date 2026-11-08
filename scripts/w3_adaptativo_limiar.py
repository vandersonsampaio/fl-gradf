"""
W3 (CONFIRMATORY, exploratory in scope): a simple adaptive attacker against the
skeleton's exclusion threshold. Plan in results/w3_adaptativo_limiar/PLANO.md
(frozen 2026-10-23, hash in PLANO.sha256). Changes nothing in `src/`.

Attack (PLANO §1), per active Byzantine client i at round t:
  u_i = (1 − ε_i) · h_i + ε_i · (−c · h̄_t),   h̄_t = mean honest update of the other clients
  feedback: accepted_i = final aggregation weight > 0 (white-box acceptance oracle)
  adaptive: ε_1 = 0.1; accepted -> ε = min(1, ε + step); excluded -> ε = ε / 2
  ipm2:     ε = 1 every round (the blind reference attack)
Honest updates are computed in client order exactly as the blind path does
(`AttackedFederatedLearner._compute_param_updates`), so the RNG consumption of local
training is the same as under any blind update-level attack.

Systems: fedavg, td3_ref, fixed_center (a = [0.5]*4, b = 0.5), fixed_b025 (b = 0.25).
Attackers [0, 1]; α ∈ {0.05; 0.1; 0.5}; H = 150.

Subcommands:
  run        one (system, attack, alpha, seed, rung) run
  verificar  PLANO §5.1 regression (15 rounds, seed 42, label_flipping α 0.05, td3_ref vs B2.7)
  gate       PLANO §4 validity gate for one rung (fixed_center + fedavg on seeds 125-129)
  analisar   PLANO §6 confirmatory analysis (only with the full 120-run grid)

Usage (CPU):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.w3_adaptativo_limiar run --system fixed_center --attack adaptive --alpha 0.05 --seed 125 --rung 1 --gate
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

OUT = "results/w3_adaptativo_limiar"
RAW = f"{OUT}/raw"
GATE_RAW = f"{OUT}/raw_gate"
ALPHAS = [0.05, 0.1, 0.5]
SEEDS = list(range(190, 200))
GATE_SEEDS = list(range(125, 130))
SYSTEMS = ["fedavg", "td3_ref", "fixed_center", "fixed_b025"]
GATE_SYSTEMS = ["fedavg", "fixed_center"]
ATTACKS = ["adaptive", "ipm2"]
HORIZONS = [15, 50, 150]
H_MAX = 150
EPS0 = 0.1
RUNGS = {1: {"c": 2.0, "step": 0.1}, 2: {"c": 4.0, "step": 0.1}, 3: {"c": 4.0, "step": 0.05}}
CONSTANTS = {"fixed_center": [0.5, 0.5, 0.5, 0.5, 0.5], "fixed_b025": [0.5, 0.5, 0.5, 0.5, 0.25]}
ACC_MARGIN = 1.0
GATE_MIN_HARM = 2.0       # p.p. below ipm2 under fixed_center
GATE_MIN_ACCEPT = 0.30
B27_RAW = "results/b27_horizonte/grade_raw.csv"


def mix_update(h: np.ndarray, h_bar: np.ndarray, eps: float, c: float) -> np.ndarray:
    return (1.0 - eps) * h + eps * (-c * h_bar)


def aimd(eps: float, accepted: bool, step: float) -> float:
    return min(1.0, eps + step) if accepted else eps / 2.0


def build(system, attack, seed, n_rounds, rung):
    from scripts.b27_horizonte import BYZ, INPUT_SHAPE
    from scripts.w_common import TAP, install_weight_tap
    from src.fl.attacked_learner import AttackedFederatedLearner

    install_weight_tap()  # before build_learner_cls (see scripts/w_common.py)
    from scripts.frente1_ablacao_adaaggrl import build_learner_cls

    cfg = RUNGS[rung]
    if system == "fedavg":
        base = AttackedFederatedLearner
        kwargs = dict(n_rounds=n_rounds, n_classes=10, byzantine_ids=BYZ, seed=seed, aggregation="fedavg")
    else:
        base = build_learner_cls()
        kwargs = dict(input_shape=INPUT_SHAPE, dataset="mnist", mode="td3", n_rounds=n_rounds,
                      n_classes=10, byzantine_ids=BYZ, seed=seed)
    # label_flipping is used only by `verificar` (regression through this script's path)
    kwargs["attack_type"] = "label_flipping" if attack == "label_flipping" else "adaptive_threshold"

    class Adaptive(base):
        def _compute_param_updates(self, participants, active):
            if self.attack_type == "label_flipping":
                return super()._compute_param_updates(participants, active)
            honest = []
            for p in participants:
                model = self._make_model(p.n_features)
                honest.append(model.fit(p.X_train, p.y_train))
            updates, is_byz = [], []
            for i, h in enumerate(honest):
                byz = (i in self.byzantine_ids) and active
                if byz:
                    h_bar = np.mean([u for j, u in enumerate(honest) if j != i], axis=0)
                    updates.append(mix_update(h, h_bar, self.eps[i], cfg["c"]))
                else:
                    updates.append(h)
                is_byz.append(byz)
            return updates, is_byz

        def _run_round(self, round_num, participants, root_data):
            TAP.clear()
            eps_used = dict(self.eps)
            rr = super()._run_round(round_num, participants, root_data)
            active = self._is_active(round_num) and self.attack_type != "label_flipping"
            if system == "fedavg":
                n_tr = np.array([p.n_train for p in participants], dtype=float)
                nw = n_tr / n_tr.sum()
            else:
                nw = TAP.normalized()
            row = {"round": round_num, "accuracy": float(rr.global_accuracy),
                   "att_mass": float(sum(nw[i] for i in BYZ)) if active else np.nan,
                   "b": float(TAP.action[4]) if TAP.action is not None else np.nan}
            for i in BYZ:
                acc_i = bool(nw[i] > 0)
                row[f"eps_{i}"] = eps_used[i] if active else np.nan
                row[f"accepted_{i}"] = acc_i if active else np.nan
                if active and attack == "adaptive":
                    self.eps[i] = aimd(self.eps[i], acc_i, cfg["step"])
            self.log.append(row)
            return rr

    learner = Adaptive(**kwargs)
    learner.eps = {i: (EPS0 if attack == "adaptive" else 1.0) for i in BYZ}
    learner.log = []
    if system in CONSTANTS:
        const = np.array(CONSTANTS[system])
        learner.agent.select_action = lambda state, *a, **k: const.copy()
        learner.agent.train_step = lambda *a, **k: None
    return learner


def _path(out_dir, system, attack, alpha, seed, rung, n_rounds):
    return f"{out_dir}/{system}_{attack}_a{alpha}_seed{seed}_rung{rung}_R{n_rounds}.csv"


def run(system, attack, alpha, seed, rung, n_rounds=H_MAX, out_dir=RAW):
    from scripts.b27_horizonte import _reseed
    from src.experiments.exp9_dominance_grid import _split_name
    from src.utils.data_loader import load_dataset_participants

    os.makedirs(out_dir, exist_ok=True)
    path = _path(out_dir, system, attack, alpha, seed, rung, n_rounds)
    if os.path.exists(path):
        print(f"already exists: {path}", flush=True)
        return path
    participants, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    _reseed(seed)
    learner = build(system, attack, seed, n_rounds, rung)
    t0 = time.time()
    learner.train(participants, root_data=root, verbose=False)
    df = pd.DataFrame(learner.log)
    for k, v in reversed(list({"system": system, "attack": attack, "alpha": alpha,
                               "seed": seed, "rung": rung}.items())):
        df.insert(0, k, v)
    df["sec"] = time.time() - t0
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    print(f"END {system} {attack} a{alpha} seed{seed} rung{rung} ({time.time() - t0:.0f}s)", flush=True)
    return path


def verificar():
    alpha, seed = 0.05, 42
    ref = pd.read_csv(B27_RAW)
    p = run("td3_ref", "label_flipping", alpha, seed, 1, n_rounds=15, out_dir=f"{OUT}/verificacao")
    new = float(pd.read_csv(p).query("round == 15").accuracy.iloc[0])
    old = float(ref[(ref.alpha == alpha) & (ref.attack_type == "label_flipping") & (ref.seed == seed)
                    & (ref.system == "td3_ref") & (ref.H == 15)].accuracy.iloc[0])
    text = (f"regression td3_ref (label_flipping a{alpha} seed{seed} H=15): new {new:.6f} vs reference "
            f"{old:.6f} |Δ| = {abs(new - old):.2e} " + ("OK" if abs(new - old) < 1e-9 else "FAIL") + "\n")
    open(f"{OUT}/verificacao.txt", "w").write(text)
    print(text)


def _accept_rate(df):
    cols = [c for c in df.columns if c.startswith("accepted_")]
    return float(np.nanmean(df[cols].to_numpy(dtype=float)))


def gate(rung):
    files = sorted(glob.glob(f"{GATE_RAW}/*_rung{rung}_R{H_MAX}.csv"))
    allr = pd.concat([pd.read_csv(f) for f in files]) if files else pd.DataFrame()
    n_exp = len(GATE_SYSTEMS) * len(ATTACKS) * len(ALPHAS) * len(GATE_SEEDS)
    got = allr.groupby(["system", "attack", "alpha", "seed"]).ngroups if len(allr) else 0
    lines = [f"W3 gate, rung {rung} {RUNGS[rung]}: {got}/{n_exp} runs"]
    if got < n_exp:
        lines.append("INCOMPLETE: no verdict")
    else:
        n_pass = 0
        for alpha in ALPHAS:
            fc = allr[(allr.system == "fixed_center") & (allr.alpha == alpha)]
            h = fc[fc["round"] == H_MAX].groupby("attack").accuracy.mean() * 100
            harm = h["ipm2"] - h["adaptive"]
            acc_rate = _accept_rate(fc[fc.attack == "adaptive"])
            fa = allr[(allr.system == "fedavg") & (allr.alpha == alpha) & (allr["round"] == H_MAX)]
            fa_acc = fa.groupby("attack").accuracy.mean() * 100
            ok = harm >= GATE_MIN_HARM and acc_rate >= GATE_MIN_ACCEPT
            n_pass += ok
            lines.append(f"  alpha {alpha}: fixed_center acc ipm2 − adaptive = {harm:+.2f} p.p. (≥ {GATE_MIN_HARM}); "
                         f"acceptance {acc_rate:.2%} (≥ {GATE_MIN_ACCEPT:.0%}) -> {'PASS' if ok else 'FAIL'}"
                         f" | fedavg acc: adaptive {fa_acc.get('adaptive', np.nan):.2f}, ipm2 {fa_acc.get('ipm2', np.nan):.2f}")
        lines.append(f"VERDICT rung {rung}: {n_pass}/3 cells -> "
                     + ("VALID (use this rung)" if n_pass >= 2 else "NOT VALID (next rung)"))
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/gate_rung{rung}.txt", "w").write(text)
    print(text)


def analisar(rung):
    from scripts.w_common import classify_family, paired_stats

    files = sorted(glob.glob(f"{RAW}/*_adaptive_*_rung{rung}_R{H_MAX}.csv"))
    allr = pd.concat([pd.read_csv(f) for f in files])
    allr.to_csv(f"{OUT}/grade_raw.csv", index=False)
    n_runs = allr.groupby(["system", "alpha", "seed"]).ngroups
    lines = [f"W3 confirmatory analysis, rung {rung} {RUNGS[rung]}: {n_runs}/"
             f"{len(SYSTEMS) * len(ALPHAS) * len(SEEDS)} runs"]
    rows = []
    for H in HORIZONS:
        h = allr[allr["round"] == H]
        fam = []
        for const in CONSTANTS:
            for alpha in ALPHAS:
                w = h[h.alpha == alpha].pivot_table(index="seed", columns="system", values="accuracy")
                d = (100 * (w["td3_ref"] - w[const])).dropna().to_numpy()
                r = paired_stats(d, ACC_MARGIN)
                r.update({"H": H, "constant": const, "alpha": alpha})
                fam.append(r)
        rows += classify_family(fam)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/deltas.csv", index=False)
    prim = res[res.H == H_MAX]
    lines.append(f"\nH1, H = {H_MAX}, D = td3_ref − constant (p.p.), Holm over 6:")
    for _, r in prim.iterrows():
        lines.append(f"  {r.constant:12s} a{r.alpha}: D = {r['mean']:+.2f} CI95 ({r.ic95_lo:+.2f}; {r.ic95_hi:+.2f}) "
                     f"CI90 ({r.ic90_lo:+.2f}; {r.ic90_hi:+.2f}) p_diff_holm {r.p_diff_holm:.4f} "
                     f"p_tost_holm {r.p_tost_holm:.4f} -> {r.verdict}")
    sup = prim[prim.verdict == "superiority"].groupby("constant").size()
    both = sum(all(((prim.alpha == a) & (prim.constant == c) & (prim.verdict == "superiority")).any()
                   for c in CONSTANTS) for a in ALPHAS)
    if both >= 2:
        overall = "adaptation pays off against this adversary"
    elif sup.get("fixed_b025", 0) == 0:
        overall = "no advantage"
    else:
        overall = "neither pre-declared reading"
    lines.append(f"Overall reading: {overall} (TD3 superior to both constants in {both}/3 cells)")
    lines.append("\nDescriptive (mean over seeds): accuracy at H=150 (%), acceptance rate, mean ε, mean b:")
    d = []
    for (alpha, system), g in allr.groupby(["alpha", "system"]):
        eps_cols = [c for c in g.columns if c.startswith("eps_")]
        d.append({"alpha": alpha, "system": system,
                  "acc_H150": 100 * g[g["round"] == H_MAX].accuracy.mean(),
                  "acceptance": _accept_rate(g), "eps_mean": float(np.nanmean(g[eps_cols].to_numpy(dtype=float))),
                  "b_mean": float(g.b.mean()), "b_sd_rounds": float(g.groupby("seed").b.std().mean())})
    lines.append(pd.DataFrame(d).round(4).to_string(index=False))
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/analise.txt", "w").write(text)
    print(text)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--system", required=True, choices=SYSTEMS)
    r.add_argument("--attack", required=True, choices=ATTACKS)
    r.add_argument("--alpha", type=float, required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--rung", type=int, required=True, choices=sorted(RUNGS))
    r.add_argument("--gate", action="store_true", help="development seeds 125-129, written to raw_gate/")
    sub.add_parser("verificar")
    for name in ("gate", "analisar"):
        s = sub.add_parser(name)
        s.add_argument("--rung", type=int, required=True, choices=sorted(RUNGS))
    a = p.parse_args()
    if a.cmd == "run":
        if a.gate:
            if a.seed not in GATE_SEEDS or a.system not in GATE_SYSTEMS:
                raise SystemExit("gate runs are fedavg/fixed_center on seeds 125-129 only (PLANO §4)")
            run(a.system, a.attack, a.alpha, a.seed, a.rung, out_dir=GATE_RAW)
        else:
            if a.seed not in SEEDS or a.attack != "adaptive":
                raise SystemExit("confirmation is the adaptive attack on seeds 190-199 (PLANO §2)")
            run(a.system, a.attack, a.alpha, a.seed, a.rung)
    elif a.cmd == "verificar":
        verificar()
    elif a.cmd == "gate":
        gate(a.rung)
    else:
        analisar(a.rung)


if __name__ == "__main__":
    main()
