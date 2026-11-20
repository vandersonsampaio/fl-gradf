"""
W2 (CONFIRMATORY): a real backdoor with Attack Success Rate in the AdaAggRL skeleton.
Plan in results/w2_backdoor_asr/PLANO.md (frozen 2026-10-23, hash in PLANO.sha256).

Attack `pixel_backdoor` (src/classification/attack_simulator.py): data-level, a square
trigger in the bottom-right corner + label t = 0 on a fraction p of each attacker's
samples, every active round. Attackers [0, 1] (BYZ of B2.7).

Systems:
  fedavg_clean    FedAvg, attack none (ASR floor, clean ceiling)
  fedavg          FedAvg under pixel_backdoor
  td3_ref         ablation learner in td3 mode (as in B2.7c)
  fixed_center    same learner, constant a = [0.5]*4, b = 0.5 (constant path of B2.7b)
  fixed_b025      same learner, constant a = [0.5]*4, b = 0.25 (fixed a priori, B2.6/B2.7c)
  cosserver_only  ablation learner in cosserver_only mode (server signal, b = 0.5)

Per round, every run records clean accuracy, ASR over the pooled test shards of the 10
clients (true label != t, triggered) and the attackers' weight mass.

Subcommands:
  run        one (system, alpha, seed) run
  gate       PLANO §4 validity gate for one rung (fedavg_clean + fedavg on seeds 92-96 only)
  verificar  PLANO §5 checks: 15-round regression vs B2.7/B2.7b + ASR sanity
  analisar   PLANO §6 confirmatory analysis (only with the full 180-run grid)

Usage (CPU):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.w2_backdoor_asr run --system td3_ref --alpha 0.05 --seed 180 --rung 1
  venv/bin/python -m scripts.w2_backdoor_asr verificar
  venv/bin/python -m scripts.w2_backdoor_asr gate --rung 1      # after the gate runs exist
  venv/bin/python -m scripts.w2_backdoor_asr analisar --rung 1
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

OUT = "results/w2_backdoor_asr"
RAW = f"{OUT}/raw"
GATE_RAW = f"{OUT}/raw_gate"
ALPHAS = [0.05, 0.1, 0.5]
SEEDS = list(range(180, 190))
GATE_SEEDS = list(range(92, 97))
SYSTEMS = ["fedavg_clean", "fedavg", "td3_ref", "fixed_center", "fixed_b025", "cosserver_only"]
GATE_SYSTEMS = ["fedavg_clean", "fedavg"]
HORIZONS = [15, 50, 150]
H_MAX = 150
TARGET = 0
RUNGS = {1: {"trigger_size": 4, "poison_fraction": 0.5},
         2: {"trigger_size": 6, "poison_fraction": 0.5},
         3: {"trigger_size": 6, "poison_fraction": 0.8}}
CONSTANTS = {"fixed_center": [0.5, 0.5, 0.5, 0.5, 0.5], "fixed_b025": [0.5, 0.5, 0.5, 0.5, 0.25]}
ASR_MARGIN = 5.0   # p.p. of ASR (PLANO §6 H1)
ACC_MARGIN = 1.0   # p.p. of clean accuracy (PLANO §6 H2)
GATE_MIN_ASR = 50.0
GATE_MAX_ACC_DROP = 5.0
B27_RAW = "results/b27_horizonte/grade_raw.csv"
B27B_RAW = "results/b27b_td3_constante/b27b_raw"


def _path(out_dir, system, alpha, seed, rung, n_rounds):
    return f"{out_dir}/{system}_a{alpha}_seed{seed}_rung{rung}_R{n_rounds}.csv"


def build(system, attack, seed, n_rounds, backdoor_kwargs):
    """Returns a learner instance that records per-round rows in `.log`."""
    from scripts.b27_horizonte import BYZ, INPUT_SHAPE
    from scripts.w_common import TAP, install_weight_tap
    from src.fl.attacked_learner import AttackedFederatedLearner

    install_weight_tap()  # before build_learner_cls, so its closure imports the tapped function
    from scripts.frente1_ablacao_adaaggrl import build_learner_cls

    if system.startswith("fedavg"):
        base = AttackedFederatedLearner
        kwargs = dict(n_rounds=n_rounds, n_classes=10, attack_type=attack, byzantine_ids=BYZ,
                      seed=seed, aggregation="fedavg")
    else:
        base = build_learner_cls()
        mode = "cosserver_only" if system == "cosserver_only" else "td3"
        kwargs = dict(input_shape=INPUT_SHAPE, dataset="mnist", mode=mode, n_rounds=n_rounds,
                      n_classes=10, attack_type=attack, byzantine_ids=BYZ, seed=seed)

    class Recorded(base):
        def _run_round(self, round_num, participants, root_data):
            TAP.clear()
            rr = super()._run_round(round_num, participants, root_data)
            active = self._is_active(round_num)
            if system.startswith("fedavg"):
                n_tr = np.array([p.n_train for p in participants], dtype=float)
                nw = n_tr / n_tr.sum()
            else:
                nw = TAP.normalized()
            m = self._make_model(participants[0].n_features)
            from src.classification.attack_simulator import attack_success_rate
            self.log.append({
                "round": round_num, "accuracy": float(rr.global_accuracy),
                "asr": attack_success_rate(m, self._Xte, self._yte, TARGET,
                                           self.backdoor_kwargs["trigger_size"]),
                "att_mass": float(sum(nw[i] for i in BYZ)) if (active and attack != "none") else np.nan,
                "b": float(TAP.action[4]) if TAP.action is not None else np.nan,
            })
            return rr

    learner = Recorded(**kwargs)
    learner.backdoor_kwargs = {"target_class": TARGET, **backdoor_kwargs}
    learner.log = []
    if system in CONSTANTS:
        const = np.array(CONSTANTS[system])
        learner.agent.select_action = lambda state, *a, **k: const.copy()
        learner.agent.train_step = lambda *a, **k: None
    return learner


def run(system, alpha, seed, rung, n_rounds=H_MAX, out_dir=RAW, attack=None):
    from scripts.b27_horizonte import _reseed
    from scripts.w_common import pooled_test
    from src.experiments.exp9_dominance_grid import _split_name
    from src.utils.data_loader import load_dataset_participants

    if system not in SYSTEMS:
        raise ValueError(system)
    os.makedirs(out_dir, exist_ok=True)
    path = _path(out_dir, system, alpha, seed, rung, n_rounds)
    if os.path.exists(path):
        print(f"already exists: {path}", flush=True)
        return path
    if attack is None:
        attack = "none" if system == "fedavg_clean" else "pixel_backdoor"
    participants, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    _reseed(seed)  # as in B2.7, before each system
    learner = build(system, attack, seed, n_rounds, RUNGS[rung])
    learner._Xte, learner._yte = pooled_test(participants)
    t0 = time.time()
    learner.train(participants, root_data=root, verbose=False)
    df = pd.DataFrame(learner.log)
    df.insert(0, "system", system)
    df.insert(1, "attack_type", attack)
    df.insert(2, "alpha", alpha)
    df.insert(3, "seed", seed)
    df.insert(4, "rung", rung)
    df["sec"] = time.time() - t0
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    print(f"END {system} a{alpha} seed{seed} rung{rung} ({time.time() - t0:.0f}s)", flush=True)
    return path


def verificar():
    """PLANO §5: (1) regression — this script's path, with the tap and the ASR recorder,
    under label_flipping must reproduce B2.7 td3_ref and B2.7b fixed_b025 at H = 15;
    (3) ASR sanity on a model trained only on triggered data, and on the untrained model."""
    from scripts.b27_horizonte import BYZ, _reseed  # noqa: F401
    from scripts.w_common import pooled_test
    from src.classification.attack_simulator import apply_trigger, attack_success_rate
    from src.experiments.exp9_dominance_grid import _split_name
    from src.fl.federated_learner import _LogisticModel
    from src.utils.data_loader import load_dataset_participants

    alpha, attack, seed = 0.05, "label_flipping", 42
    lines = []
    ref = {"td3_ref": pd.read_csv(B27_RAW),
           "fixed_b025": pd.read_csv(f"{B27B_RAW}/fixed_b025_{attack}_a{alpha}_seed{seed}_R150.csv")}
    vdir = f"{OUT}/verificacao"
    for system in ["td3_ref", "fixed_b025"]:
        p = run(system, alpha, seed, 1, n_rounds=15, out_dir=vdir, attack=attack)
        new = float(pd.read_csv(p).query("round == 15").accuracy.iloc[0])
        r = ref[system]
        old = float(r[(r.alpha == alpha) & (r.attack_type == attack) & (r.seed == seed)
                      & (r.system == system) & (r.H == 15)].accuracy.iloc[0])
        lines.append(f"regression {system} ({attack} a{alpha} seed{seed} H=15): new {new:.6f} vs reference "
                     f"{old:.6f} |Δ| = {abs(new - old):.2e} " + ("OK" if abs(new - old) < 1e-9 else "FAIL"))

    participants, _ = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    Xte, yte = pooled_test(participants)
    n_feat = Xte.shape[1]
    for ts in (4, 6):
        X0 = np.concatenate([p.X_train for p in participants])
        Xtr = apply_trigger(X0, ts)
        m = _LogisticModel(n_feat, n_classes=10)
        for _ in range(5):
            m.set_params(m.params + m.fit(Xtr, np.full(len(Xtr), TARGET)))
        asr_bd = attack_success_rate(m, Xte, yte, TARGET, ts)
        m0 = _LogisticModel(n_feat, n_classes=10)
        asr_0 = attack_success_rate(m0, Xte, yte, TARGET, ts)
        lines.append(f"ASR sanity trigger {ts}x{ts}: all-triggered model ASR = {asr_bd:.4f} "
                     + ("OK" if asr_bd > 0.99 else "FAIL")
                     + f"; untrained model ASR = {asr_0:.4f} (base rate of predicting t)")
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/verificacao.txt", "w").write(text)
    print(text)


def _at(df, H):
    return df[df["round"] == H]


def gate(rung):
    files = sorted(glob.glob(f"{GATE_RAW}/*_rung{rung}_R{H_MAX}.csv"))
    allr = pd.concat([pd.read_csv(f) for f in files])
    n_exp = len(GATE_SYSTEMS) * len(ALPHAS) * len(GATE_SEEDS)
    got = allr.groupby(["system", "alpha", "seed"]).ngroups
    lines = [f"W2 gate, rung {rung} {RUNGS[rung]}: {got}/{n_exp} runs"]
    if got < n_exp:
        lines.append("INCOMPLETE: no verdict")
    else:
        ok_all = True
        h = _at(allr, H_MAX)
        for alpha in ALPHAS:
            g = h[h.alpha == alpha].groupby("system")[["asr", "accuracy"]].mean() * 100
            d_asr = g.loc["fedavg", "asr"] - g.loc["fedavg_clean", "asr"]
            d_acc = g.loc["fedavg_clean", "accuracy"] - g.loc["fedavg", "accuracy"]
            ok = d_asr >= GATE_MIN_ASR and d_acc <= GATE_MAX_ACC_DROP
            ok_all &= ok
            lines.append(f"  alpha {alpha}: ASR fedavg − clean = {d_asr:+.2f} p.p. (≥ {GATE_MIN_ASR}); "
                         f"clean acc drop = {d_acc:+.2f} p.p. (≤ {GATE_MAX_ACC_DROP}) -> {'PASS' if ok else 'FAIL'}")
        lines.append(f"VERDICT rung {rung}: " + ("VALID (use this rung)" if ok_all else "NOT VALID (next rung)"))
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/gate_rung{rung}.txt", "w").write(text)
    print(text)


def analisar(rung):
    from scripts.w_common import classify_family, paired_stats

    files = sorted(glob.glob(f"{RAW}/*_rung{rung}_R{H_MAX}.csv"))
    allr = pd.concat([pd.read_csv(f) for f in files])
    allr.to_csv(f"{OUT}/grade_raw.csv", index=False)
    n_runs = allr.groupby(["system", "alpha", "seed"]).ngroups
    lines = [f"W2 confirmatory analysis, rung {rung} {RUNGS[rung]}: {n_runs}/"
             f"{len(SYSTEMS) * len(ALPHAS) * len(SEEDS)} runs"]

    rows = []
    for H in HORIZONS:
        h = _at(allr, H)
        for metric, margin in (("asr", ASR_MARGIN), ("accuracy", ACC_MARGIN)):
            fam = []
            for const in CONSTANTS:
                for alpha in ALPHAS:
                    w = h[h.alpha == alpha].pivot_table(index="seed", columns="system", values=metric)
                    d = (100 * (w["td3_ref"] - w[const])).dropna().to_numpy()
                    r = paired_stats(d, margin)
                    r.update({"H": H, "metric": metric, "constant": const, "alpha": alpha,
                              "td3_ref": 100 * w["td3_ref"].mean(), "const_mean": 100 * w[const].mean()})
                    fam.append(r)
            rows += classify_family(fam)
    res = pd.DataFrame(rows)
    # PLANO §6: D = td3_ref − c. For ASR, negative favors TD3 (lower attack success).
    res["reading"] = [
        ({"superiority": "TD3 less secure", "inferiority": "TD3 more secure"}.get(v, v) if m == "asr"
         else {"superiority": "TD3 more accurate", "inferiority": "TD3 less accurate"}.get(v, v))
        for v, m in zip(res.verdict, res.metric)]
    res.to_csv(f"{OUT}/deltas.csv", index=False)

    prim = res[res.H == H_MAX]
    for metric, title in (("asr", "H1 (primary, ASR, margin ±5 p.p.)"), ("accuracy", "H2 (clean accuracy, margin ±1 p.p.)")):
        lines.append(f"\n{title}, H = {H_MAX}, D = td3_ref − constant (p.p.), Holm over 6:")
        for _, r in prim[prim.metric == metric].iterrows():
            lines.append(f"  {r.constant:12s} a{r.alpha}: D = {r['mean']:+.2f} CI95 ({r.ic95_lo:+.2f}; {r.ic95_hi:+.2f}) "
                         f"CI90 ({r.ic90_lo:+.2f}; {r.ic90_hi:+.2f}) p_diff_holm {r.p_diff_holm:.4f} "
                         f"p_tost_holm {r.p_tost_holm:.4f} -> {r.reading}")
    lines.append("\nJoint reading (accuracy equivalent but ASR not equivalent):")
    for const in CONSTANTS:
        for alpha in ALPHAS:
            a = prim[(prim.metric == "accuracy") & (prim.constant == const) & (prim.alpha == alpha)].verdict.iloc[0]
            s = prim[(prim.metric == "asr") & (prim.constant == const) & (prim.alpha == alpha)].verdict.iloc[0]
            flag = a == "equivalence" and s != "equivalence"
            lines.append(f"  {const:12s} a{alpha}: acc {a}, ASR {s}" + ("  <-- FLAG" if flag else ""))

    lines.append("\nDescriptive, H = 150, mean over seeds (%):")
    desc = _at(allr, H_MAX).groupby(["alpha", "system"])[["accuracy", "asr", "att_mass"]].mean()
    desc[["accuracy", "asr"]] *= 100
    lines.append(desc.round(3).to_string())
    lines.append("\nDescriptive: mean ASR over rounds 1-150 (%):")
    lines.append((allr.groupby(["alpha", "system"]).asr.mean() * 100).unstack().round(2).to_string())
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/analise.txt", "w").write(text)
    print(text)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--system", required=True, choices=SYSTEMS)
    r.add_argument("--alpha", type=float, required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--rung", type=int, required=True, choices=sorted(RUNGS))
    r.add_argument("--gate", action="store_true", help="development seeds 92-96, written to raw_gate/")
    sub.add_parser("verificar")
    g = sub.add_parser("gate")
    g.add_argument("--rung", type=int, required=True, choices=sorted(RUNGS))
    a = sub.add_parser("analisar")
    a.add_argument("--rung", type=int, required=True, choices=sorted(RUNGS))
    args = p.parse_args()
    if args.cmd == "run":
        if args.gate:
            if args.seed not in GATE_SEEDS or args.system not in GATE_SYSTEMS:
                raise SystemExit("gate runs are fedavg_clean/fedavg on seeds 92-96 only (PLANO §4)")
            run(args.system, args.alpha, args.seed, args.rung, out_dir=GATE_RAW)
        else:
            if args.seed not in SEEDS:
                raise SystemExit("confirmation seeds are 180-189 (PLANO §2)")
            run(args.system, args.alpha, args.seed, args.rung)
    elif args.cmd == "verificar":
        verificar()
    elif args.cmd == "gate":
        gate(args.rung)
    else:
        analisar(args.rung)


if __name__ == "__main__":
    main()
