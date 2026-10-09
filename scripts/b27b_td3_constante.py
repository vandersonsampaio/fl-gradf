"""
B2.7a/b (results/b27b_td3_constante/PLANO.md). Exploratory.

B2.7a: re-runs B2.7's `td3_ref` (the ablation learner, td3 mode) on the 3
cells with a positive trend, H = 150, seeds 42-51, and records per round
the mean state fed to the policy and the executed action, plus the actor
parameters (W_actor, b_actor) at the start of rounds 0, 50 and 100 and at the end of round 150.
The logging only wraps `agent.select_action` and `_run_round` (copies, without consuming
RNG), so the trajectory is B2.7's. The analysis checks this (|Δacc| < 1e-9).

Measures (states of rounds 101-150, deterministic π = sigmoid(s·W + b)):
  drift      mean |π150(s) − π0(s)|
  sd_estados mean over the 5 dims of the standard deviation of π150(s) across states
  a_TD3      mean of π150(s) over those states (5-vector), and distance to the center 0.5
σ_a = 0.15 (exploration_sigma of the in-house framework's TD3 agent).

Usage (CPU, no GPU):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.b27b_td3_constante b27a --alpha 0.05 --attack label_flipping --seed 42
  venv/bin/python -m scripts.b27b_td3_constante analisar_b27a
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

OUT = "results/b27b_td3_constante"
CELLS = [(0.05, "label_flipping"), (0.1, "label_flipping"), (0.05, "low_mag_backdoor")]
SEEDS = list(range(42, 52))
HORIZONS = [15, 50, 150]
H_MAX = 150
SNAP_ROUNDS = [0, 50, 100]  # number of rounds already completed at the snapshot; 150 = end of the last one
SIGMA_A = 0.15
CENTER = 0.5
B27_RAW = "results/b27_horizonte/grade_raw.csv"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def policy(state, W, b):
    return _sigmoid(np.asarray(state) @ W + b)


def run_b27a(alpha, attack, seed, out_dir=None):
    from scripts.b27_horizonte import BYZ, INPUT_SHAPE, _reseed
    from scripts.frente1_ablacao_adaaggrl import build_learner_cls
    from src.experiments.exp9_dominance_grid import _split_name
    from src.utils.data_loader import load_dataset_participants

    out_dir = out_dir or f"{OUT}/b27a_raw"
    os.makedirs(out_dir, exist_ok=True)
    tag = f"td3_{attack}_a{alpha}_seed{seed}"
    path = f"{out_dir}/{tag}.npz"
    if os.path.exists(path):
        print(f"already exists: {path}", flush=True)
        return path
    Ablation = build_learner_cls()
    participants, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    _reseed(seed)  # as in B2.7, before each system
    learner = Ablation(input_shape=INPUT_SHAPE, dataset="mnist", mode="td3", n_rounds=H_MAX, n_classes=10,
                       attack_type=attack, byzantine_ids=BYZ, seed=seed)
    agent = learner.agent
    states, actions, snaps = [], [], {}
    orig_select = agent.select_action

    def recording_select(state, *a, **k):
        act = orig_select(state, *a, **k)
        states.append(np.array(state, dtype=np.float64))
        actions.append(np.array(act, dtype=np.float64))
        return act
    agent.select_action = recording_select

    orig_round = learner._run_round

    def recording_round(round_num, parts, root_data):
        done = round_num - 1  # train numbers rounds starting from 1
        if done in SNAP_ROUNDS:
            snaps[done] = (agent.W_actor.copy(), agent.b_actor.copy())
        return orig_round(round_num, parts, root_data)
    learner._run_round = recording_round

    t0 = time.time()
    res = learner.train(participants, root_data=root, verbose=False)
    snaps[H_MAX] = (agent.W_actor.copy(), agent.b_actor.copy())
    accs = {H: float(res[H - 1].global_accuracy) for H in HORIZONS}
    np.savez(path + ".tmp.npz", states=np.stack(states), actions=np.stack(actions),
             **{f"W{r}": snaps[r][0] for r in snaps}, **{f"b{r}": snaps[r][1] for r in snaps},
             acc=np.array([accs[H] for H in HORIZONS]), horizons=np.array(HORIZONS))
    os.replace(path + ".tmp.npz", path)
    print(f"END {tag} ({time.time() - t0:.0f}s)", flush=True)
    return path


def analisar_b27a():
    ref = pd.read_csv(B27_RAW)
    ref = ref[ref.system == "td3_ref"]
    rows = []
    for alpha, attack in CELLS:
        for s in SEEDS:
            p = f"{OUT}/b27a_raw/td3_{attack}_a{alpha}_seed{s}.npz"
            if not os.path.exists(p):
                continue
            z = np.load(p)
            S = z["states"][100:150]
            pi0, pi150 = policy(S, z["W0"], z["b0"]), policy(S, z["W150"], z["b150"])
            a_td3 = pi150.mean(axis=0)
            old = ref[(ref.alpha == alpha) & (ref.attack_type == attack) & (ref.seed == s)].set_index("H")["accuracy"]
            dacc = max(abs(z["acc"][i] - old.get(H, np.nan)) for i, H in enumerate(HORIZONS))
            rows.append({"alpha": alpha, "attack_type": attack, "seed": s,
                         "drift": float(np.mean(np.abs(pi150 - pi0))),
                         "sd_estados": float(np.mean(pi150.std(axis=0))),
                         "sd_estados_pi0": float(np.mean(pi0.std(axis=0))),
                         "dist_centro": float(np.mean(np.abs(a_td3 - CENTER))),
                         **{f"a_td3_{i}": float(a_td3[i]) for i in range(5)},
                         "acao_exec_media_b": float(z["actions"][100:150, 4].mean()),
                         "max_dacc_vs_b27": float(dacc)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/b27a_por_run.csv", index=False)
    df[["alpha", "attack_type", "seed"] + [f"a_td3_{i}" for i in range(5)]].to_csv(f"{OUT}/a_td3.csv", index=False)
    ok = df["max_dacc_vs_b27"].max() < 1e-9
    med = df.groupby(["alpha", "attack_type"])[["drift", "sd_estados", "sd_estados_pi0", "dist_centro"]].median()
    cond1 = int((med["sd_estados"] > SIGMA_A).sum())
    lines = ["B2.7a — TD3 mechanism at H = 150 (descriptive)", f"runs: {len(df)} (expected 30)",
             f"Check against B2.7: max |Δacc| = {df['max_dacc_vs_b27'].max():.2e} -> "
             + ("REPRODUCES" if ok else "DOES NOT REPRODUCE: B2.7a invalidated"), "",
             f"Median across seeds (σ_a = {SIGMA_A}):", med.round(4).to_string(), "",
             "Mean a_TD3 per cell (mean across seeds; dims a1..a4, b):",
             df.groupby(["alpha", "attack_type"])[[f"a_td3_{i}" for i in range(5)]].mean().round(3).to_string(), "",
             f"B2.7c condition 1 (median sd_estados > σ_a in ≥ 2/3 cells): {cond1}/3 -> "
             + ("MET" if cond1 >= 2 else "NOT met")]
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/b27a_analise.txt", "w").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("b27a")
    c.add_argument("--alpha", type=float, required=True)
    c.add_argument("--attack", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--out_dir", default=None)
    sub.add_parser("analisar_b27a")
    a = ap.parse_args()
    if a.cmd == "b27a":
        run_b27a(a.alpha, a.attack, a.seed, a.out_dir)
    else:
        analisar_b27a()


if __name__ == "__main__":
    main()
