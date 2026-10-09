"""
B2.1 + B2.2 analysis, following results/b21_replicacao_oficial/PREREGISTRO.md.
Written before any grid result existed. Reuses the tests from
`analisar.py` (Step 2) without changing it.

B2.1 (unit = attack×seed pair, n=20):
  Primary: median accuracy over rounds 401-500 (runs truncated at 500 steps).
  H1: fixed - td3. Paired TOST (t), margin ±1.0 p.p., alpha 0.05; paired Wilcoxon
      for a difference. Per attack: Δ, CI95, d, Wilcoxon + Holm.
  Secondary: mean 451-500 (Step 2 metric), number of resets, mass on attackers.

B2.2 (family of 3 hypotheses, Holm over H3-H5, alpha 0.05; one-sided tests):
  σ_a = 0.1 × 0.95/2 = 0.0475 (std of SB3's exploration noise in action units).
  H3: step-by-step correlation of td3's EXECUTED actions (rounds 101-500, 5 dims)
      between EB and LMP of the same seed is > 0.9. One-sided Wilcoxon on
      atanh(r) - atanh(0.9) > 0, n=10.
  H4: the final deterministic policy barely differs from the initial one:
      drift = mean |π_500(s) - π_0(s)| over the states s of rounds 401-500 and
      the 5 dims. One-sided Wilcoxon drift - σ_a < 0, n=20 td3 runs.
  H5: the final policy barely depends on its input:
      S_swap = mean |π_500(s_attack,t) - π_500(s_other_attack,t)|, t in 401-500,
      same seed. One-sided Wilcoxon S_swap - σ_a < 0, n=20.
  Reported without a test: S_shuffle (state from another round of the same run), and
  drift/S_swap of the initial actor π_0 as a reference.

Usage:
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_b21.py
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analisar as P2  # noqa: E402  (tost_paired, describe, holm)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

MARGIN = 0.010
ALPHA = 0.05
ATTACKS = ["LMP", "EB"]
SEEDS = list(range(105, 115))
SIGMA_A = 0.1 * 0.95 / 2
R_THRESH = 0.9
ROUNDS = 500


def _tag(att, cond, seed):
    return f"MNIST_{att}_q0.5_{cond}_seed{seed}_R{ROUNDS}"


def load_runs(raw):
    rows = []
    for att in ATTACKS:
        for cond in ["fixed", "td3"]:
            for s in SEEDS:
                f = os.path.join(raw, _tag(att, cond, s) + ".json")
                if not os.path.exists(f):
                    continue
                d = json.load(open(f))
                steps = d["steps"][:ROUNDS]  # Step 2 addendum 1: SB3's td3 runs 501 steps
                acc = np.array([x["acc"] for x in steps])
                real = np.array([x["n_att_real"] for x in steps])
                mass = np.array([np.nan if x["att_weight_mass"] is None else x["att_weight_mass"] for x in steps])
                rows.append({
                    "attack": att, "condition": cond, "seed": s, "n_steps": len(steps),
                    "n_steps_raw": len(d["steps"]),
                    "primary_median_401_500": float(np.median(acc[400:500])),
                    "mean_451_500": float(np.mean(acc[450:500])),
                    "n_resets_extra": len(d["resets"]) - 1,
                    "att_mass_mean": float(np.nanmean(mass[real > 0])) if (real > 0).any() else np.nan,
                    "actions": np.array([x["action"] for x in steps]),
                })
    return pd.DataFrame(rows)


def paired_block(tab, metric_label, lines):
    pair = tab[["fixed", "td3"]].dropna()
    d = (pair["fixed"] - pair["td3"]).to_numpy()
    desc = P2.describe(d)
    p_tost = max(P2.tost_paired(d, MARGIN))
    lines += [f"== H1 ({metric_label}): fixed − td3 (pooled, attack×seed pairs) ==",
              f"  n={desc['n']}  Δ={desc['delta_pp']:+.2f} p.p.  CI95=({desc['ci95_pp'][0]:+.2f}, {desc['ci95_pp'][1]:+.2f})  d={desc['d']:+.2f}",
              f"  TOST ±{100*MARGIN:.1f} p.p.: p={p_tost:.4f}   Wilcoxon: p={desc['wilcoxon_p']:.4f}",
              f"  VERDICT: {P2.verdict(desc, p_tost)}"]
    per = []
    for att in ATTACKS:
        if att in pair.index.get_level_values(0):
            da = (pair.loc[att]["fixed"] - pair.loc[att]["td3"]).to_numpy()
            if len(da) >= 2:
                per.append((att, P2.describe(da)))
    if per:
        adj = P2.holm(np.array([x[1]["wilcoxon_p"] for x in per]))
        for (att, ds), pa in zip(per, adj):
            lines.append(f"    {att}: n={ds['n']} Δ={ds['delta_pp']:+.2f} p.p. CI95=({ds['ci95_pp'][0]:+.2f}, {ds['ci95_pp'][1]:+.2f}) d={ds['d']:+.2f} Wilcoxon p={ds['wilcoxon_p']:.4f} Holm={pa:.4f}")
    lines.append("")


# ---------------------------------------------------------------------------
# B2.2: rebuild the SB3 actor to evaluate the deterministic policy
# ---------------------------------------------------------------------------

def make_actor_evaluator():
    import gymnasium
    import torch
    from stable_baselines3 import TD3

    class _Dummy(gymnasium.Env):
        observation_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(10, 4), dtype=np.float32)
        action_space = gymnasium.spaces.Box(0.0, 0.95, shape=(5,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            return np.zeros((10, 4), np.float32), {}

        def step(self, a):
            return np.zeros((10, 4), np.float32), 0.0, False, False, {}

    model = TD3("MlpPolicy", _Dummy(), policy_kwargs={"net_arch": [256, 128]}, device="cpu", verbose=0)

    def policy(actor_path, obs):
        model.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
        acts, _ = model.predict(obs.astype(np.float32), deterministic=True)
        return np.asarray(acts, dtype=np.float64)

    return policy


def one_sided_wilcoxon_less(x):
    """p-value for H1: median(x) < 0."""
    return float(stats.wilcoxon(x, alternative="less").pvalue)


def one_sided_wilcoxon_greater(x):
    return float(stats.wilcoxon(x, alternative="greater").pvalue)


def b22(df, raw, lines):
    policy = make_actor_evaluator()
    td3 = df[df["condition"] == "td3"].set_index(["attack", "seed"])
    rng = np.random.RandomState(0)

    # H3: correlation of the executed actions across attacks, same seed
    rs = []
    for s in SEEDS:
        if ("EB", s) in td3.index and ("LMP", s) in td3.index:
            a, b = td3.loc[("EB", s), "actions"][100:500], td3.loc[("LMP", s), "actions"][100:500]
            rs.append((s, float(np.corrcoef(a.ravel(), b.ravel())[0, 1])))
    r = np.array([x[1] for x in rs])
    p3 = one_sided_wilcoxon_greater(np.arctanh(np.clip(r, -0.999999, 0.999999)) - np.arctanh(R_THRESH))

    # H4 and H5: deterministic policy
    rows = []
    for (att, s) in td3.index:
        other = "LMP" if att == "EB" else "EB"
        tag, otag = _tag(att, "td3", s), _tag(other, "td3", s)
        p0 = os.path.join(raw, "actors", f"{tag}_step000.pt")
        p500 = os.path.join(raw, "actors", f"{tag}_step500.pt")
        obs = np.load(os.path.join(raw, "obs", f"{tag}.npy"))[:ROUNDS]
        oobs_path = os.path.join(raw, "obs", f"{otag}.npy")
        if not (os.path.exists(p0) and os.path.exists(p500) and os.path.exists(oobs_path)):
            continue
        oobs = np.load(oobs_path)[:ROUNDS]
        S = obs[400:500]
        S_other = oobs[400:500]
        S_shuf = obs[rng.randint(0, len(obs), size=len(S))]
        pi0, pi0_o = policy(p0, S), policy(p0, S_other)
        pi5, pi5_o, pi5_sh = policy(p500, S), policy(p500, S_other), policy(p500, S_shuf)
        rows.append({
            "attack": att, "seed": s,
            "drift": float(np.mean(np.abs(pi5 - pi0))),
            "S_swap": float(np.mean(np.abs(pi5 - pi5_o))),
            "S_shuffle": float(np.mean(np.abs(pi5 - pi5_sh))),
            "S_swap_pi0": float(np.mean(np.abs(pi0 - pi0_o))),
            "pi0_dist_center": float(np.mean(np.abs(pi0 - 0.475))),
        })
    m = pd.DataFrame(rows)
    p4 = one_sided_wilcoxon_less(m["drift"].to_numpy() - SIGMA_A) if len(m) else np.nan
    p5 = one_sided_wilcoxon_less(m["S_swap"].to_numpy() - SIGMA_A) if len(m) else np.nan
    adj = P2.holm(np.array([p3, p4, p5]))

    lines += ["== B2.2: mechanistic hypotheses (Holm over H3–H5) ==",
              f"  σ_a (exploration noise, action units) = {SIGMA_A:.4f}",
              f"  H3 corr(EB, LMP) of actions > {R_THRESH}: n={len(r)}  r per seed={np.round(r, 3).tolist()}",
              f"     median r={np.median(r):.3f}  one-sided Wilcoxon p={p3:.4f}  Holm={adj[0]:.4f}  -> {'CONFIRMED' if adj[0] < ALPHA else 'NOT confirmed'}",
              f"  H4 drift |π500−π0| < σ_a: n={len(m)}  median={m['drift'].median():.4f}  max={m['drift'].max():.4f}",
              f"     one-sided Wilcoxon p={p4:.4f}  Holm={adj[1]:.4f}  -> {'CONFIRMED' if adj[1] < ALPHA else 'NOT confirmed'}",
              f"  H5 S_swap |π500(s)−π500(s')| < σ_a: n={len(m)}  median={m['S_swap'].median():.4f}  max={m['S_swap'].max():.4f}",
              f"     one-sided Wilcoxon p={p5:.4f}  Holm={adj[2]:.4f}  -> {'CONFIRMED' if adj[2] < ALPHA else 'NOT confirmed'}",
              "  References (no test): median S_shuffle={:.4f}; median S_swap of π0={:.4f}; median |π0−center|={:.4f}".format(
                  m["S_shuffle"].median(), m["S_swap_pi0"].median(), m["pi0_dist_center"].median()),
              ""]
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="results/b21_replicacao_oficial/raw")
    p.add_argument("--out", default="results/b21_replicacao_oficial/analise.txt")
    a = p.parse_args()

    df = load_runs(a.raw_dir)
    lines = [f"B2.1 + B2.2 — {len(df)} runs loaded (expected {len(ATTACKS) * 2 * len(SEEDS)})", ""]
    tab = df.pivot_table(index=["attack", "seed"], columns="condition", values="primary_median_401_500")
    lines += ["Primary metric (median accuracy over rounds 401–500):", tab.round(4).to_string(), ""]
    paired_block(tab, "primary: median 401–500", lines)
    tab2 = df.pivot_table(index=["attack", "seed"], columns="condition", values="mean_451_500")
    paired_block(tab2, "secondary: mean 451–500, Step 2 metric", lines)
    lines += ["== Descriptive secondary metrics (mean per attack × condition) ==",
              df.groupby(["attack", "condition"])[["n_resets_extra", "att_mass_mean"]].mean().round(4).to_string(), ""]

    m = b22(df, a.raw_dir, lines)

    out_dir = os.path.dirname(a.out)
    df.drop(columns=["actions"]).to_csv(os.path.join(out_dir, "resumo_runs.csv"), index=False)
    m.to_csv(os.path.join(out_dir, "b22_mecanismo.csv"), index=False)
    text = "\n".join(lines)
    open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
