"""
Analysis of the P1 step-parity reruns (results/p1_parity/PLAN.md, items 1-4),
with the shared P1 inference code (scripts/p1/inference.py).

  Sec. 1  completeness and server_steps of every parity run
  Sec. 2  free regression check against results/p1_invariance, and the root
          invariant under parity
  Sec. 3  MNIST E1: root dependence under parity; the 2x2 root x server steps
          of FLTrust against the native seeded runs
  Sec. 4  MNIST E2: d*, in-sample and cross-fitted headroom
  Sec. 5  MNIST E3: capture on the 18 confirmatory cells
  Sec. 6  CIFAR-10: block C with parity replaces E0b block C for sigma_E0 and
          the tau_CNN check (results/p1_cifar10/e0b/PLAN.md, Secs. 3-4, R = 110)
  Sec. 7  compute budget with the measured costs, without and with the E1
          factorial (FLTrust native arms at both roots)

AdaAggRL does not use g0 and was not rerun: its rows come from the published
runs (unseeded). Under fltrust_aligned they still hold the native g0, which is
one more reason that attack stays outside the confirmatory grid.

Writes results/p1_parity/analysis/ and results/p1_tables/parity/.

Usage: venv/bin/python scripts/p1/analyze_parity.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import analyze_e0b as e0b  # noqa: E402
import audit_mnist as audit  # noqa: E402
import inference as inf  # noqa: E402
import make_tables_figures as mk  # noqa: E402
from analyze_invariance import load_seeded  # noqa: E402

RAW = "results/p1_parity"
OUT = f"{RAW}/analysis"
TAB = "results/p1_tables/parity"
E0B = "results/p1_cifar10/e0b"
INV_JSON = "results/p1_invariance/analysis.json"
FA = "fltrust_aligned"
RELATIONAL = ["median", "trimmed_mean", "krum"]
ROOTS = (100, 3000, 50)
SEEDS = list(range(42, 52))
R = 110
SYSTEMS = ["Random", "Oracle (decoupled)", "GRADF", "FedStrategist", "AdaAggRL"]
SENSITIVITY = ["AdaAggRL (pretrained)", "FedStrategist (GRADF detector)"]
E3_COLS = ["family", "system", "mean", "ci95_lo", "ci95_hi", "p_wilcoxon", "p_wilcoxon_holm",
           "p_t", "d_z", "p_tost", "category"]


def load_parity_fixed():
    frames = []
    for d in sorted(glob.glob(f"{RAW}/mnist_fixed/raw/root*_seed*")):
        if not os.path.exists(f"{d}/done"):
            raise SystemExit(f"incomplete job: {d}")
        root = int(os.path.basename(d).split("_")[0].removeprefix("root"))
        frames.append(pd.read_csv(glob.glob(f"{d}/*_stepsmedian_raw.csv")[0]).assign(root=root))
    return pd.concat(frames, ignore_index=True)


def load_parity_systems():
    frames = []
    for v in "ba":
        for d in sorted(glob.glob(f"{RAW}/mnist_rq3/raw/variant{v}_seed*")):
            if not os.path.exists(f"{d}/done"):
                raise SystemExit(f"incomplete job: {d}")
            df = pd.read_csv(glob.glob(f"{d}/*_stepsmedian_raw.csv")[0])
            if v == "a":
                df = df.assign(system="FedStrategist (GRADF detector)")
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_published_adaaggrl(sy):
    pub = audit.load_systems()
    pub = pub[pub.system.isin(["AdaAggRL", "AdaAggRL (pretrained)"])]
    cols = audit.CELL + ["system", "seed", "accuracy"]
    return pd.concat([sy[cols], pub[cols]], ignore_index=True)


def completeness(fx, sy, blockc, out):
    out += ["## Sec. 1 Completeness"]
    n = fx.groupby("root").size().to_dict()
    out.append(f"mnist_fixed rows per root: {n} (expected 100: 1470, 3000: 840, 50: 840)")
    out.append(f"mnist_rq3 rows: {sy.groupby('system').size().to_dict()} (expected 210 each)")
    out.append(f"cifar_blockC jobs: {len(blockc)}/12; rounds {sorted({len(d['accuracy']) for d in blockc.values()})}")
    steps = set(fx.server_steps) | set(sy.server_steps) | {d["server_steps"] for d in blockc.values()}
    out.append(f"server_steps values: {sorted(steps)}")
    ok = n == {100: 1470, 3000: 840, 50: 840} and (sy.groupby("system").size() == 210).all() \
        and len(blockc) == 12 and steps == {"median"}
    out.append("OK" if ok else "NOT OK")
    return ok


def regression(fx, nat, out):
    out += ["", "## Sec. 2 Regression and invariant"]
    key = ["alpha", "attack_type", "strategy", "seed", "root"]
    m = fx.merge(nat, on=key, suffixes=("_par", "_nat"))
    m["diff"] = m.accuracy_par - m.accuracy_nat
    rows = []
    for name, g in (("non-FLTrust rules, other 6 attacks", m[(m.strategy != "fltrust") & (m.attack_type != FA)]),
                    (f"non-FLTrust rules, {FA}", m[(m.strategy != "fltrust") & (m.attack_type == FA)]),
                    ("FLTrust, all attacks", m[m.strategy == "fltrust"])):
        rows.append({"subset": name, "n_pairs": len(g), "n_nonzero": int((g["diff"] != 0).sum()),
                     "max_abs": float(g["diff"].abs().max())})
    reg = pd.DataFrame(rows)
    reg.to_csv(f"{OUT}/regression_vs_native.csv", index=False)
    out.append("Parity vs results/p1_invariance (expected: first row all zero; the others change):")
    out.append(reg.to_string(index=False))

    wide = fx.pivot_table(index=["alpha", "attack_type", "strategy", "seed"], columns="root", values="accuracy")
    inv = []
    for r in (100, 50):
        d = (wide[r] - wide[3000]).rename("diff").reset_index()
        d = d[d.strategy.isin(RELATIONAL)]
        for fa, g in d.groupby(d.attack_type == FA):
            inv.append({"root": r, "subset": FA if fa else "other 6 attacks", "n_pairs": len(g),
                        "n_nonzero": int((g["diff"] != 0).sum()), "max_abs": float(g["diff"].abs().max())})
    inv = pd.DataFrame(inv)
    inv.to_csv(f"{OUT}/invariance_check.csv", index=False)
    out += ["", "Root invariant under parity (median, TM, Krum; root r vs 3000):", inv.to_string(index=False)]
    return bool(reg.iloc[0].n_nonzero == 0 and (inv[inv.subset != FA].n_nonzero == 0).all())


def fmt(r):
    return (f"mean {100 * r['mean']:+6.2f} p.p., CI95 ({100 * r['ci95_lo']:+.2f}; {100 * r['ci95_hi']:+.2f}), "
            f"Wilcoxon p {r['p_wilcoxon']:.4f}, t p {r['p_t']:.4f}, d_z {r['d_z']:.2f}, {r['category']}")


def e1(fx, nat, res, out):
    out += ["", "## Sec. 3 MNIST E1 (V = FLTrust - TM; D_s over seeds)"]
    roots = {r: fx[fx.root == r].drop(columns="root") for r in ROOTS}
    natr = {r: nat[nat.root == r].drop(columns="root") for r in ROOTS}
    for name, excl in (("21 cells", ()), ("18 cells (no fltrust_aligned)", (FA,))):
        for lab, src in (("parity", roots), ("native", natr)):
            a, _ = inf.e1_root_dependence(src[100], src[3000], exclude=excl)
            b, _ = inf.e1_root_dependence(src[50], src[100], exclude=excl)
            res[f"E1_3000_vs_100_{lab}_{name}"] = a
            res[f"E1_100_vs_50_{lab}_{name}"] = b
            out.append(f"{name:<30} {lab:<7} 3000 vs 100: {fmt(a)}")
            out.append(f"{name:<30} {lab:<7}  100 vs  50: {fmt(b)}")

    # 2x2 of FLTrust: root x server steps, 18 confirmatory cells, per seed
    f = {}
    for lab, src in (("median", fx), ("native", nat)):
        s = src[(src.strategy == "fltrust") & (src.attack_type != FA)]
        f[lab] = s.pivot_table(index=["alpha", "attack_type", "seed"], columns="root", values="accuracy")
    contrasts = {
        "root effect at parity (3000 - 100)": f["median"][3000] - f["median"][100],
        "root effect, native (3000 - 100)": f["native"][3000] - f["native"][100],
        "steps effect at root 100 (parity - native)": f["median"][100] - f["native"][100],
        "steps effect at root 3000 (parity - native)": f["median"][3000] - f["native"][3000],
    }
    out += ["", "FLTrust 2x2, 18 cells, seed-level:"]
    rows = []
    for k, d in contrasts.items():
        r = inf.describe(d.groupby(level="seed").mean().values)
        res[f"E1_2x2 {k}"] = r
        out.append(f"  {k:<45} {fmt(r)}")
        by_a = d.groupby(level="alpha").mean()
        rows.append({"contrast": k, **{f"alpha_{a}": 100 * by_a[a] for a in (0.5, 0.1, 0.05)},
                     "cells_positive": int((d.groupby(level=["alpha", "attack_type"]).mean() > 0).sum())})
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/e1_2x2_by_alpha.csv", index=False)
    out += ["", "By alpha (p.p.), cells_positive of 18:", t.round(2).to_string(index=False)]


def e2(fx, res, out):
    out += ["", "## Sec. 4 MNIST E2 (root 100; D_R = FLTrust, median, TM, Krum)"]
    r100 = fx[fx.root == 100].drop(columns="root")
    with open(INV_JSON) as f:
        native = json.load(f)
    for name, key, g in (("21 cells", "E2_all_21", r100), ("18 cells", "E2_no_fltrust_aligned_18", r100[r100.attack_type != FA])):
        r = {**inf.e2_headroom(g), **inf.e2_selection_stability(g)}
        res[f"E2_{name}"] = r
        means = g[g.strategy.isin(inf.D_R)].groupby("strategy").accuracy.mean().mul(100).round(2).to_dict()
        n = native[key]
        out.append(f"{name}: d* = {r['dstar_insample']} (LOSO agreement {r['dstar_agreement']:.2f}), "
                   f"cell-winner agreement {r['cell_winner_agreement']:.2f}")
        out.append(f"  global means %: {means}")
        out.append(f"  H in-sample {100 * r['H_insample']:.2f} p.p.; H_LOSO {100 * r['H_loso']:.2f} p.p., "
                   f"CI95 ({100 * r['H_loso_ci95'][0]:.2f}; {100 * r['H_loso_ci95'][1]:.2f}), {r['category']}")
        out.append(f"  native, for comparison: d* = {n['dstar_insample']}, H in-sample {100 * n['H_insample']:.2f}, "
                   f"H_LOSO {100 * n['H_loso']:.2f}, CI95 ({100 * n['H_loso_ci95'][0]:.2f}; {100 * n['H_loso_ci95'][1]:.2f})")


def e3(fx, sy, res, out):
    out += ["", "## Sec. 5 MNIST E3 (18 confirmatory cells; accuracy differences as fractions)"]
    r100 = fx[fx.root == 100].drop(columns="root")
    fx18, sy18 = r100[r100.attack_type != FA], sy[sy.attack_type != FA]
    t, desc = inf.e3_capture(fx18, sy18, SYSTEMS)
    ts, _ = inf.e3_capture(fx18, sy18, SENSITIVITY + ["Random"])
    t.to_csv(f"{OUT}/e3_confirmatory_18cells.csv", index=False)
    ts.to_csv(f"{OUT}/e3_sensitivity_18cells.csv", index=False)
    res["E3_descriptive_18"] = desc
    out += [t[E3_COLS].round(4).to_string(index=False), "", "Sensitivity:", ts[E3_COLS].round(4).to_string(index=False),
            "", f"Descriptive: {desc}"]
    prev = pd.read_csv("results/p1_audit/mnist_e3_confirmatory_18cells_seeded.csv")
    out += ["", "Earlier figures (native fixed rules, published unseeded systems; not a causal contrast, parity PLAN notes):",
            prev[["family", "system", "mean", "ci95_lo", "ci95_hi", "category"]].round(4).to_string(index=False)]


def tables(fx, sy, out):
    os.makedirs(TAB, exist_ok=True)
    fixed = {str(r): fx[fx.root == r].drop(columns="root") for r in (3000, 100, 50)}
    S = mk.systems_by_cell(sy)
    t3 = audit.table3(fixed["100"], sy).rename(
        columns={"Attack-aware": "Oracle (decoupled)", "FedStrat.": "FedStrategist"})
    rnd = audit.per_seed(sy, "system")["Random"]
    t4, t5 = audit.vs_reference(sy, rnd), audit.vs_reference(sy, rnd, by="attack_type")
    tex1, t1 = mk.table1_tex(fixed)
    tex2, t2 = mk.table2_tex(fixed)
    parts = [("Table 1: win counts by root size", tex1), ("Table 2: FLTrust wins by alpha", tex2),
             ("Table 3: full grid under canonical root", mk.table3_tex(t3)),
             ("Table 4: pooled paired test vs Random", mk.table4_tex(t4)),
             ("Table 5: per-attack-type paired test vs Random", mk.table5_tex(t5))]
    with open(f"{TAB}/tables.tex", "w") as f:
        f.write("% Generated by scripts/p1/analyze_parity.py from the raw per-seed CSVs (server_steps = median;\n"
                "% AdaAggRL from the published runs, native g0 under fltrust_aligned); do not hand-edit.\n\n")
        for title, lines in parts:
            f.write(f"% ---- {title} ----\n" + "\n".join(lines) + "\n\n")
    mk.figure1(fixed, f"{TAB}/fig1_winner_heatmap_by_root.png")
    counts = mk.figure2_counts(fixed["100"], sy)
    mk.figure2(counts, f"{TAB}/fig2_captured_headroom.png")
    mk.figure3(S, f"{TAB}/fig3_systems_by_attack.png")
    with open(f"{TAB}/figure_data.tex", "w") as f:
        f.write("% Generated by scripts/p1/analyze_parity.py. Do not hand-edit.\n\n")
        f.write("\n".join(mk.figure_data_tex(fixed, counts, S)) + "\n")
    t3.to_csv(f"{OUT}/table3_grid_root100.csv")
    out += ["", "Tables and figures (results/p1_tables/parity/):", t1.to_string(), t2.to_string(index=False),
            f"Fig. 2 counts (21 cells): {counts}"]
    above = {k: int(t3[f"bold_{k}"].sum()) for k in ("GRADF", "FedStrat.", "AdaAggRL")}
    out.append(f"Cells above the cell-wise best fixed rule (of 21): {above}")
    h18 = audit.headroom(fixed["100"], sy, drop_fa=True)
    out.append(f"18 cells, own d*: d* = {h18['dstar']}, headroom cells {h18['headroom_cells']}, "
               f"captured {h18['captured']}, H {100 * h18['H_global']:.2f} p.p.")


def cifar(res, out):
    out += ["", f"## Sec. 6 CIFAR-10: block C with parity (R = {R})"]
    native = e0b.load_raw(f"{E0B}/raw")
    parity = e0b.load_raw(f"{RAW}/cifar_blockC/raw")
    runs = {k: d for k, d in native.items() if k[2] != "fltrust"}
    runs.update(parity)
    out.append(f"Merged runs: {len(runs)}/27 (blocks A and B from E0b, block C with parity)")
    if any(k not in runs for k in e0b.expected_keys()):
        raise SystemExit("incomplete CIFAR-10 merge")
    tau, max_honest = e0b.layer1(runs, out)
    tau_old = json.load(open(f"{E0B}/decisions.json"))["tau_cnn"]
    out.append(f"tau_CNN frozen in E0b: {tau_old:.4f}; with parity block C: {tau:.4f}")
    s1, s3 = e0b.sigma(runs, R, out)
    res.update(tau_cnn=tau, tau_cnn_e0b=tau_old, max_honest_l2=max_honest, sigma_e0_e1=s1, sigma_e0_e3=s3)

    rows = []
    for (a, attack, _, root, seed), d in sorted(parity.items()):
        nat = native[(a, attack, "fltrust", root, seed)]
        rows.append({"attack": attack, "root": root, "seed": seed, "server_steps": d["server_steps_n"],
                     "g0_median": float(np.median(d["g0_l2"])),
                     "acc_R_parity": 100 * d["accuracy"][R - 1], "acc_R_native": 100 * nat["accuracy"][R - 1],
                     "acc_180_parity": 100 * d["accuracy"][-1], "acc_180_native": 100 * nat["accuracy"][-1]})
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/cifar_blockC_runs.csv", index=False)
    g = t.groupby(["attack", "root"])
    summ = g.agg(server_steps=("server_steps", "median"), g0=("g0_median", "median"),
                 parity_R=("acc_R_parity", "mean"), parity_sd=("acc_R_parity", "std"),
                 native_R=("acc_R_native", "mean"), native_sd=("acc_R_native", "std"),
                 parity_180=("acc_180_parity", "mean"), native_180=("acc_180_native", "mean")).round(2)
    out += ["", "Per cell, 3 seeds, % (descriptive):", summ.to_string(), "", "Per run:", t.round(2).to_string(index=False)]
    for attack in e0b.ATTACKS:
        v = t[(t.attack == attack)].pivot(index="seed", columns="root", values="acc_R_parity")
        out.append(f"Root contrast at R, parity, {attack} (2500 - 100): "
                   f"{(v[2500] - v[100]).mean():+.2f} (per seed {', '.join(f'{x:+.2f}' for x in v[2500] - v[100])})")
    fedavg = {att: np.mean([100 * native[(0.05, att, 'fedavg', 100, s)]['accuracy'][R - 1] for s in e0b.SEEDS]) for att in e0b.ATTACKS}
    out.append(f"FedAvg at R (same seeds), %: { {k: round(v, 2) for k, v in fedavg.items()} }")

    # stability of FLTrust after R: 5-round trailing means and round-to-round
    # variation over rounds 51-180 (the first 50 are the steep part of the curve)
    rows = []
    for arm, src in (("parity", parity), ("native", native)):
        for (a, attack, rule, root, seed), d in src.items():
            if rule != "fltrust":
                continue
            c = 100 * np.array(d["accuracy"])
            rows.append({"arm": arm, "attack": attack, "root": root, "seed": seed,
                         "S110": c[R - 5:R].mean(), "S150": c[145:150].mean(), "S180": c[175:180].mean(),
                         "mean_abs_round_change": np.abs(np.diff(c[50:])).mean(), "s_per_round": d["seconds"] / len(c)})
    st = pd.DataFrame(rows)
    st.to_csv(f"{OUT}/cifar_fltrust_stability.csv", index=False)
    summ = st.groupby(["arm", "attack", "root"])[["S110", "S150", "S180", "mean_abs_round_change", "s_per_round"]].mean()
    summ["gain_S110_S180"] = summ.S180 - summ.S110
    out += ["", "FLTrust stability (seed means; trailing 5-round means; rounds 51-180 for the change):", summ.round(2).to_string()]


def budget(res, out):
    costs_path = f"{E0B}/costs.json"
    costs = json.load(open(costs_path))
    out += ["", f"## Sec. 7 Budget ({e0b.BUDGET_H:.0f} h; measured costs, {costs_path})"]
    plan = e0b.budget(R, True, costs_path, out)
    s = costs["s_round"]
    h = 1 / 3600 / e0b.PROCS
    runs = 2 * 21 * e0b.N_SEEDS                # FLTrust native at roots 100 and 2500
    add_fixed = runs * R * s["fixed"] * h
    add_flt = runs * R * s["fltrust"] * h
    out.append(f"E1 factorial: {runs} more runs (FLTrust native x 2 roots x 21 cells x 10 seeds): "
               f"+{add_fixed:.2f} h at s_fixed, +{add_flt:.2f} h at s_fltrust")
    factorial = {}
    for lvl, desc, k in e0b.LEVELS:
        c = e0b.cost(R, k, costs)
        factorial[lvl] = {"without": c, "with_s_fixed": c + add_fixed, "with_s_fltrust": c + add_flt}
        out.append(f"level {lvl} (k = {k:2d}): without {c:6.2f} h; with factorial {c + add_fixed:6.2f} h (s_fixed), "
                   f"{c + add_flt:6.2f} h (s_fltrust); margin {e0b.BUDGET_H - c - add_fixed:+.2f} h / {e0b.BUDGET_H - c - add_flt:+.2f} h")
    res["budget"] = {**plan, "factorial_h": factorial, "factorial_runs": runs}


def main():
    os.makedirs(OUT, exist_ok=True)
    out = ["P1 step-parity analysis (results/p1_parity/PLAN.md). MNIST seeds 42-51; CIFAR-10 seeds 0-2.", ""]
    res = {}
    fx = load_parity_fixed()
    sy_raw = load_parity_systems()
    blockc = {}
    for p in glob.glob(f"{RAW}/cifar_blockC/raw/*.json"):
        blockc[p] = json.load(open(p))
    ok = completeness(fx, sy_raw, blockc, out)
    if not ok:
        print("\n".join(out))
        sys.exit(1)
    nat = load_seeded()
    sy = add_published_adaaggrl(sy_raw)
    res["regression_ok"] = regression(fx, nat, out)
    e1(fx, nat, res, out)
    e2(fx, res, out)
    e3(fx, sy, res, out)
    tables(fx, sy, out)
    cifar(res, out)
    budget(res, out)
    text = "\n".join(out)
    with open(f"{OUT}/analysis.txt", "w") as f:
        f.write(text + "\n")
    with open(f"{OUT}/analysis.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(text)


if __name__ == "__main__":
    main()
