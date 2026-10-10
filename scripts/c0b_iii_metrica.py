"""
C0b-iii (results/c0b_iii_metrica/PLANO.md): metric sensitivity of C0b, α ≤ 0.1.
Same systems, regime and reseeding as C0b, saving the per-client accuracies at
H = 15/50/150 (uniform metric and metric weighted by test-set size), plus the
FedAvg-10 and oracle-8 ceilings with uniform and sample-size weighting.

Usage (CPU):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.c0b_iii_metrica celula --alpha 0.1 --attack sign_flipping --seed 72
  ... -m scripts.c0b_iii_metrica teto --alpha 0.1 --seed 72
  ... -m scripts.c0b_iii_metrica analisar
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/c0b_iii_metrica"
RAW = f"{OUT}/raw"
SEEDS = list(range(72, 82))
ALPHAS = [0.05, 0.1]
ATTACKS = ["trim_attack", "krum_collusion", "low_mag_backdoor", "sign_flipping", "gaussian_noise", "label_flipping"]
VARIANTS = ["sr_only", "sr_bin", "sr_b025", "cosserver_only"]
RULES = ["fltrust", "trimmed_mean", "clustering", "median"]
HORIZONS = [15, 50, 150]
H_MAX = 150
THRESH_PP = 2.0


def _rows(base, results, parts):
    n = np.array([len(p.y_test) for p in parts], dtype=float)
    out = []
    for H in HORIZONS:
        if len(results) >= H:
            pa = results[H - 1].per_participant_accuracy
            accs = np.array([pa[p.id] for p in parts])
            out.append({**base, "H": H, "acc_uniforme": float(accs.mean()),
                        "acc_ponderada": float((accs * n).sum() / n.sum()),
                        "global_accuracy": float(results[H - 1].global_accuracy),
                        **{f"acc_c{i}": float(a) for i, a in enumerate(accs)},
                        **{f"n_test_c{i}": int(x) for i, x in enumerate(n)}})
    return out


def run_cell(alpha, attack, seed):
    from scripts.b26_decomposicao import build_learner_cls
    from scripts.b27_horizonte import BYZ, INPUT_SHAPE, _reseed, build_plain_rule_learner
    from src.experiments.exp9_dominance_grid import _split_name
    from src.utils.data_loader import load_dataset_participants

    os.makedirs(RAW, exist_ok=True)
    path = f"{RAW}/celula_{attack}_a{alpha}_seed{seed}.csv"
    if os.path.exists(path):
        return path
    Variant = build_learner_cls()
    PlainRule, _ = build_plain_rule_learner()
    parts, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    common = dict(n_rounds=H_MAX, n_classes=10, attack_type=attack, byzantine_ids=BYZ, seed=seed)
    rows = []
    for system in VARIANTS + RULES:
        t0 = time.time()
        _reseed(seed)
        lr = (Variant(input_shape=INPUT_SHAPE, dataset="mnist", variant=system, **common) if system in VARIANTS
              else PlainRule(rule=system, **common))
        res = lr.train(parts, root_data=root, verbose=False)
        rows += _rows({"alpha": alpha, "attack_type": attack, "seed": seed, "system": system}, res, parts)
        print(f"alpha={alpha} attack={attack} seed={seed} {system} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    print(f"END {path}", flush=True)
    return path


def run_teto(alpha, seed):
    from scripts.b27_horizonte import BYZ, _reseed
    from src.experiments.exp9_dominance_grid import _split_name
    from src.fl import federated_learner as FL
    from src.fl.attacked_learner import AttackedFederatedLearner
    from src.utils.data_loader import load_dataset_participants

    os.makedirs(RAW, exist_ok=True)
    path = f"{RAW}/teto_a{alpha}_seed{seed}.csv"
    if os.path.exists(path):
        return path
    parts, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    orig = FL.FedAvgStrategy.aggregate
    uniform = lambda self, updates, sample_sizes=None, **k: orig(self, updates, None, **k)  # noqa: E731
    rows = []
    for pond, agg in [("amostra", orig), ("uniforme", uniform)]:
        FL.FedAvgStrategy.aggregate = agg
        for name, clients in [("teto_fedavg10", parts), ("teto_oraculo8", [p for i, p in enumerate(parts) if i not in BYZ])]:
            accs_h = {}

            class _Ceil(AttackedFederatedLearner):
                def _run_round(self, round_num, ps, root_data):
                    rr = super()._run_round(round_num, ps, root_data)
                    if round_num in HORIZONS:  # evaluates on the 10 clients' test sets
                        m = self._make_model(parts[0].n_features)
                        m.set_params(self._global_params)
                        accs_h[round_num] = {p.id: m.accuracy(p.X_test, p.y_test) for p in parts}
                    return rr

            _reseed(seed)
            byz = BYZ if name == "teto_fedavg10" else []
            _Ceil(n_rounds=H_MAX, n_classes=10, attack_type="none", byzantine_ids=byz, seed=seed,
                  aggregation="fedavg").train(clients, root_data=root, verbose=False)

            class _R:  # adapter for _rows
                def __init__(self, pa):
                    self.per_participant_accuracy, self.global_accuracy = pa, float(np.mean(list(pa.values())))
            fake = [None] * H_MAX
            for H, pa in accs_h.items():
                fake[H - 1] = _R(pa)
            rows += _rows({"alpha": alpha, "seed": seed, "system": f"{name}_{pond}"}, fake, parts)
    FL.FedAvgStrategy.aggregate = orig
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    print(f"END {path}", flush=True)
    return path


def _ci95(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def analisar():
    cel = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RAW}/celula_*.csv"))])
    tet = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RAW}/teto_*.csv"))])
    cel.to_csv(f"{OUT}/grade_raw.csv", index=False)
    tet.to_csv(f"{OUT}/tetos_raw.csv", index=False)
    lines = ["C0b-iii — metric sensitivity of C0b (α ≤ 0.1; seeds 72–81)",
             f"cell rows: {len(cel)} (expected {12 * 10 * 8 * 3}); ceiling rows: {len(tet)} (expected {2 * 10 * 4 * 3})", ""]
    # reproduction checks
    c0b = pd.read_csv("results/c0b_espaco_h150/grade_raw.csv")
    m = cel.merge(c0b, on=["alpha", "attack_type", "seed", "system", "H"])
    dmax = float((m.acc_uniforme - m.accuracy).abs().max()) if len(m) else float("nan")
    t_old = pd.read_csv("results/c0b_espaco_h150/tetos_raw.csv")
    t_old = t_old[t_old.system == "teto_oraculo_fedavg8"]
    mt = tet[tet.system == "teto_oraculo8_amostra"].merge(t_old, on=["alpha", "seed", "H"])
    dt = float((mt.acc_uniforme - mt.accuracy).abs().max()) if len(mt) else float("nan")
    t_u = pd.concat([pd.read_csv(f) for f in glob.glob("results/c0b_espaco_h150/sensibilidade_oraculo_uniforme/raw/*.csv")])
    t_u = t_u[(t_u.system == "teto_oraculo8_uniforme") & t_u.alpha.isin(ALPHAS)]
    mu = tet[tet.system == "teto_oraculo8_uniforme"].merge(t_u, on=["alpha", "seed", "H"])
    du = float((mu.acc_uniforme - mu.accuracy).abs().max()) if len(mu) else float("nan")
    lines += [f"Reproduction: uniform metric × C0b grade_raw max |Δ| = {dmax:.2e}; sample-weighted oracle-8 × C0b = {dt:.2e}; "
              f"uniform oracle-8 × sensitivity = {du:.2e}", ""]

    pares = {"U": ("acc_uniforme", "teto_oraculo8_uniforme"), "P": ("acc_ponderada", "teto_oraculo8_amostra")}
    res = []
    for par, (metric, teto) in pares.items():
        for (alpha, attack), g in cel.groupby(["alpha", "attack_type"]):
            for H, gh in g.groupby("H"):
                w = gh.pivot_table(index="seed", columns="system", values=metric)
                bv, br, best = w[VARIANTS].mean().idxmax(), w[RULES].mean().idxmax(), w.mean().idxmax()
                d, dlo, dhi = _ci95(w[bv] - w[br])
                t = tet[(tet.alpha == alpha) & (tet.H == H) & (tet.system == teto)].set_index("seed")[metric]
                gp, glo, ghi = _ci95((t - w[best]).dropna())
                res.append({"par": par, "alpha": alpha, "attack_type": attack, "H": H, "melhor_variante": bv,
                            "melhor_regra": br, "melhor_existente": best, "D_pp": 100 * d, "D_lo": 100 * dlo,
                            "D_hi": 100 * dhi, "classe": "variante" if dlo > 0 else ("regra" if dhi < 0 else "empate"),
                            "gap_pp": 100 * gp, "gap_lo": 100 * glo, "gap_hi": 100 * ghi,
                            "espaco": bool(100 * gp > THRESH_PP and glo > 0)})
    r = pd.DataFrame(res)
    r.to_csv(f"{OUT}/por_celula.csv", index=False)
    pd.set_option("display.width", 220)
    for H in HORIZONS:
        u = r[(r.par == "U") & (r.H == H)].set_index(["alpha", "attack_type"])
        p = r[(r.par == "P") & (r.H == H)].set_index(["alpha", "attack_type"])
        conc = float((u.classe == p.classe).mean())
        dir_u = int((u.classe == "variante").sum()) - int((u.classe == "regra").sum())
        dir_p = int((p.classe == "variante").sum()) - int((p.classe == "regra").sum())
        same_dir = np.sign(dir_u) == np.sign(dir_p)
        if not same_dir:
            crit = "(iii) DEPENDS ON THE METRIC (majority direction changes)"
        elif conc >= 0.75:
            crit = "(iii) ROBUST"
        else:
            crit = "(iii) ROBUST IN DIRECTION, SENSITIVE PER CELL"
        lines += [f"== H = {H} ==",
                  f"(iii) classes U: variant {int((u.classe == 'variante').sum())}, rule {int((u.classe == 'regra').sum())}, "
                  f"tie {int((u.classe == 'empate').sum())} | P: variant {int((p.classe == 'variante').sum())}, "
                  f"rule {int((p.classe == 'regra').sum())}, tie {int((p.classe == 'empate').sum())}; agreement {conc:.0%}"
                  + (f"  -> {crit}" if H == H_MAX else ""),
                  "    cells that change: " + "; ".join(f"{t} α={a}: {u.loc[(a, t), 'classe']}→{p.loc[(a, t), 'classe']}"
                                                         for (a, t) in u.index if u.loc[(a, t), 'classe'] != p.loc[(a, t), 'classe']),
                  f"(i) cells with headroom: U {int(u.espaco.sum())} [" + "; ".join(f"{t} α={a}" for (a, t) in u.index[u.espaco]) +
                  f"] | P {int(p.espaco.sum())} [" + "; ".join(f"{t} α={a}" for (a, t) in p.index[p.espaco]) + "]", ""]
    cols = ["par", "alpha", "attack_type", "melhor_variante", "melhor_regra", "D_pp", "D_lo", "D_hi", "classe",
            "melhor_existente", "gap_pp", "gap_lo", "espaco"]
    lines += ["Detail at H = 150:", r[r.H == H_MAX][cols].round(3).to_string(index=False)]
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
    {"celula": lambda: run_cell(a.alpha, a.attack, a.seed), "teto": lambda: run_teto(a.alpha, a.seed),
     "analisar": analisar}[a.cmd]()


if __name__ == "__main__":
    main()
