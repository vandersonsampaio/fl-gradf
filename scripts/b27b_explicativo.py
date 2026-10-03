"""
B2.7b explicativo (results/b27b_td3_constante/PLANO.md + ADENDO2.md).

Pergunta (revista após o B2.7a, que mostrou política independente do estado):
de onde vem o Δ(td3_ref − fixed) em H = 150, de uma constante deslocada ou do
ruído de exploração?

Sistemas novos (3 células × sementes 42-51, H = 150, learner da ablação):
  td3_frozen     o mesmo agente TD3, mesmo aquecimento e mesmo processo de ruído,
                 SEM aprendizado do ator: train_step roda normalmente (crítico,
                 buffer e RNG consumidos igual ao td3_ref) e o ator (W, b e alvos)
                 é restaurado ao inicial logo depois. A atualização do ator não
                 consome RNG, então a sequência de ruído é a mesma do td3_ref.
  fixed_td3mean  ação constante, determinística = média por célula de a_TD3
                 (π150 nos estados 101-150, a_td3.csv do B2.7a)
  fixed_b025     a = [0,5]*4, b = 0,25
  fixed_b075     a = [0,5]*4, b = 0,75
As constantes passam pelo modo td3 com select_action → constante e train_step →
no-op; o RNG do agente é próprio e não toca o np.random global, então isso é
equivalente ao modo fixed (verificado: constante [0,5]*5 reproduz o `fixed`).
td3_ref e fixed (centro) vêm do B2.7 (results/b27_horizonte/grade_raw.csv).

Uso (CPU):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.b27b_explicativo run --alpha 0.05 --attack label_flipping --seed 42 --system td3_frozen
  venv/bin/python -m scripts.b27b_explicativo analisar
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/b27b_td3_constante"
RAW = f"{OUT}/b27b_raw"
CELLS = [(0.05, "label_flipping"), (0.1, "label_flipping"), (0.05, "low_mag_backdoor")]
SEEDS = list(range(42, 52))
SYSTEMS = ["td3_frozen", "fixed_td3mean", "fixed_b025", "fixed_b075"]
HORIZONS = [15, 50, 150]
H_MAX = 150
B27_RAW = "results/b27_horizonte/grade_raw.csv"
MARGIN_PP = 1.0  # TOST descritivo


def td3mean_action(alpha, attack):
    a = pd.read_csv(f"{OUT}/a_td3.csv")
    a = a[(a.alpha == alpha) & (a.attack_type == attack)]
    return np.array([a[f"a_td3_{i}"].mean() for i in range(5)])


def constant_for(system, alpha, attack):
    if system == "fixed_td3mean":
        return td3mean_action(alpha, attack)
    if system == "fixed_b025":
        return np.array([0.5, 0.5, 0.5, 0.5, 0.25])
    if system == "fixed_b075":
        return np.array([0.5, 0.5, 0.5, 0.5, 0.75])
    if system == "fixed_center_check":
        return np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    return None


def run(alpha, attack, seed, system, n_rounds=H_MAX, out_dir=RAW):
    from scripts.b27_horizonte import BYZ, INPUT_SHAPE, _reseed
    from scripts.frente1_ablacao_adaaggrl import build_learner_cls
    from src.experiments.exp9_dominance_grid import _split_name
    from src.utils.data_loader import load_dataset_participants

    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{system}_{attack}_a{alpha}_seed{seed}_R{n_rounds}.csv"
    if os.path.exists(path):
        print(f"já existe: {path}", flush=True)
        return path
    Ablation = build_learner_cls()
    participants, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    _reseed(seed)
    learner = Ablation(input_shape=INPUT_SHAPE, dataset="mnist", mode="td3", n_rounds=n_rounds, n_classes=10,
                       attack_type=attack, byzantine_ids=BYZ, seed=seed)
    agent = learner.agent
    const = constant_for(system, alpha, attack)
    if const is not None:
        agent.select_action = lambda state, *a, **k: const.copy()
        agent.train_step = lambda *a, **k: None
    elif system == "td3_frozen":
        init = {k: getattr(agent, k).copy() for k in ("W_actor", "b_actor", "W_actor_target", "b_actor_target")}
        orig_train = agent.train_step

        def frozen_train(*a, **k):
            orig_train(*a, **k)
            for k_, v in init.items():  # lr do ator = 0: restaura ator e alvo do ator
                setattr(agent, k_, v.copy())
        agent.train_step = frozen_train
    else:
        raise ValueError(system)
    t0 = time.time()
    res = learner.train(participants, root_data=root, verbose=False)
    rows = [{"alpha": alpha, "attack_type": attack, "seed": seed, "system": system, "H": H,
             "accuracy": float(res[H - 1].global_accuracy), "sec": time.time() - t0}
            for H in HORIZONS if len(res) >= H]
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    print(f"FIM {system} {attack} a{alpha} seed{seed} ({time.time() - t0:.0f}s)", flush=True)
    return path


def _ci(d):
    n = len(d)
    m = d.mean()
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def _tost(d, margin):
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(m) < margin else 1.0
    return max(1 - stats.t.cdf((m + margin) / se, n - 1), stats.t.cdf((m - margin) / se, n - 1))


def analisar():
    new = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RAW}/*_R{H_MAX}.csv"))])
    old = pd.read_csv(B27_RAW)
    old = old[old.system.isin(["td3_ref", "fixed"])]
    cells = pd.DataFrame(CELLS, columns=["alpha", "attack_type"])
    old = old.merge(cells, on=["alpha", "attack_type"])
    allr = pd.concat([old[new.columns.intersection(old.columns)], new])
    allr = allr[allr.H == H_MAX]
    allr.to_csv(f"{OUT}/b27b_grade_raw.csv", index=False)

    comps = [("td3_ref", "td3_frozen"), ("td3_frozen", "fixed"), ("td3_ref", "fixed_td3mean"),
             ("fixed_td3mean", "fixed"), ("td3_ref", "fixed"), ("td3_ref", "fixed_b025"),
             ("td3_ref", "fixed_b075")]
    rows = []
    for (alpha, attack), g in allr.groupby(["alpha", "attack_type"]):
        w = g.pivot_table(index="seed", columns="system", values="accuracy")
        for a, b in comps:
            if a not in w or b not in w:
                continue
            d = (w[a] - w[b]).dropna().to_numpy()
            m, lo, hi = _ci(d)
            rows.append({"alpha": alpha, "attack_type": attack, "comparacao": f"{a} − {b}", "n": len(d),
                         "delta_pp": 100 * m, "ic95_lo_pp": 100 * lo, "ic95_hi_pp": 100 * hi,
                         "p_tost_1pp": _tost(d, MARGIN_PP / 100)})
        consts = [c for c in ["fixed", "fixed_b025", "fixed_b075", "fixed_td3mean"] if c in w]
        best = w[consts].mean().idxmax()
        d = (w["td3_ref"] - w[best]).dropna().to_numpy()
        m, lo, hi = _ci(d)
        rows.append({"alpha": alpha, "attack_type": attack, "comparacao": f"td3_ref − melhor constante ({best})",
                     "n": len(d), "delta_pp": 100 * m, "ic95_lo_pp": 100 * lo, "ic95_hi_pp": 100 * hi,
                     "p_tost_1pp": _tost(d, MARGIN_PP / 100)})
    r = pd.DataFrame(rows)
    r.to_csv(f"{OUT}/b27b_deltas.csv", index=False)

    def get(cmp_):
        return r[r.comparacao == cmp_].set_index(["alpha", "attack_type"])

    rf, ff, rm = get("td3_ref − td3_frozen"), get("td3_frozen − fixed"), get("td3_ref − fixed_td3mean")
    c1 = bool(((rf.ic95_lo_pp <= 0) & (rf.ic95_hi_pp >= 0)).all()) and len(rf) == 3
    c2 = int(((ff.delta_pp > 0) & (ff.ic95_lo_pp > 0)).sum())
    expl = c1 and c2 >= 1
    desl = bool(((rm.ic95_lo_pp <= 0) & (rm.ic95_hi_pp >= 0)).all()) and len(rm) == 3
    best_rows = r[r.comparacao.str.startswith("td3_ref − melhor")]
    old_crit = int(((best_rows.delta_pp > 0) & (best_rows.ic95_lo_pp > 0)).sum())

    pd.set_option("display.width", 220)
    lines = ["B2.7b explicativo — H = 150, sementes 42–51, 3 células (exploratório; B2.7a já visto)",
             f"runs novos: {len(new[new.H == H_MAX])} (esperado 120)", "",
             "Médias por sistema (acurácia em H = 150):",
             allr.pivot_table(index=["alpha", "attack_type"], columns="system", values="accuracy").round(4).to_string(), "",
             "Δ pareados por semente (p.p., IC95 t; TOST ±1 p.p. descritivo):",
             r.round(3).to_string(index=False), "",
             f"Critério 1 — 'o ganho é exploração, não aprendizado': IC95 de (td3_ref − td3_frozen) contém 0 nas 3 células "
             f"[{'sim' if c1 else 'não'}] E (td3_frozen − fixed) > 0 com IC95 > 0 em ≥ 1 célula [{c2}/3] -> "
             + ("ATENDIDO" if expl else "NÃO atendido"),
             "Critério 2 — 'constante deslocada' (fixed_td3mean ≈ td3_ref): IC95 de (td3_ref − fixed_td3mean) contém 0 nas 3 "
             "células -> " + ("ATENDIDO" if desl else "NÃO atendido"),
             f"Critério original do PLANO (secundário): td3_ref > melhor constante (IC95 > 0) em {old_crit}/3 células -> "
             + ("atendido" if old_crit >= 2 else "não atendido")]
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/b27b_analise.txt", "w").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--alpha", type=float, required=True)
    r.add_argument("--attack", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--system", required=True, choices=SYSTEMS + ["fixed_center_check"])
    r.add_argument("--n_rounds", type=int, default=H_MAX)
    r.add_argument("--out_dir", default=RAW)
    sub.add_parser("analisar")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.alpha, a.attack, a.seed, a.system, a.n_rounds, a.out_dir)
    else:
        analisar()


if __name__ == "__main__":
    main()
