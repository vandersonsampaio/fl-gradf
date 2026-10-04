"""
C0b — sensibilidade POST HOC (declarada em results/c0b_espaco_h150/VERIFICACOES.md §3):
tetos com ponderação UNIFORME em vez de por tamanho de amostra.

Motivo: o `FedAvgStrategy` do framework pondera por tamanho de amostra; com α ≤ 0,1
os tamanhos variam de ~10² a ~10⁴ por cliente e a métrica do C0/C0b é a média
UNIFORME das acurácias nos test sets dos 10 clientes. O oráculo FedAvg-8 por amostra
subestima o teto (~4,5 p.p. em α 0,1). Aqui:
  teto_oraculo8_uniforme  FedAvg só nos 8 honestos, pesos iguais
  teto_fedavg10_uniforme  FedAvg nos 10, sem ataque, pesos iguais
Mesmo código do run_ceiling do B2.7/C0b (ressemeadura igual), só a ponderação muda
(sample_sizes ignorado no FedAvgStrategy, neste processo). Sementes 72-81, α {0,05; 0,1; 0,5},
H 15/50/150. Depois reaplica o critério do C0b (gap > 2 p.p., IC95 > 0) contra esses tetos.

Uso: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.c0b_sensibilidade_oraculo_uniforme teto --alpha 0.1 --seed 72
     venv/bin/python -m scripts.c0b_sensibilidade_oraculo_uniforme analisar
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

OUT = "results/c0b_espaco_h150/sensibilidade_oraculo_uniforme"
HORIZONS = [15, 50, 150]
THRESH_PP = 2.0


def run_teto(alpha, seed, n_rounds=150):
    from scripts.b27_horizonte import BYZ, _reseed
    from src.experiments.exp9_dominance_grid import _split_name
    from src.fl import federated_learner as FL
    from src.fl.attacked_learner import AttackedFederatedLearner
    from src.utils.data_loader import load_dataset_participants

    os.makedirs(f"{OUT}/raw", exist_ok=True)
    path = f"{OUT}/raw/teto_uniforme_a{alpha}_seed{seed}_R{n_rounds}.csv"
    if os.path.exists(path):
        return path
    orig = FL.FedAvgStrategy.aggregate
    FL.FedAvgStrategy.aggregate = lambda self, updates, sample_sizes=None, **k: orig(self, updates, None, **k)
    parts, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    rows = []
    for name, clients in [("teto_fedavg10_uniforme", parts),
                          ("teto_oraculo8_uniforme", [p for i, p in enumerate(parts) if i not in BYZ])]:
        acc = {}

        class _Ceil(AttackedFederatedLearner):
            def _run_round(self, round_num, ps, root_data):
                rr = super()._run_round(round_num, ps, root_data)
                if round_num in HORIZONS:  # round_num = nº de rodadas concluídas
                    m = self._make_model(parts[0].n_features)
                    m.set_params(self._global_params)
                    acc[round_num] = float(np.mean([m.accuracy(p.X_test, p.y_test) for p in parts]))
                return rr

        _reseed(seed)
        _Ceil(n_rounds=n_rounds, n_classes=10, attack_type="none", byzantine_ids=[], seed=seed,
              aggregation="fedavg").train(clients, root_data=root, verbose=False)
        rows += [{"alpha": alpha, "seed": seed, "system": name, "H": H, "accuracy": a} for H, a in acc.items()]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"FIM {path}", flush=True)
    return path


def _ci95(x):
    from scipy import stats
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def analisar():
    raw = pd.read_csv("results/c0b_espaco_h150/grade_raw.csv")
    tet = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/raw/teto_uniforme_*_R150.csv"))])
    tet_old = pd.read_csv("results/c0b_espaco_h150/tetos_raw.csv")
    rows = []
    for (alpha, attack), g in raw.groupby(["alpha", "attack_type"]):
        for H, gh in g.groupby("H"):
            w = gh.pivot_table(index="seed", columns="system", values="accuracy")
            best = w.mean().idxmax()
            t = tet[(tet.alpha == alpha) & (tet.H == H)].pivot_table(index="seed", columns="system", values="accuracy")
            row = {"alpha": alpha, "attack_type": attack, "H": H, "melhor_existente": best, "acc_melhor": w[best].mean()}
            for tname in ["teto_oraculo8_uniforme", "teto_fedavg10_uniforme"]:
                for ref, rl in [(best, "global"), ("sr_only", "esqueleto_ref")]:
                    m, lo, hi = _ci95((t[tname] - w[ref]).dropna())
                    row.update({f"gap_{rl}_{tname}_pp": 100 * m, f"gap_{rl}_{tname}_lo": 100 * lo,
                                f"gap_{rl}_{tname}_hi": 100 * hi})
            row["espaco_global_or8u"] = bool(row["gap_global_teto_oraculo8_uniforme_pp"] > THRESH_PP
                                             and row["gap_global_teto_oraculo8_uniforme_lo"] > 0)
            row["espaco_esqref_or8u"] = bool(row["gap_esqueleto_ref_teto_oraculo8_uniforme_pp"] > THRESH_PP
                                             and row["gap_esqueleto_ref_teto_oraculo8_uniforme_lo"] > 0)
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/espaco_restante_oraculo_uniforme.csv", index=False)
    pd.set_option("display.width", 250)
    tab = pd.concat([tet, tet_old]).groupby(["alpha", "system", "H"])["accuracy"].mean().unstack("H")
    lines = ["C0b — sensibilidade POST HOC: tetos com ponderação uniforme (VERIFICACOES.md §3)", "",
             "Tetos por α × H (média de 10 sementes):", (100 * tab).round(2).to_string(), ""]
    for H in HORIZONS:
        r = res[res.H == H]
        lines.append(f"H={H:3d}: células com espaço contra o MELHOR EXISTENTE (oráculo-8 UNIFORME, gap > {THRESH_PP} p.p., "
                     f"IC95 > 0) = {int(r.espaco_global_or8u.sum())}/{len(r)}; contra o esqueleto de referência = "
                     f"{int(r.espaco_esqref_or8u.sum())}/{len(r)}")
    cols = ["alpha", "attack_type", "melhor_existente", "acc_melhor", "gap_global_teto_oraculo8_uniforme_pp",
            "gap_global_teto_oraculo8_uniforme_lo", "gap_global_teto_oraculo8_uniforme_hi", "espaco_global_or8u",
            "gap_global_teto_fedavg10_uniforme_pp", "gap_esqueleto_ref_teto_oraculo8_uniforme_pp", "espaco_esqref_or8u"]
    lines += ["", "Mapa em H = 150:", res[res.H == 150][cols].round(3).to_string(index=False)]
    text = "\n".join(lines) + "\n"
    open(f"{OUT}/analise.txt", "w").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teto")
    t.add_argument("--alpha", type=float, required=True)
    t.add_argument("--seed", type=int, required=True)
    sub.add_parser("analisar")
    a = ap.parse_args()
    if a.cmd == "teto":
        run_teto(a.alpha, a.seed)
    else:
        analisar()


if __name__ == "__main__":
    main()
