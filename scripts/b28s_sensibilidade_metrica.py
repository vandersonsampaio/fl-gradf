"""
B2.8s (results/b28s_sensibilidade_metrica/PLANO.md): o veredito da granularidade
do B2.8 depende da métrica (média uniforme × média ponderada pelo tamanho do test set)?

Um job por célula × semente roda 9 sistemas, com `_reseed(seed)` antes de cada um, e
grava as acurácias por cliente do modelo final (rodada 15):
  oracle_u, oracle_w   oráculo guloso por rodada (B2.8) escolhendo pela métrica uniforme / ponderada
  fixed, sr_only, td3_ref   esqueleto (learner da ablação)
  fltrust, krum, median, trimmed_mean   regras estáticas como no exp9

Uso (CPU):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.b28s_sensibilidade_metrica celula --alpha 0.1 --attack sign_flipping --seed 42
  venv/bin/python -m scripts.b28s_sensibilidade_metrica analisar
"""

import argparse
import glob
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/b28s_sensibilidade_metrica"
RAW = f"{OUT}/raw"
SEEDS = list(range(42, 52))
N_ROUNDS = 15
BYZ = [0, 1]
EXCLUDED = {(0.05, "fltrust_aligned"), (0.1, "fltrust_aligned")}
SKELETON = ["fixed", "sr_only", "td3_ref"]
STATIC = ["fltrust", "krum", "median", "trimmed_mean"]
SYSTEMS = ["oracle_u", "oracle_w"] + SKELETON + STATIC


def _metrics(per_acc, participants):
    accs = np.array([per_acc[p.id] for p in participants])
    n = np.array([len(p.y_test) for p in participants], dtype=float)
    return float(accs.mean()), float((accs * n).sum() / n.sum()), accs, n


def run_cell(alpha, attack, seed, out_dir=RAW):
    from scripts.b27_horizonte import INPUT_SHAPE, _reseed
    from scripts.b28_oraculo_por_regra import build_learner_cls as build_oracle
    from scripts.frente1_ablacao_adaaggrl import build_learner_cls as build_ablation
    from src.experiments.exp9_dominance_grid import INFORMED_ATTACKS, _split_name
    from src.fl.attacked_learner import AttackedFederatedLearner, InformedAttackedFederatedLearner
    from src.utils.data_loader import load_dataset_participants

    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/celula_{attack}_a{alpha}_seed{seed}.csv"
    if os.path.exists(path):
        print(f"já existe: {path}", flush=True)
        return path
    Oracle, _ = build_oracle()
    Ablation = build_ablation()
    parts, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    n_test = np.array([len(p.y_test) for p in parts], dtype=float)

    class OracleW(Oracle):  # escolhe pela métrica ponderada
        def _eval(self, participants):
            accs = np.array([self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants])
            n = np.array([len(p.y_test) for p in participants], dtype=float)
            return float((accs * n).sum() / n.sum())

    common = dict(n_rounds=N_ROUNDS, n_classes=10, attack_type=attack, byzantine_ids=BYZ, seed=seed)
    rows = []
    for system in SYSTEMS:
        t0 = time.time()
        _reseed(seed)
        if system == "oracle_u":
            lr = Oracle(**common)
        elif system == "oracle_w":
            lr = OracleW(**common)
        elif system in SKELETON:
            lr = Ablation(input_shape=INPUT_SHAPE, dataset="mnist",
                          mode="td3" if system == "td3_ref" else system, **common)
        else:
            cls = InformedAttackedFederatedLearner if attack in INFORMED_ATTACKS else AttackedFederatedLearner
            lr = cls(aggregation=system, **common)
        res = lr.train(parts, root_data=root, verbose=False)
        u, w, accs, _ = _metrics(res[-1].per_participant_accuracy, parts)
        row = {"alpha": alpha, "attack_type": attack, "seed": seed, "system": system,
               "acc_uniforme": u, "acc_ponderada": w, "global_accuracy": float(res[-1].global_accuracy),
               "escolhas": "|".join(getattr(lr, "choices", [])), "sec": time.time() - t0}
        row.update({f"acc_c{i}": float(a) for i, a in enumerate(accs)})
        row.update({f"n_test_c{i}": int(n) for i, n in enumerate(n_test)})
        rows.append(row)
        print(f"alpha={alpha} attack={attack} seed={seed} {system} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    print(f"FIM {path}", flush=True)
    return path


def _ci95(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def _veredito(df, metric, oracle):
    rows = []
    for (alpha, attack), g in df.groupby(["alpha", "attack_type"]):
        if (alpha, attack) in EXCLUDED:
            continue
        w = g.pivot_table(index="seed", columns="system", values=metric)
        best_sk = w[SKELETON].mean().idxmax()
        best_fx = w[STATIC].mean().idxmax()
        q, qlo, qhi = _ci95(w[oracle] - w[best_sk])
        ww, wlo, whi = _ci95(w[best_sk] - w[best_fx])
        gg, glo, ghi = _ci95(w[oracle] - w[best_fx])
        esc = Counter("|".join(g[g.system == oracle].escolhas.fillna("")).split("|"))
        rows.append({"alpha": alpha, "attack_type": attack, "melhor_esqueleto": best_sk, "melhor_estatica": best_fx,
                     "Q_pp": 100 * q, "Q_lo": 100 * qlo, "Q_hi": 100 * qhi, "W_pp": 100 * ww, "W_lo": 100 * wlo,
                     "W_hi": 100 * whi, "G_pp": 100 * gg, "G_lo": 100 * glo, "G_hi": 100 * ghi,
                     "escolhas_oraculo": ", ".join(f"{k}:{v}" for k, v in esc.most_common() if k)})
    r = pd.DataFrame(rows)
    alvo = r[r.W_lo > 0].copy()
    alvo["sinal_Q"] = np.where(alvo.Q_hi < 0, "abaixo", np.where(alvo.Q_lo > 0, "acima", "inconclusivo"))
    half = len(alvo) / 2
    below, above = (alvo.sinal_Q == "abaixo").sum(), (alvo.sinal_Q == "acima").sum()
    if len(alvo) == 0:
        v = "SEM CÉLULAS-ALVO"
    elif below >= half:
        v = "A GRANULARIDADE EXPLICA"
    elif above >= half:
        v = "A GRANULARIDADE NÃO EXPLICA"
    else:
        v = "INCONCLUSIVO"
    return r, alvo, v


def analisar():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RAW}/celula_*.csv"))])
    df.to_csv(f"{OUT}/grade_raw.csv", index=False)
    pd.set_option("display.width", 250)
    ru, au, vu = _veredito(df, "acc_uniforme", "oracle_u")
    rw, aw, vw = _veredito(df, "acc_ponderada", "oracle_w")
    ru.to_csv(f"{OUT}/celulas_uniforme.csv", index=False)
    rw.to_csv(f"{OUT}/celulas_ponderada.csv", index=False)
    comuns = au.merge(aw, on=["alpha", "attack_type"], suffixes=("_u", "_w"))
    conc = float((comuns.sinal_Q_u == comuns.sinal_Q_w).mean()) if len(comuns) else float("nan")
    if vu != vw:
        crit = "DEPENDE DA MÉTRICA (veredito muda)"
    elif len(comuns) and conc >= 0.75:
        crit = "GRANULARIDADE BLINDADA (veredito igual e ≥ 75% de concordância de sinal nas células-alvo comuns)"
    else:
        crit = "BLINDADA NO VEREDITO, SENSÍVEL POR CÉLULA"
    cols = ["alpha", "attack_type", "melhor_esqueleto", "melhor_estatica", "Q_pp", "Q_lo", "Q_hi", "W_pp", "W_lo", "G_pp",
            "escolhas_oraculo"]
    b28 = pd.read_csv("results/b28_oraculo_por_regra/oraculo_por_celula.csv")
    lines = [f"B2.8s — sensibilidade da métrica; {len(df)} linhas (esperado {9 * 21 * 10})", "",
             "== Métrica UNIFORME (oracle_u) ==", ru[cols].round(3).to_string(index=False),
             f"células-alvo: {len(au)}; abaixo {int((au.sinal_Q == 'abaixo').sum())}, acima {int((au.sinal_Q == 'acima').sum())} "
             f"-> VEREDITO: {vu}", "",
             "== Métrica PONDERADA por tamanho do test set (oracle_w) ==", rw[cols].round(3).to_string(index=False),
             f"células-alvo: {len(aw)}; abaixo {int((aw.sinal_Q == 'abaixo').sum())}, acima {int((aw.sinal_Q == 'acima').sum())} "
             f"-> VEREDITO: {vw}", "",
             f"Células-alvo comuns: {len(comuns)}; concordância do sinal de Q: {conc:.0%}",
             "Diferenças: " + "; ".join(f"{r.attack_type} α={r.alpha}: {r.sinal_Q_u}→{r.sinal_Q_w}"
                                        for r in comuns.itertuples() if r.sinal_Q_u != r.sinal_Q_w), "",
             f"CRITÉRIO DA SENSIBILIDADE (PLANO §3): {crit}", "",
             f"(Descritivo) veredito do B2.8 original: 8/14 células-alvo abaixo -> 'a granularidade explica'; "
             f"nesta grade, métrica uniforme: {vu}"]
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
    c.add_argument("--out_dir", default=RAW)
    sub.add_parser("analisar")
    a = ap.parse_args()
    if a.cmd == "celula":
        run_cell(a.alpha, a.attack, a.seed, a.out_dir)
    else:
        analisar()


if __name__ == "__main__":
    main()
