"""
B2.8 (references/roadmap_tese_gradf_v3.md §4.1): teto oracular da seleção
POR RODADA (eixo da granularidade). Exploratório; plano em
results/b28_oraculo_por_regra/PLANO.md.

Pergunta: se uma política escolhesse, a cada rodada, a MELHOR das 7 regras
robustas (oráculo guloso, com acesso ao teste), ela alcançaria a filtragem
por cliente do esqueleto (fixed / sr_only / AdaAggRL)? Se nem o oráculo
alcança, a diferença que o Paper 1 atribuiu a "discreto contra contínuo" é
de GRANULARIDADE (uma regra para todos contra peso por cliente), não de
aprendizado.

Oráculo guloso: a cada rodada, os updates (com ataque) são calculados uma vez;
cada regra do arsenal de 7 (o mesmo do exp10) agrega esses updates; o
candidato com maior acurácia (mesma métrica do C.0: média sobre os X_test dos
10 clientes) vira o novo global. É um LIMITE SUPERIOR da seleção por regra,
não um método implantável (usa o teste para escolher).

Regime idêntico ao C.0/ablação/exp9: MNIST, 10 clientes, bizantinos [0, 1],
15 rodadas, root 100, sementes 42-51, 3 alphas x 7 ataques.

Uso (CPU, baixa prioridade; o framework próprio não usa GPU):
  rodar:    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.b28_oraculo_por_regra run --seeds 42 43
  analisar: venv/bin/python -m scripts.b28_oraculo_por_regra analisar
"""

import argparse
import glob
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/b28_oraculo_por_regra"
ALPHAS = [0.5, 0.1, 0.05]
BYZ = [0, 1]
N_ROUNDS = 15
ABL = "results/frente1_ablacao_adaaggrl/grade_combined_raw.csv"
EXP9 = "results/tables/exp9_dominance_grid_10seeds_ALL_root100_raw.csv"
C0_CEIL = "results/c0_espaco_restante/teto_fedavg_sem_ataque_raw.csv"
SKELETON = ["fixed", "sr_only", "td3_ref"]
EXCLUDED = {(0.05, "fltrust_aligned"), (0.1, "fltrust_aligned")}  # artefato do ataque (C.0 NOTAS §2)


def build_learner_cls():
    from src.experiments.exp10_selector_comparison import FULL_ARSENAL, _make_arsenal_strategy
    from src.fl.attacked_learner import AttackedFederatedLearner, compute_param_updates_auto
    from src.fl.federated_learner import RoundResult

    class GreedyRuleOracleLearner(AttackedFederatedLearner):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("aggregation", "fedavg")  # placeholder; a regra é escolhida por rodada
            super().__init__(*args, **kwargs)
            self.choices = []

        def _eval(self, participants) -> float:
            return float(np.mean([self._make_model(p.n_features).accuracy(p.X_test, p.y_test)
                                  for p in participants]))

        def _run_round(self, round_num, participants, root_data):
            active = self._is_active(round_num)
            updates, is_byz = compute_param_updates_auto(self, participants, active, root_data)
            n_feat = participants[0].n_features
            server_root = root_data or self._carve_root(participants[0])
            server_update = self._make_model(n_feat).fit(server_root["X"], server_root["y"])
            sizes = [p.n_train for p in participants]
            n_byz = sum(1 for b in is_byz if b)

            base = self._global_params.copy()
            best_rule, best_acc, best_params = None, -1.0, None
            for rule in FULL_ARSENAL:
                delta, _ = _make_arsenal_strategy(rule, n_byz).aggregate(
                    updates, sample_sizes=sizes, server_update=server_update)
                self._global_params = base + delta
                acc = self._eval(participants)
                if acc > best_acc:
                    best_rule, best_acc, best_params = rule, acc, self._global_params.copy()
            self._global_params = best_params
            self.choices.append(best_rule)
            per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
            return RoundResult(round_num, float(np.mean(list(per_acc.values()))), per_acc,
                               trust_scores=None, n_accepted=None)

    return GreedyRuleOracleLearner, FULL_ARSENAL


def run(seeds):
    from src.experiments.exp9_dominance_grid import BLIND_ATTACKS, INFORMED_ATTACKS, _split_name
    from src.utils.data_loader import load_dataset_participants

    Learner, arsenal = build_learner_cls()
    os.makedirs(f"{OUT}/raw", exist_ok=True)
    for seed in seeds:
        path = f"{OUT}/raw/seed{seed}.csv"
        if os.path.exists(path):
            print(f"seed {seed}: já existe, pulando", flush=True)
            continue
        rows = []
        for alpha in ALPHAS:
            participants, root = load_dataset_participants(
                "mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
            for attack in INFORMED_ATTACKS + BLIND_ATTACKS:
                t0 = time.time()
                lr = Learner(n_rounds=N_ROUNDS, n_classes=10, attack_type=attack, byzantine_ids=BYZ, seed=seed)
                acc = lr.train(participants, root_data=root, verbose=False)[-1].global_accuracy
                cnt = Counter(lr.choices)
                rows.append({"alpha": alpha, "attack_type": attack, "seed": seed, "accuracy": acc,
                             "choices": "|".join(lr.choices),
                             **{f"n_{r}": cnt.get(r, 0) for r in arsenal}})
                print(f"seed={seed} alpha={alpha} attack={attack} acc={acc:.4f} ({time.time() - t0:.0f}s) "
                      f"regras={dict(cnt)}", flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)


def ci95(x):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def analisar():
    ora = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/raw/seed*.csv"))])
    abl, ex9, ceil = pd.read_csv(ABL), pd.read_csv(EXP9), pd.read_csv(C0_CEIL)
    rows = []
    for (alpha, attack), g in ora.groupby(["alpha", "attack_type"]):
        o = g.set_index("seed")["accuracy"]
        sk = abl[(abl.alpha == alpha) & (abl.attack_type == attack)].pivot_table(
            index="seed", columns="mode", values="accuracy")[SKELETON]
        fx = ex9[(ex9.alpha == alpha) & (ex9.attack_type == attack)].pivot_table(
            index="seed", columns="strategy", values="accuracy")
        best_sk, best_fx = sk.mean().idxmax(), fx.mean().idxmax()
        q, qlo, qhi = ci95((o - sk[best_sk]).dropna())            # oráculo por regra − melhor esqueleto
        w, wlo, whi = ci95((sk[best_sk] - fx[best_fx]).dropna())  # esqueleto − melhor regra fixa estática
        g2, glo, ghi = ci95((o - fx[best_fx]).dropna())            # ganho do oráculo por rodada sobre o estático
        t = ceil[ceil.alpha == alpha]["accuracy"].mean()
        choices = Counter("|".join(g["choices"]).split("|"))
        rows.append({
            "alpha": alpha, "attack_type": attack, "excluida": (alpha, attack) in EXCLUDED,
            "oraculo_por_rodada": o.mean(), "melhor_esqueleto": best_sk, "acc_esqueleto": sk[best_sk].mean(),
            "melhor_regra_estatica": best_fx, "acc_regra_estatica": fx[best_fx].mean(), "teto_fedavg10": t,
            "Q_oraculo_menos_esqueleto_pp": 100 * q, "Q_ic95_lo": 100 * qlo, "Q_ic95_hi": 100 * qhi,
            "W_esqueleto_menos_estatica_pp": 100 * w, "W_ic95_lo": 100 * wlo, "W_ic95_hi": 100 * whi,
            "G_oraculo_menos_estatica_pp": 100 * g2, "G_ic95_lo": 100 * glo, "G_ic95_hi": 100 * ghi,
            "regras_escolhidas": ", ".join(f"{r}:{n}" for r, n in choices.most_common()),
        })
    res = pd.DataFrame(rows).sort_values(["alpha", "attack_type"])
    res.to_csv(f"{OUT}/oraculo_por_celula.csv", index=False)

    valid = res[~res.excluida]
    s_cells = valid[valid.W_ic95_lo > 0]  # esqueleto supera a melhor regra fixa estática (IC95 > 0)
    below = s_cells[s_cells.Q_ic95_hi < 0]
    above = s_cells[s_cells.Q_ic95_lo > 0]
    half = len(s_cells) / 2
    if len(s_cells) == 0:
        verdict = "SEM CÉLULAS-ALVO (o esqueleto não supera a melhor regra estática em nenhuma célula válida)"
    elif len(below) >= half:
        verdict = "A GRANULARIDADE EXPLICA: nem o oráculo por rodada alcança o esqueleto na maioria das células-alvo"
    elif len(above) >= half:
        verdict = "A GRANULARIDADE NÃO EXPLICA: o oráculo por rodada supera o esqueleto na maioria das células-alvo"
    else:
        verdict = "INCONCLUSIVO"
    cols = ["alpha", "attack_type", "oraculo_por_rodada", "melhor_esqueleto", "acc_esqueleto",
            "melhor_regra_estatica", "acc_regra_estatica", "Q_oraculo_menos_esqueleto_pp", "Q_ic95_lo",
            "Q_ic95_hi", "W_esqueleto_menos_estatica_pp", "G_oraculo_menos_estatica_pp"]
    lines = [f"B2.8 — oráculo guloso por rodada (7 regras), {len(ora)} runs; 19 células válidas "
             "(fltrust_aligned α ≤ 0,1 excluídas)", "",
             valid[cols].round(3).to_string(index=False), "",
             "Regras escolhidas pelo oráculo (contagem de rodadas por célula):",
             valid[["alpha", "attack_type", "regras_escolhidas"]].to_string(index=False), "",
             f"Células-alvo (esqueleto > melhor regra estática, IC95 > 0): {len(s_cells)}/19 — "
             + "; ".join(f"{r.attack_type} α={r.alpha}" for r in s_cells.itertuples()),
             f"  nelas, oráculo por rodada ABAIXO do esqueleto (IC95 < 0): {len(below)}"
             + ("  [" + "; ".join(f"{r.attack_type} α={r.alpha}" for r in below.itertuples()) + "]" if len(below) else ""),
             f"  nelas, oráculo por rodada ACIMA do esqueleto (IC95 > 0): {len(above)}"
             + ("  [" + "; ".join(f"{r.attack_type} α={r.alpha}" for r in above.itertuples()) + "]" if len(above) else ""),
             f"VEREDITO (critério do PLANO §4): {verdict}"]
    text = "\n".join(lines)
    open(f"{OUT}/analise.txt", "w").write(text + "\n")
    print(text)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    sub.add_parser("analisar")
    a = p.parse_args()
    if a.cmd == "run":
        run(a.seeds)
    else:
        analisar()


if __name__ == "__main__":
    main()
