"""
B2.7 (references/roadmap_tese_gradf_v4.md §4.1): horizonte 15/50/150.
Plano em results/b27_horizonte/PLANO.md. Exploratório, sementes 42-51.

Pergunta: o ganho de cada agente sobre a versão fixa NO MESMO ESPAÇO DE AÇÃO
(mesmo sinal, granularidade e memória) cresce com o horizonte?

  agente                       versão fixa                            referência extra
  td3_ref (AdaAggRL, ablação)  fixed (mesmo esqueleto, ação no centro)  —
  linucb (FedStrategist b)     arm_rule_<r>: a regra r aplicada como o  rand_rule: regra uniforme
                               LinUCB aplica (mesmo round, sem bandit)  por rodada
  dqn (GRADF v1)               arm_gradf_<r>: pipeline do GRADF com     rand_gradf: RandomSelector
                               FixedActionSelector(r)                   (como no exp10)
  r ∈ arsenal de 7 regras (FULL_ARSENAL do exp10).

Horizontes ANINHADOS: cada sistema roda 150 rodadas e a acurácia nos
horizontes H ∈ {15, 50, 150} é a do modelo global ao fim da rodada H. Nada nos
learners depende de n_rounds (só o laço de treino), então é idêntico a rodar
H rodadas; o subcomando `verificar` confirma isso numa célula×semente.

Tetos por α×semente×H: FedAvg-10 sem ataque (como no C.0) e oráculo FedAvg-8
(só os 8 honestos, avaliado nos test sets dos 10, como no c0_teto_oraculo).

Uso (CPU, sem GPU):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.b27_horizonte celula --alpha 0.05 --attack label_flipping --seed 42
  ... -m scripts.b27_horizonte teto --alpha 0.05 --seed 42
  ... -m scripts.b27_horizonte verificar
  ... -m scripts.b27_horizonte analisar
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/b27_horizonte"
HORIZONS = [15, 50, 150]
H_MAX = max(HORIZONS)
SEEDS = list(range(42, 52))
CELLS = [
    (0.05, "label_flipping"), (0.1, "label_flipping"),
    (0.05, "sign_flipping"), (0.1, "sign_flipping"),
    (0.05, "gaussian_noise"), (0.1, "krum_collusion"),
    (0.05, "low_mag_backdoor"), (0.5, "trim_attack"),
]
BYZ = [0, 1]
INPUT_SHAPE = (28, 28, 1)
MIN_CELLS = 3  # "o aprendizado se paga em H": Δ > 0 com IC95 > 0 em ≥ 3 das 8 células


def _reseed(seed):
    """O `seed` dos learners não semeia o np.random global (permutação do treino
    local, ruído DP), então sem isto cada sistema herdaria o estado deixado pelo
    anterior no mesmo processo. Ressemeia python/numpy/TF antes de cada sistema
    (o mesmo que o extrator do td3_ref/fixed já faz no __init__)."""
    import keras
    keras.utils.set_random_seed(seed)


def _accs(results):
    return {H: float(results[H - 1].global_accuracy) for H in HORIZONS if len(results) >= H}


def build_plain_rule_learner():
    """Regra fixa (ou aleatória por rodada) aplicada exatamente como o
    FedStrategistGridLearner aplica a regra escolhida: mesmos updates
    (compute_param_updates_auto), server_update ajustado no root a cada rodada,
    _make_arsenal_strategy(regra, n_byz). Só sem detecção e sem bandit."""
    from src.experiments.exp10_selector_comparison import FULL_ARSENAL, _make_arsenal_strategy
    from src.fl.attacked_learner import AttackedFederatedLearner, compute_param_updates_auto
    from src.fl.federated_learner import RoundResult

    class PlainRuleLearner(AttackedFederatedLearner):
        def __init__(self, *args, rule: str = "fedavg", random_seed=None, **kwargs):
            kwargs.setdefault("aggregation", "fedavg")  # placeholder; a regra vem de `rule`
            super().__init__(*args, **kwargs)
            self.rule = rule
            self._rng = None if random_seed is None else np.random.RandomState(random_seed)

        def _run_round(self, round_num, participants, root_data):
            active = self._is_active(round_num)
            updates, is_byz = compute_param_updates_auto(self, participants, active, root_data)
            n_feat = participants[0].n_features
            rule = self.rule if self._rng is None else FULL_ARSENAL[self._rng.randint(len(FULL_ARSENAL))]
            server_root = root_data or self._carve_root(participants[0])
            server_update = self._make_model(n_feat).fit(server_root["X"], server_root["y"])
            n_byz = sum(1 for b in is_byz if b)
            delta, _ = _make_arsenal_strategy(rule, n_byz).aggregate(
                updates, sample_sizes=[p.n_train for p in participants], server_update=server_update)
            self._global_params = self._global_params + delta
            per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
            return RoundResult(round_num, float(np.mean(list(per_acc.values()))), per_acc,
                               trust_scores=None, n_accepted=None)

    return PlainRuleLearner, FULL_ARSENAL


def run_cell(alpha, attack, seed, n_rounds=H_MAX, systems=None, out_dir=None):
    from scripts.frente1_ablacao_adaaggrl import build_learner_cls
    from src.defense.rl_selector import RLDefenseSelector
    from src.experiments.exp1_baseline import (
        FixedActionSelector, GroundTruthClassifierStub, RandomSelector, train_gradf_models,
    )
    from src.experiments.exp10_selector_comparison import FedStrategistGridLearner
    from src.experiments.exp9_dominance_grid import _split_name
    from src.fl.gradf_learner import GRADFFederatedLearner
    from src.utils.data_loader import load_dataset_participants

    out_dir = out_dir or f"{OUT}/raw"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/celula_{attack}_a{alpha}_seed{seed}_R{n_rounds}.csv"
    if os.path.exists(path):
        print(f"já existe: {path}", flush=True)
        return path
    Ablation = build_learner_cls()
    PlainRule, arsenal = build_plain_rule_learner()
    participants, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    labels = ["none", attack]
    common = dict(n_rounds=n_rounds, n_classes=10, attack_type=attack, byzantine_ids=BYZ, seed=seed)
    all_systems = (["td3_ref", "fixed", "linucb", "dqn", "rand_rule", "rand_gradf"]
                   + [f"arm_rule_{r}" for r in arsenal] + [f"arm_gradf_{r}" for r in arsenal])
    systems = systems or all_systems

    rows, t_all = [], time.time()
    t0 = time.time()
    _reseed(seed)
    clf, sel, _labels, _omap = train_gradf_models(participants, attack, 0.2, 10, n_rounds=8, seed=seed)
    t_pre = time.time() - t0

    def make(system):
        if system in ("td3_ref", "fixed"):
            return Ablation(input_shape=INPUT_SHAPE, dataset="mnist",
                            mode="td3" if system == "td3_ref" else "fixed", **common)
        if system == "linucb":
            return FedStrategistGridLearner(variant="b", shared_classifier=clf, n_labels=len(labels), **common)
        if system == "dqn":
            return GRADFFederatedLearner(aggregation="fedavg", classifier=clf, selector=sel,
                                         attack_labels=labels, **common)
        if system == "rand_gradf":
            return GRADFFederatedLearner(aggregation="fedavg", classifier=GroundTruthClassifierStub(),
                                         selector=RandomSelector(RLDefenseSelector().actions, seed=seed),
                                         attack_labels=labels, **common)
        if system == "rand_rule":
            return PlainRule(random_seed=seed, **common)
        if system.startswith("arm_rule_"):
            return PlainRule(rule=system[len("arm_rule_"):], **common)
        if system.startswith("arm_gradf_"):
            return GRADFFederatedLearner(aggregation="fedavg", classifier=clf,
                                         selector=FixedActionSelector(system[len("arm_gradf_"):]),
                                         attack_labels=labels, **common)
        raise ValueError(system)

    for system in systems:
        t0 = time.time()
        _reseed(seed)
        res = make(system).train(participants, root_data=root, verbose=False)
        dt = time.time() - t0
        for H, acc in _accs(res).items():
            rows.append({"alpha": alpha, "attack_type": attack, "seed": seed, "system": system,
                         "H": H, "accuracy": acc, "sec": dt})
        print(f"alpha={alpha} attack={attack} seed={seed} {system} ({dt:.0f}s)", flush=True)
    df = pd.DataFrame(rows).assign(pretrain_sec=t_pre)
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    print(f"FIM {path} ({time.time() - t_all:.0f}s)", flush=True)
    return path


def run_ceiling(alpha, seed, n_rounds=H_MAX, out_dir=None):
    from src.experiments.exp9_dominance_grid import _split_name
    from src.fl.attacked_learner import AttackedFederatedLearner
    from src.utils.data_loader import load_dataset_participants

    out_dir = out_dir or f"{OUT}/raw"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/teto_a{alpha}_seed{seed}_R{n_rounds}.csv"
    if os.path.exists(path):
        print(f"já existe: {path}", flush=True)
        return path
    participants, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
    rows = []

    _reseed(seed)
    fa10 = AttackedFederatedLearner(n_rounds=n_rounds, n_classes=10, attack_type="none",
                                    byzantine_ids=BYZ, seed=seed, aggregation="fedavg")
    for H, acc in _accs(fa10.train(participants, root_data=root, verbose=False)).items():
        rows.append({"alpha": alpha, "seed": seed, "system": "teto_fedavg10", "H": H, "accuracy": acc})

    honest = [p for i, p in enumerate(participants) if i not in BYZ]
    acc10 = {}

    class _Oracle8(AttackedFederatedLearner):
        def _run_round(self, round_num, parts, root_data):
            rr = super()._run_round(round_num, parts, root_data)
            if round_num + 1 in HORIZONS:
                m = self._make_model(participants[0].n_features)
                m.set_params(self._global_params)
                acc10[round_num + 1] = float(np.mean([m.accuracy(p.X_test, p.y_test) for p in participants]))
            return rr

    _reseed(seed)
    _Oracle8(n_rounds=n_rounds, n_classes=10, attack_type="none", byzantine_ids=[], seed=seed,
             aggregation="fedavg").train(honest, root_data=root, verbose=False)
    for H, acc in acc10.items():
        rows.append({"alpha": alpha, "seed": seed, "system": "teto_oraculo_fedavg8", "H": H, "accuracy": acc})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"FIM {path}", flush=True)
    return path


def verificar():
    """(1) aninhamento: na célula (0.05, label_flipping), semente 42, rodar com
    n_rounds=15 e 50 deve dar exatamente a acurácia do run de 150 em H=15 e 50;
    (2) reprodução em H=15 contra os runs antigos (descritivo)."""
    lines = ["B2.7 — verificações", ""]
    tmp = f"{OUT}/verificacao"
    full = f"{OUT}/raw/celula_label_flipping_a0.05_seed42_R{H_MAX}.csv"
    if os.path.exists(full):
        ref = pd.read_csv(full)
        for n in (15, 50):
            p = run_cell(0.05, "label_flipping", 42, n_rounds=n,
                         systems=["td3_ref", "fixed", "linucb", "dqn", "rand_rule", "rand_gradf",
                                  "arm_rule_median", "arm_gradf_median"], out_dir=tmp)
            short = pd.read_csv(p)
            m = short.merge(ref[ref.H == n], on=["system", "H"], suffixes=("_curto", "_150"))
            d = (m["accuracy_curto"] - m["accuracy_150"]).abs()
            lines.append(f"(1) aninhamento H={n}: {len(m)} sistemas, máx |Δ| = {d.max():.2e}  "
                         f"-> {'OK' if d.max() < 1e-9 else 'DIFERENTE: ' + ', '.join(m.loc[d >= 1e-9, 'system'])}")
    else:
        lines.append(f"(1) aninhamento: falta {full}")

    raw = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/raw/celula_*_R{H_MAX}.csv")])
    r15 = raw[raw.H == 15]
    e10 = pd.concat([pd.read_csv(f) for f in glob.glob("results/tables/exp10_selector_comparison_variantb_full_seed*_raw.csv")])
    abl = pd.concat([pd.read_csv(f) for f in glob.glob("results/frente1_ablacao_adaaggrl/raw/fixed_seed*.csv")])
    ex9 = pd.read_csv("results/tables/exp9_dominance_grid_10seeds_ALL_root100_raw.csv")
    refs = [("td3_ref", e10[e10.system == "AdaAggRL"]), ("linucb", e10[e10.system == "FedStrategist"]),
            ("dqn", e10[e10.system == "GRADF"]), ("rand_gradf", e10[e10.system == "Random"]),
            ("fixed", abl)]
    refs += [(f"arm_rule_{r}", ex9[ex9.strategy == r]) for r in ["fltrust", "median", "trimmed_mean", "krum"]]
    lines += ["", "(2) reprodução em H=15 contra os runs antigos (descritivo; exp9 usa outro caminho de agregação):"]
    for sysname, old in refs:
        m = r15[r15.system == sysname].merge(old[["alpha", "attack_type", "seed", "accuracy"]],
                                              on=["alpha", "attack_type", "seed"], suffixes=("", "_antigo"))
        if len(m):
            d = (m["accuracy"] - m["accuracy_antigo"]).abs()
            lines.append(f"  {sysname:22s} n={len(m):3d}  máx |Δ|={d.max():.4f}  média |Δ|={d.mean():.4f}  "
                         f"idênticos={int((d < 1e-9).sum())}")
    text = "\n".join(lines) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/verificacao.txt", "w").write(text)
    print(text)


def _ci95(d):
    n = len(d)
    m = d.mean()
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def analisar():
    raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/raw/celula_*_R{H_MAX}.csv"))])
    tetos = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/raw/teto_*_R{H_MAX}.csv"))])
    raw.to_csv(f"{OUT}/grade_raw.csv", index=False)
    tetos.to_csv(f"{OUT}/tetos_raw.csv", index=False)
    rule_arms = sorted(s for s in raw.system.unique() if s.startswith("arm_rule_"))
    gradf_arms = sorted(s for s in raw.system.unique() if s.startswith("arm_gradf_"))
    agents = [("td3_ref", None, None), ("linucb", rule_arms, "rand_rule"), ("dqn", gradf_arms, "rand_gradf")]

    rows = []
    for (alpha, attack), g in raw.groupby(["alpha", "attack_type"]):
        for H, gh in g.groupby("H"):
            w = gh.pivot_table(index="seed", columns="system", values="accuracy")
            for agent, arms, rnd in agents:
                if agent not in w:
                    continue
                if arms is None:
                    ref_name = "fixed"
                else:
                    ref_name = w[arms].mean().idxmax()  # melhor braço em retrospecto (referência otimista)
                d = (w[agent] - w[ref_name]).dropna().to_numpy()
                m, lo, hi = _ci95(d)
                row = {"alpha": alpha, "attack_type": attack, "H": H, "agente": agent, "ref_fixa": ref_name,
                       "n": len(d), "acc_agente": w[agent].mean(), "acc_ref": w[ref_name].mean(),
                       "delta_pp": 100 * m, "ic95_lo_pp": 100 * lo, "ic95_hi_pp": 100 * hi,
                       "se_paga": bool(m > 0 and lo > 0)}
                if rnd is not None and rnd in w:
                    dr = (w[agent] - w[rnd]).dropna().to_numpy()
                    mr, lor, hir = _ci95(dr)
                    row.update({"delta_vs_aleatorio_pp": 100 * mr, "ic95_aleat_lo_pp": 100 * lor,
                                "ic95_aleat_hi_pp": 100 * hir})
                t = tetos[(tetos.alpha == alpha) & (tetos.H == H)].pivot_table(index="seed", columns="system",
                                                                                 values="accuracy")
                for tname in ["teto_fedavg10", "teto_oraculo_fedavg8"]:
                    if tname in t:
                        gap = (t[tname] - w[agent]).dropna()
                        row[f"gap_{tname}_pp"] = 100 * gap.mean()
                rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/delta_por_celula.csv", index=False)

    pd.set_option("display.width", 250)
    lines = ["B2.7 — horizonte 15/50/150 (exploratório; sementes 42–51; 8 células)",
             "Critério: o aprendizado se paga em H se Δ(agente − fixo) > 0 com IC95 > 0 em ≥ 3 das 8 células.",
             "Referência fixa do LinUCB e do DQN = melhor braço em retrospecto por célula e H (OTIMISTA para o fixo).",
             "Sem correção de multiplicidade (3 agentes × 3 horizontes), declarado no plano.", ""]
    for agent, _, _ in agents:
        a = res[res.agente == agent]
        if not len(a):
            continue
        lines.append(f"== {agent} ==")
        for H in HORIZONS:
            ah = a[a.H == H]
            k = int(ah.se_paga.sum())
            lines.append(f"  H={H:3d}: células com Δ>0 e IC95>0 = {k}/{len(ah)} -> "
                         f"{'SE PAGA' if k >= MIN_CELLS else 'não se paga'};  Δ médio sobre as células = "
                         f"{ah.delta_pp.mean():+.2f} p.p.")
        piv = a.pivot_table(index=["alpha", "attack_type"], columns="H", values="delta_pp")
        slope = piv.apply(lambda r: np.polyfit(np.log(HORIZONS), r.values, 1)[0], axis=1)
        lines.append("  Δ (p.p.) por célula × H, e inclinação por log(H):")
        lines.append((piv.assign(inclinacao_logH=slope)).round(2).to_string())
        cols = ["alpha", "attack_type", "H", "ref_fixa", "acc_agente", "acc_ref", "delta_pp", "ic95_lo_pp",
                "ic95_hi_pp", "se_paga"] + [c for c in a.columns if c.startswith("delta_vs") or c.startswith("gap_")]
        lines.append(a[cols].round(4).to_string(index=False))
        lines.append("")
    lines.append("Tetos por α × H (média de 10 sementes):")
    lines.append(tetos.groupby(["alpha", "system", "H"])["accuracy"].mean().unstack("H").round(4).to_string())
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
    c.add_argument("--n_rounds", type=int, default=H_MAX)
    c.add_argument("--systems", nargs="+", default=None)
    c.add_argument("--out_dir", default=None)
    t = sub.add_parser("teto")
    t.add_argument("--alpha", type=float, required=True)
    t.add_argument("--seed", type=int, required=True)
    sub.add_parser("verificar")
    sub.add_parser("analisar")
    a = ap.parse_args()
    if a.cmd == "celula":
        run_cell(a.alpha, a.attack, a.seed, a.n_rounds, a.systems, a.out_dir)
    elif a.cmd == "teto":
        run_ceiling(a.alpha, a.seed)
    elif a.cmd == "verificar":
        verificar()
    else:
        analisar()


if __name__ == "__main__":
    main()
