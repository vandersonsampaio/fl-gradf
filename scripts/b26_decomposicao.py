"""
B2.6 (references/roadmap_tese_gradf_v3.md §4.1): decomposição confirmatória
do esqueleto do AdaAggRL no framework próprio. Pré-registro em
results/b26_decomposicao/PREREGISTRO.md. Não altera `src/` nem a ablação
(scripts/frente1_ablacao_adaaggrl.py).

Referência `sr_only` = exatamente o modo `sr_only` da ablação: score = S_R da
inversão de gradiente, ação a=[1,0,0,0], b=0,5, min-max de w_hat, limiar
delta=max(w_tilde)*b, contador h e penalidade lam**h com lam=2, agregação
ponderada dos parâmetros completos theta_k = global + update_k.
Cada variante muda UMA peça:
  sr_nomem        lam = 1 (memória desligada)
  sr_memof        memória do código oficial: peso x 0,9**flag com o flag ANTERIOR
                  ao decremento (auditoria §3.5); min-max e limiar iguais
  sr_bin          máscara binária: acima de delta recebe 1 / lam**h
  cosserver_only  score = cosseno até o update do servidor no root (sem inversão)
  sr_cosserver    score = 0,5 S_R + 0,5 cos_server (exploratório)
  sr_b025, sr_b075  b = 0,25 e 0,75 (sensibilidade)

O extrator aleatório é SEMPRE construído (como na ablação): seu __init__
re-semeia o np.random global, e omiti-lo quebraria o pareamento por semente.
O `cos_server` é calculado com save/restore do np.random, como na ablação.

Uso (CPU, baixa prioridade):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.b26_decomposicao run --variant sr_only --seed 52
  venv/bin/python -m scripts.b26_decomposicao analisar
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

OUT = "results/b26_decomposicao"
INPUT_SHAPE = (28, 28, 1)
ALPHAS = [0.5, 0.1, 0.05]
BYZ = [0, 1]
N_ROUNDS = 15
SEEDS = list(range(52, 62))
VARIANTS = ["sr_only", "sr_nomem", "sr_memof", "sr_bin", "cosserver_only", "sr_cosserver", "sr_b025", "sr_b075"]
EXCLUDED = {(0.05, "fltrust_aligned"), (0.1, "fltrust_aligned")}
MARGIN = 0.010
ALPHA_TEST = 0.05
NEEDS_SR = {"sr_only", "sr_nomem", "sr_memof", "sr_bin", "sr_cosserver", "sr_b025", "sr_b075"}
NEEDS_COS = {"cosserver_only", "sr_cosserver"}


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _minmax_threshold(score: np.ndarray, b: float):
    lo, hi = score.min(), score.max()
    w_tilde = (score - lo) / (hi - lo + 1e-8)
    delta = float(w_tilde.max() * b)
    flagged = w_tilde <= delta
    return w_tilde, flagged


def weights_for(variant: str, score: np.ndarray, h: np.ndarray):
    """Pesos finais (não normalizados) e novo contador h para cada variante."""
    b = {"sr_b025": 0.25, "sr_b075": 0.75}.get(variant, 0.5)
    w_tilde, flagged = _minmax_threshold(score, b)
    new_h = np.where(flagged, h + 1, np.maximum(h - 1, 0))
    if variant == "sr_memof":  # oficial: 0,9**flag com o flag anterior, só para os incluídos
        w = np.where(flagged, 0.0, w_tilde * (0.9 ** h))
    else:
        lam = 1.0 if variant == "sr_nomem" else 2.0
        base = np.where(flagged, 0.0, 1.0 if variant == "sr_bin" else w_tilde)
        w = base / (lam ** new_h)
    return w, new_h


def build_learner_cls():
    from src.defense.adaaggrl_agent import reconstruct_client_distribution
    from src.experiments.exp10_selector_comparison import AdaAggRLGridLearner
    from src.fl.attacked_learner import compute_param_updates_auto
    from src.fl.federated_learner import RoundResult

    class DecompLearner(AdaAggRLGridLearner):
        def __init__(self, *args, variant: str = "sr_only", **kwargs):
            kwargs["feature_extractor"] = "random"  # sempre construído — ver docstring
            super().__init__(*args, **kwargs)
            if variant not in VARIANTS:
                raise ValueError(variant)
            self.variant = variant

        def _run_round(self, round_num, participants, root_data):
            active = self._is_active(round_num)
            param_updates, _ = compute_param_updates_auto(self, participants, active, root_data)
            n_feat = participants[0].n_features
            K = 1 if self.n_classes == 2 else self.n_classes
            theta_list = [self._global_params + u for u in param_updates]
            n = len(participants)

            s_r = np.full(n, np.nan)
            if self.variant in NEEDS_SR:
                for i, (theta_k, u) in enumerate(zip(theta_list, param_updates)):
                    W = theta_k[: n_feat * K].reshape(n_feat, K).astype(np.float32)
                    b = theta_k[n_feat * K:].astype(np.float32)
                    _, s_r[i] = reconstruct_client_distribution(
                        W, b, u, self.local_lr, n_feat, K,
                        num_images=self.num_images, max_iters=self.max_iters, seed=round_num,
                    )
            cos_srv = None
            if self.variant in NEEDS_COS:
                rng_state = np.random.get_state()
                root = root_data or self._carve_root(participants[0])
                srv_update = self._make_model(n_feat).fit(root["X"], root["y"])
                np.random.set_state(rng_state)
                cos_srv = np.array([_cos(u, srv_update) for u in param_updates])

            if self.variant == "cosserver_only":
                score = cos_srv
            elif self.variant == "sr_cosserver":
                score = 0.5 * s_r + 0.5 * cos_srv
            else:
                score = s_r

            if self._h is None:
                self._h = np.zeros(n)
            w, self._h = weights_for(self.variant, score, self._h)
            tw = float(w.sum())
            nw = np.ones(n) / n if tw < 1e-8 else w / tw
            self._global_params = sum(wi * th for wi, th in zip(nw, theta_list))

            per_acc = {p.id: self._make_model(p.n_features).accuracy(p.X_test, p.y_test) for p in participants}
            return RoundResult(round_num, float(np.mean(list(per_acc.values()))), per_acc,
                               trust_scores=None, n_accepted=int((w > 0).sum()))

    return DecompLearner


def run(variant: str, seed: int):
    from src.experiments.exp9_dominance_grid import BLIND_ATTACKS, INFORMED_ATTACKS, _split_name
    from src.utils.data_loader import load_dataset_participants

    os.makedirs(f"{OUT}/raw", exist_ok=True)
    path = f"{OUT}/raw/{variant}_seed{seed}.csv"
    if os.path.exists(path):
        print(f"{path} já existe, pulando")
        return
    Learner = build_learner_cls()
    rows = []
    for alpha in ALPHAS:
        participants, root = load_dataset_participants("mnist", _split_name(alpha), 10, root_size=100, root_seed=seed)
        for attack in INFORMED_ATTACKS + BLIND_ATTACKS:
            t0 = time.time()
            lr = Learner(n_rounds=N_ROUNDS, n_classes=10, attack_type=attack, byzantine_ids=BYZ, seed=seed,
                         input_shape=INPUT_SHAPE, dataset="mnist", variant=variant)
            acc = lr.train(participants, root_data=root, verbose=False)[-1].global_accuracy
            rows.append({"alpha": alpha, "attack_type": attack, "variant": variant, "seed": seed, "accuracy": acc})
            print(f"variant={variant} seed={seed} alpha={alpha} attack={attack} ({time.time() - t0:.0f}s)", flush=True)
    tmp = path + ".tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Análise (pré-escrita; roda uma vez, com a grade completa)
# ---------------------------------------------------------------------------

def _desc(d):
    n, m = len(d), d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    h95, h90 = stats.t.ppf(0.975, n - 1) * se, stats.t.ppf(0.95, n - 1) * se
    return {"n": n, "delta_pp": 100 * m, "ic95": (100 * (m - h95), 100 * (m + h95)),
            "ic90": (100 * (m - h90), 100 * (m + h90)), "d": m / sd if sd > 0 else np.nan, "se": se}


def _tost(d, margin):
    n, m = len(d), d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(m) < margin else 1.0
    return max(1 - stats.t.cdf((m + margin) / se, n - 1), stats.t.cdf((m - margin) / se, n - 1))


def _holm(p):
    p = np.asarray(p, float)
    order, adj, run_max = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run_max
    return adj


def _wil(d, alternative="two-sided"):
    d = np.asarray(d, float)
    if np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(d, alternative=alternative).pvalue)


def analisar(raw_dir: str = f"{OUT}/raw", out: str = f"{OUT}/analise.txt"):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{raw_dir}/*_seed*.csv"))])
    df["valida"] = [(a, t) not in EXCLUDED for a, t in zip(df.alpha, df.attack_type)]
    v = df[df.valida]
    per_seed = v.groupby(["variant", "seed"])["accuracy"].mean().unstack("variant")  # média das 19 células

    lines = [f"B2.6 — {len(df)} linhas (esperadas {len(VARIANTS) * len(SEEDS) * 21}); unidade = semente "
             f"(média das 19 células válidas), n = {len(per_seed)}", "",
             "Média por variante (19 células, 10 sementes):",
             (per_seed.mean() * 100).round(2).to_string(), ""]

    ref = per_seed["sr_only"]
    specs = [  # (id, variante, teste, alternativa, lado do D)
        ("H1", "sr_nomem", "wilcoxon", "greater", "ref-var"),   # sr_only − sr_nomem > 0
        ("H2", "sr_memof", "wilcoxon", "two-sided", "ref-var"),
        ("H3", "sr_bin", "tost", None, "var-ref"),
        ("H4", "cosserver_only", "tost", None, "var-ref"),
    ]
    pvals, rows = [], []
    for hid, var, test, alt, side in specs:
        d = (ref - per_seed[var]) if side == "ref-var" else (per_seed[var] - ref)
        d = d.dropna().to_numpy()
        ds = _desc(d)
        p = _wil(d, alt) if test == "wilcoxon" else _tost(d, MARGIN)
        pvals.append(p)
        rows.append((hid, var, test, side, ds, p))
    adj = _holm(pvals)
    lines.append("== Hipóteses confirmatórias (Holm sobre H1–H4, α = 0,05) ==")
    labels = {"H1": "a memória contribui (sr_only − sr_nomem > 0)",
              "H2": "a força da memória importa (sr_only − sr_memof ≠ 0)",
              "H3": "máscara binária ≈ suave (sr_bin − sr_only, TOST ±1,0 p.p.)",
              "H4": "cos_server ≈ S_R (cosserver_only − sr_only, TOST ±1,0 p.p.)"}
    for (hid, var, test, side, ds, p), pa in zip(rows, adj):
        ok = pa < ALPHA_TEST
        lines.append(f"  {hid} {labels[hid]}: Δ={ds['delta_pp']:+.2f} p.p. IC95=({ds['ic95'][0]:+.2f}, {ds['ic95'][1]:+.2f}) "
                     f"IC90=({ds['ic90'][0]:+.2f}, {ds['ic90'][1]:+.2f}) d={ds['d']:+.2f} p={p:.4f} Holm={pa:.4f} -> "
                     f"{'CONFIRMADA' if ok else 'NÃO confirmada'}")
    lines.append("")

    lines.append("== Por ataque (descritivo; Wilcoxon bilateral, Holm sobre os 7 ataques; média das células válidas do ataque por semente) ==")
    for hid, var, _t, side, _ds, _p in rows:
        per_att = []
        for att, g in v.groupby("attack_type"):
            ps = g.groupby(["variant", "seed"])["accuracy"].mean().unstack("variant")
            d = ((ps["sr_only"] - ps[var]) if side == "ref-var" else (ps[var] - ps["sr_only"])).dropna().to_numpy()
            per_att.append((att, _desc(d), _wil(d)))
        adj_a = _holm([x[2] for x in per_att])
        lines.append(f"  {hid} ({var}):")
        for (att, ds, p), pa in zip(per_att, adj_a):
            lines.append(f"    {att:<18} Δ={ds['delta_pp']:+.2f} IC95=({ds['ic95'][0]:+.2f}, {ds['ic95'][1]:+.2f}) p={p:.4f} Holm={pa:.4f}")
    lines.append("")

    lines.append("== Exploratório (sem critério) ==")
    for var in ["sr_cosserver", "sr_b025", "sr_b075"]:
        ds = _desc((per_seed[var] - ref).dropna().to_numpy())
        lines.append(f"  {var} − sr_only: Δ={ds['delta_pp']:+.2f} p.p. IC95=({ds['ic95'][0]:+.2f}, {ds['ic95'][1]:+.2f})")
    best_single = np.maximum(per_seed["sr_only"], per_seed["cosserver_only"])
    ds = _desc((per_seed["sr_cosserver"] - best_single).dropna().to_numpy())
    lines.append(f"  sr_cosserver − max(sr_only, cosserver_only) por semente: Δ={ds['delta_pp']:+.2f} p.p. "
                 f"IC95=({ds['ic95'][0]:+.2f}, {ds['ic95'][1]:+.2f})")
    cell = v.groupby(["alpha", "attack_type", "variant"])["accuracy"].mean().unstack("variant")
    lines += ["", "Média por célula válida (10 sementes) — S_R contra cos_server contra combinado:",
              (cell[["sr_only", "cosserver_only", "sr_cosserver"]] * 100).round(2).to_string()]

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(text + "\n")
    per_seed.to_csv(os.path.join(os.path.dirname(out), "por_semente.csv"))
    print(text)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--variant", required=True, choices=VARIANTS)
    r.add_argument("--seed", type=int, required=True)
    an = sub.add_parser("analisar")
    an.add_argument("--raw_dir", default=f"{OUT}/raw")
    an.add_argument("--out", default=f"{OUT}/analise.txt")
    a = p.parse_args()
    if a.cmd == "run":
        run(a.variant, a.seed)
    else:
        analisar(a.raw_dir, a.out)


if __name__ == "__main__":
    main()
