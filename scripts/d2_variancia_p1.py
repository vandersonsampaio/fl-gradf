"""
D2 (results/d2_variancia_p1/PLANO.md): variância de execução dos números do P1.

`run --seed S --rep R --worktree DIR`: roda, num processo novo, o exp10 da tag p1.0.0
(worktree DIR, sem alterações) com a mesma chamada do P1 (variant b, root 100, 5 sistemas,
21 células) e salva o CSV em results/d2_variancia_p1/raw/rep{R}_seed{S}.csv.
`analisar`: compara as 3 repetições + a execução original do P1 (CSV da tag).

Uso:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python scripts/d2_variancia_p1.py run --seed 42 --rep 1 --worktree /caminho/p1_worktree
  venv/bin/python scripts/d2_variancia_p1.py analisar
"""

import argparse
import io
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "d2_variancia_p1")
SEEDS = [42, 43, 44]
SYSTEMS_MAP = {"Random": "random", "Oracle (decoupled)": "oracle", "GRADF": "gradf",
               "FedStrategist": "fedstrategist", "AdaAggRL": "adaaggrl"}
PUBLICADO = {"GRADF": -0.042, "FedStrategist": -0.044, "AdaAggRL": 0.030}


def run(seed, rep, worktree):
    path = os.path.join(OUT, "raw", f"rep{rep}_seed{seed}.csv")
    if os.path.exists(path):
        print(f"já existe: {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.chdir(worktree)
    sys.path.insert(0, worktree)
    from src.experiments.exp10_selector_comparison import run_selector_comparison_grid
    df = run_selector_comparison_grid(dataset="mnist", seed=seed, variant="b", root_size=100)
    df["seed"] = seed
    df["rep"] = rep
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    print(f"FIM {path}")


def analisar():
    import glob

    import numpy as np
    import pandas as pd

    orig = pd.read_csv(io.StringIO(subprocess.check_output(
        ["git", "-C", REPO, "show", "p1.0.0:results/tables/exp10_FULL_variantb_raw.csv"]).decode()))
    orig_all = orig.copy()
    orig = orig[orig.seed.isin(SEEDS)].assign(rep=0)
    reps = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "raw", "rep*_seed*.csv")))])
    cols = ["alpha", "attack_type", "system", "accuracy", "seed", "rep"]
    d = pd.concat([orig[cols], reps[cols]])
    d.to_csv(os.path.join(OUT, "grade_raw.csv"), index=False)

    cel = d.groupby(["system", "alpha", "attack_type", "seed"])["accuracy"].agg(["std", "min", "max", "count"])
    cel["amplitude"] = cel["max"] - cel["min"]
    cel.to_csv(os.path.join(OUT, "sd_exec_por_celula.csv"))
    por_sis = cel.groupby("system")[["std", "amplitude"]].agg(["mean", "max"])

    def agg_delta(df):
        w = df.pivot_table(index=["alpha", "attack_type"], columns="system", values="accuracy")
        return {s: float((w[s] - w["Random"]).mean()) for s in PUBLICADO if s in w}

    rows = []
    for (seed, rep), g in d.groupby(["seed", "rep"]):
        rows.append({"seed": seed, "rep": rep, **agg_delta(g)})
    ag = pd.DataFrame(rows)
    ag.to_csv(os.path.join(OUT, "delta_agregado_por_execucao.csv"), index=False)
    sd_exec = ag.groupby("seed")[list(PUBLICADO)].std().mean()
    sd_sem = pd.DataFrame([{"seed": s, **agg_delta(g)} for s, g in orig_all.groupby("seed")])[list(PUBLICADO)].std()

    sinais_ok = bool((ag["GRADF"] < 0).all() and (ag["FedStrategist"] < 0).all() and (ag["AdaAggRL"] > 0).all())
    sd_ok = bool(all(sd_exec[s] < 0.5 * abs(PUBLICADO[s]) for s in PUBLICADO))
    veredito = ("CONCLUSÃO DO P1 ROBUSTA À VARIÂNCIA DE EXECUÇÃO" if sinais_ok and sd_ok
                else "A VARIÂNCIA DE EXECUÇÃO É DA ORDEM DO EFEITO (nota de limitação com magnitudes)")
    pd.set_option("display.width", 220)
    lines = ["D2 — variância de execução do P1 (tag p1.0.0; sementes 42–44; 3 repetições + execução original)",
             f"linhas: {len(d)} (esperado {21 * 5 * 3 * 4})", "",
             "SD_exec por sistema (entre as 4 execuções, por célula × semente; média e máximo):", por_sis.round(4).to_string(), "",
             "Δ agregado (sistema − Random, média das 21 células) por execução × semente:", ag.round(4).to_string(index=False), "",
             "SD_exec do Δ agregado (média nas 3 sementes): " + ", ".join(f"{s} {sd_exec[s]:.4f}" for s in PUBLICADO),
             "SD entre as 10 sementes do P1 original:      " + ", ".join(f"{s} {sd_sem[s]:.4f}" for s in PUBLICADO),
             "Δ publicado (Tabela 4):                       " + ", ".join(f"{s} {v:+.3f}" for s, v in PUBLICADO.items()), "",
             f"Sinais em todas as 12 combinações (GRADF<0, FedStrat<0, AdaAggRL>0): {sinais_ok}",
             f"SD_exec < ½|Δ publicado| nos três: {sd_ok}",
             f"REGRA DE LEITURA (PLANO §4): {veredito}"]
    text = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "analise.txt"), "w").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--rep", type=int, required=True)
    r.add_argument("--worktree", required=True)
    sub.add_parser("analisar")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.seed, a.rep, a.worktree)
    else:
        analisar()


if __name__ == "__main__":
    main()
