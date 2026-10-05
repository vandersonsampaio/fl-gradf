"""
Análise do B3.0 (sanity), conforme results/b31_medmnist_oficial/PREREGISTRO.md §3.
Aplica mecanicamente as regras fixadas antes de qualquer run:
  T = mediana da acurácia nas rodadas 401-500 (FedAvg uniforme), sementes 130-132.
  Convergência: média de T (BloodMNIST, sem ataque) ≥ 50% E |média 451-500 − média
    401-450| < 2 p.p. (média entre sementes); senão: parar e consultar o autor.
  Inclusão: ataque entra se média de T sob ataque ≤ média de T sem ataque − 5 p.p.
  Margem: M = 1,0 × (100 − T_Blood) / (100 − T_MNIST), arredondada para cima ao
    múltiplo de 0,25 p.p., limitada a [1,0; 3,0].

Uso (venv oficial): external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_b30.py
"""

import json
import math
import os

import numpy as np

OUT = "results/b30_medmnist_sanity"
RAW = f"{OUT}/raw"
SEEDS = [130, 131, 132]
ROUNDS = 500


def _load(dataset, attack, seed):
    p = f"{RAW}/{dataset}_{attack}_q0.5_fedavg_seed{seed}_R{ROUNDS}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    acc = np.array([s["acc"] for s in d["steps"][:ROUNDS]])
    return {"T": float(np.median(acc[400:500])), "m401_450": float(acc[400:450].mean()),
            "m451_500": float(acc[450:500].mean()), "resets": len(d["resets"]) - 1,
            "sec_por_rodada": float(np.mean([s["sec"] for s in d["steps"]]))}


def main():
    lines = ["B3.0 — sanity BloodMNIST (FedAvg uniforme, 500 rodadas, sementes 130–132)", ""]
    res = {}
    for ds, att in [("BloodMNIST", "none"), ("BloodMNIST", "LMP"), ("BloodMNIST", "EB"), ("MNIST", "none")]:
        rs = [_load(ds, att, s) for s in SEEDS]
        if any(r is None for r in rs):
            raise SystemExit(f"faltam runs de {ds}/{att}")
        res[(ds, att)] = rs
        lines.append(f"{ds:10s} {att:4s}: T por semente = {[round(100 * r['T'], 2) for r in rs]}%, "
                     f"média {100 * np.mean([r['T'] for r in rs]):.2f}%; resets {[r['resets'] for r in rs]}; "
                     f"s/rodada {np.mean([r['sec_por_rodada'] for r in rs]):.1f}")
    T_blood = float(np.mean([r["T"] for r in res[("BloodMNIST", "none")]]))
    T_mnist = float(np.mean([r["T"] for r in res[("MNIST", "none")]]))
    drift = float(np.mean([r["m451_500"] - r["m401_450"] for r in res[("BloodMNIST", "none")]]))
    conv = T_blood >= 0.50 and abs(100 * drift) < 2.0
    lines += ["", f"Convergência: T_Blood = {100 * T_blood:.2f}% (≥ 50%?) e média 451–500 − 401–450 = "
                  f"{100 * drift:+.2f} p.p. (|·| < 2?) -> " + ("CONVERGE" if conv else "NÃO converge: parar e consultar o autor")]
    incl = []
    for att in ["LMP", "EB"]:
        Ta = float(np.mean([r["T"] for r in res[("BloodMNIST", att)]]))
        ok = (T_blood - Ta) >= 0.05
        incl.append(att) if ok else None
        lines.append(f"Inclusão {att}: queda = {100 * (T_blood - Ta):+.2f} p.p. (≥ 5?) -> " + ("ENTRA" if ok else "NÃO entra"))
    raw_m = 1.0 * (1 - T_blood) / (1 - T_mnist)
    M = min(3.0, max(1.0, math.ceil(raw_m / 0.25 - 1e-9) * 0.25))
    lines += [f"Margem: T_MNIST = {100 * T_mnist:.2f}%; M bruta = {raw_m:.3f} p.p. -> M = {M:.2f} p.p."
              + (" (limitada a 3,0: equivalência fraca)" if raw_m > 3.0 else ""), "",
              "DECISÃO: " + ("B3.1 com ataques " + str(incl) + f" e margem ±{M:.2f} p.p." if conv and incl
                             else "parar e consultar o autor (portão não atendido)")]
    text = "\n".join(lines) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/analise.txt", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
