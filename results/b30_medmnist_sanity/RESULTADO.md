# Resultado — B3.0: sanity do BloodMNIST no AdaAggRL oficial

**Data:** 2026-10-05. Regras fixadas em `results/b31_medmnist_oficial/PREREGISTRO.md` §3 (commit `315b880`), antes de qualquer run. Código e artefatos com hash em `CODIGO.sha256`.

**Grade:** 12 runs de FedAvg uniforme, 500 rodadas, sementes 130–132 (BloodMNIST sem ataque, LMP e EB; MNIST sem ataque), de 04/10 12h29 a 05/10 17h26, com pausa 7h–16h29 em 05/10. Sem falhas.

## Aplicação mecânica das regras (`analisar_b30.py` → `analise.txt`)

| condição | T por semente (mediana 401–500) | média | resets |
|---|---|---|---|
| BloodMNIST, sem ataque | 77,39 / 77,83 / 78,26% | **77,83%** | 0 / 0 / 0 |
| BloodMNIST, LMP | 23,75 / 19,55 / 19,60% | 20,97% | 108 / 101 / 113 |
| BloodMNIST, EB | 21,79 / 7,13 / 22,29% | 17,07% | 113 / 14 / 114 |
| MNIST, sem ataque | 96,84 / 96,85 / 97,05% | **96,92%** | 0 / 0 / 0 |

- **Convergência: CONVERGE.** T_Blood = 77,8% (≥ 50%) e média 451–500 − 401–450 = **+1,79 p.p.** (< 2). **No limite:** a curva ainda sobe devagar no fim do run.
- **Inclusão dos ataques: LMP e EB ENTRAM.** O FedAvg cai 56,9 p.p. com LMP e 60,8 p.p. com EB (limiar: 5 p.p.). Sem defesa, os dois ataques levam o ambiente oficial a cascatas de reset (~100 por run).
- **Margem:** M = 1,0 × (100 − 77,83) / (100 − 96,92) = 7,19 p.p. → **M = 3,00 p.p.** pelo teto da regra. Pelo pré-registro, é uma **equivalência fraca**.

**Decisão (regra do pré-registro):** B3.1 com ataques **LMP e EB** e margem **±3,00 p.p.**

## Notas

- **Custo:** ~8–11 h por run com 6 processos na GPU, ou 57–92 s por rodada. Os runs da semente 132 somam ~17–18 h de relógio porque incluem a pausa de 7h–16h29.
- **Extrator do BloodMNIST:** 90,5% de acurácia no teste após 15 épocas (`data/models/extract_feature_bloodmnist.pt.json`).
- **Métrica e agregação no código oficial:** ver `results/b31_medmnist_oficial/ADENDO0_metrica_agregacao.md` (acurácia no test set global; o FedAvg do B3.0 é média uniforme).
