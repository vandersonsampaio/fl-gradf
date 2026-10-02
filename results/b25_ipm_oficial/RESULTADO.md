# Resultado — B2.5: IPM real no AdaAggRL oficial (fixed contra td3)

**Data:** 2026-10-02.
- Plano: `PLANO.md`, commit `1e92014`.
- Adendo 1 (ε = 2): `ADENDO1.md`, commit `4faad80`.
- Grade: 20 runs (sementes 105–114 × {td3, fixed}, 500 rodadas), de 01/10 14h12 a 02/10 19h36. Sem falhas.
- Saídas: `analise.txt`, `resumo_runs.csv`, `mecanismo.csv`, `raw/`.

## H1 (pré-registrada): INCONCLUSIVA

| métrica | Δ fixed − td3 | IC90 | IC95 | TOST ±1 p.p. | Wilcoxon |
|---|---|---|---|---|---|
| **primária:** mediana 401–500 | +2,78 p.p. | −4,63 a +10,19 | −6,37 a +11,92 | p = 0,66 | p = 0,63 |
| secundária: média 451–500 | +4,23 p.p. | −5,65 a +14,12 | −7,96 a +16,43 | p = 0,72 | p = 0,56 |

- Nem equivalência nem diferença. O sinal favorece o **fixed**, mas sem significância.
- **Não há evidência de que o TD3 contribua sob IPM.**

## Por que inconclusiva: cascatas de reset

- O sd das diferenças por semente é **12,8 p.p.**, contra ~0,3 p.p. no B2.1.
- O IPM com ε = 2 provoca muitos resets do ambiente oficial (recompensa < −80 → reinicialização do modelo): mediana de **2,5 resets extras por run nas duas condições**.
- Em duas sementes há **cascatas tardias** dentro da janela 401–500:
  - **110, td3:** 7 resets, 5 deles entre as rodadas 414 e 494 → mediana 57,9%;
  - **111, fixed:** 7 resets, 4 deles entre as rodadas 402 e 435 → 68,4%;
  - **111, td3:** resets em 412 e 428 → 81,8%.
- Como o reset é consequência da ação, o plano não permite excluí-lo. A mediana de 100 rodadas não resiste a cascatas com mais de ~50 rodadas contaminadas, ao contrário do B2.1, onde os resets eram isolados.
- **Sensibilidade post hoc**, sem valor confirmatório: excluindo as sementes 110 e 111, Δ = +0,58 p.p., IC90 de −0,99 a +2,15, sd 2,35. O resultado continua sem equivalência a ±1 p.p. e sem diferença.

## Mecanismo (descritivo): o TD3 continua sem aprender

| | mediana | máx. |
|---|---|---|
| drift \|π₅₀₀ − π₀\| | 0,0107 | 0,0168 |
| sd_estados de π₅₀₀ | 0,0029 | 0,0106 |

- Os dois ficam bem abaixo do ruído de exploração (σ_a = 0,0475), como no B2.1 (drift ~0,01).
- Sob IPM, a política final é praticamente a inicial e não depende do estado.
- A diferença de acurácia entre as condições vem da dinâmica de resets, não de uma política aprendida.

**Massa nos atacantes** (mediana entre sementes, rodadas com atacantes reais): fixed 0,089, td3 0,096. Com ε = 2, os dois filtros dão peso apreciável aos atacantes, como o sanity antecipou.

## Achados de auditoria

1. **O IPM do código oficial é um update nulo** (`teste_unitario.txt`): a rede é posta em `old_weights` antes do `craft`, então o atacante envia o próprio modelo global. Os resultados de IPM do paper original não testam o IPM de Xie et al.
2. O IPM real (via o gancho do LMP, desvio declarado) é um ataque efetivo no código oficial: degrada o FedAvg em 5,9 p.p. com ε = 2 e em 63,5 p.p. com ε = 10 (sanity). Ele também desencadeia cascatas de reset no mecanismo de reinicialização do ambiente.

## Leitura para a tese

- A conclusão confirmatória do P2 ("fixed ≡ TD3") continua valendo **para LMP e EB** (B2.1).
- **Sob IPM real:**
  - não há equivalência estabelecida, por causa da variância induzida pelos resets;
  - não há evidência de contribuição do TD3: o Δ favorece o fixed, sem significância;
  - o mecanismo (política sem drift e independente do estado) se mantém.
- Formulação sugerida: "sob IPM, a política TD3 não aprende (drift ≪ ruído de exploração) e não há evidência de ganho sobre a ação fixa; a equivalência não pôde ser estabelecida devido a cascatas de reset do ambiente oficial".
- **Limitação:** n = 10, uma célula (MNIST, q = 0,5, ε = 2), métrica primária sensível a cascatas de reset longas.
