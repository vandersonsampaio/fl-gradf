# Plano — B2.8: teto oracular da seleção por rodada (eixo da granularidade)

**Status:** FINAL antes dos dados (hash em `PLANO.sha256`). Exploratório.
**Data:** 2026-09-30
**Roadmap:** `references/roadmap_tese_gradf_v3.md` §4.1 (ordem 6). Custo recalibrado: **não sai dos logs**, porque os contrafactuais por rodada não estão registrados; são runs novos, baratos e em CPU.
**Registro:** local, com hash; o commit fica a critério do autor.

## 1. Pergunta

O P1 atribuiu a diferença entre a seleção discreta de regras (GRADF, FedStrategist) e a ponderação por cliente (AdaAggRL) a "discreto contra contínuo". O P2 (B2.1–B2.3) mostra que o aprendizado não explica: o esqueleto fixo faz o trabalho. **A granularidade explica?** Ou seja, uma política que escolhesse a **melhor regra possível a cada rodada** alcançaria a filtragem por cliente do esqueleto?

## 2. Desenho

- **Oráculo guloso por rodada:**
  - a cada rodada, os updates (com ataque) são calculados uma vez;
  - cada uma das 7 regras do arsenal do exp10 (`fedavg`, `fedprox`, `median`, `trimmed_mean`, `fltrust`, `clustering`, `krum`) agrega esses updates;
  - o candidato de maior acurácia vira o novo global.

  A métrica é a mesma do C.0: acurácia média sobre os `X_test` dos 10 clientes.
- **É um limite superior**, não um método implantável: escolhe olhando o teste. Se **nem ele** alcança o esqueleto, a conclusão é forte.
- **Regime idêntico ao C.0, à ablação e ao exp9:** MNIST, 10 clientes, bizantinos [0, 1], 15 rodadas, root 100, **sementes 42–51**, 3 alphas × 7 ataques = 21 células, 210 runs.
- **Referências pareadas por semente** (dados que já existem):
  - melhor do esqueleto entre `fixed`, `sr_only` e `td3_ref` (ablação);
  - melhor regra fixa **estática** (exp9, root 100);
  - FedAvg sem ataque (C.0).
- Custo: ~0,68 s por rodada (medido), ≈ 36 min de processo; CPU, `nice 19`, 2 threads, em paralelo com o B2.3b na GPU.
- Script: `scripts/b28_oraculo_por_regra.py` (`run` e `analisar`).

## 3. Quantidades por célula (média pareada por semente, IC95)

- **Q** = oráculo por rodada − melhor esqueleto;
- **W** = melhor esqueleto − melhor regra estática;
- **G** = oráculo por rodada − melhor regra estática (quanto a escolha por rodada ganha sobre a escolha fixa).

## 4. Critério (fixado antes dos dados)

- **Células válidas:** 19 (exclui `fltrust_aligned` α = 0,05 e 0,1, artefato do ataque; C.0 `NOTAS.md` §2).
- **Células-alvo:** células válidas em que o esqueleto supera a melhor regra estática (**W com IC95 > 0**). É onde a ponderação por cliente "ganha" e a pergunta faz sentido.
- **Veredito:**
  - **"A granularidade explica"** se, em pelo menos metade das células-alvo, o oráculo por rodada fica **abaixo** do esqueleto (Q com IC95 < 0);
  - **"A granularidade não explica"** se, em pelo menos metade das células-alvo, ele fica **acima** (Q com IC95 > 0);
  - caso contrário, **inconclusivo**;
  - se não houver células-alvo, reportar como tal.
- Reportados sem critério: G por célula, frequência das regras escolhidas pelo oráculo, e as 2 células excluídas.

## 5. Limitações declaradas

- **Exploratório:** sementes gastas; referências da ablação e do exp9 já conhecidas.
- **Oráculo guloso:** ótimo por rodada, não necessariamente na trajetória. Mesmo assim, é um teto generoso para qualquer seletor implantável.
- **15 rodadas:** o gap mistura robustez e convergência (C.0 §6.3); o B2.7 trata do horizonte.
