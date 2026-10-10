# Plano — D2: variância de execução dos números do P1

**Status:** FINAL antes de qualquer run (hash em `PLANO.sha256`). Descritivo, com regra de leitura fixada agora.
**Data:** 2026-10-04
**Roadmap:** `references/roadmap_tese_gradf_v4_1.md` §4.1 (D2) e §6.4.
**Motivo:** o B2.7 mostrou que o DQN do GRADF v1 é não-determinístico entre execuções (~4 p.p. numa célula) e que o RNG global do framework depende da ordem de execução. Os números do P1 (Tabelas 3–5 de `docs/Rule Selection Fails Where Client Weighting Succeeds/Sampaio_Macedo_2026.pdf`) têm, portanto, uma variância de execução não reportada.

## 1. Pergunta

Rodando de novo **o mesmo código** do P1 (tag `p1.0.0`), com as **mesmas sementes** e na **mesma ordem**, quanto os números mudam? A conclusão do P1 sobrevive a essa variância? (GRADF e FedStrategist abaixo do Random; AdaAggRL acima; Tabela 4.)

## 2. Desenho

- **Código:** `git worktree` da tag `p1.0.0`, sem nenhuma alteração. `data/processed` (não versionado) entra por link simbólico.
- **Chamada:** `src.experiments.exp10_selector_comparison.run_selector_comparison_grid(seed=s, variant="b", root_size=100)`, com os 5 sistemas do P1 (random, oracle, gradf, fedstrategist, adaaggrl), nas 21 células e na ordem do próprio código, como no P1.
- **Sementes:** 42, 43 e 44 (3 das 10 do P1).
- **Repetições:** 3 por semente, cada uma num processo novo (9 processos, em paralelo na CPU). A execução original do P1 (`results/tables/exp10_FULL_variantb_raw.csv` na tag) conta como a 4ª execução.
- **Custo:** ~1,5 h por processo (21 células × pré-treino do GRADF + 5 sistemas); ~2–3 h de relógio.
- **"Depois do D1":** o D1 (determinismo do DQN e RNG isolado) ainda não foi implementado. A rodada "depois" fica para quando ele existir, como adendo. Aqui só se mede o **antes**.

## 3. Medidas

1. **Por sistema × célula × semente:** desvio-padrão da acurácia entre as 4 execuções (SD_exec) e amplitude máx. − mín.
2. **Δ agregado da Tabela 4,** por execução e semente: média sobre as 21 células de (sistema − Random), para GRADF, FedStrategist, AdaAggRL e Oracle. Depois:
   - **SD_exec** do Δ agregado, entre as 4 execuções de cada semente, em média nas 3 sementes;
   - **SD_sementes** do Δ agregado entre as 10 sementes do P1 original, como referência.

## 4. Regra de leitura (fixada agora)

- **"Conclusão do P1 robusta à variância de execução"** se, nas 12 combinações execução × semente (4 × 3):
  - o Δ agregado de **GRADF − Random** e de **FedStrategist − Random** for **< 0** em todas;
  - o Δ agregado de **AdaAggRL − Random** for **> 0** em todas;
  - e o SD_exec do Δ agregado for **< ½ |Δ publicado|** para os três: GRADF −0,042, FedStrategist −0,044, AdaAggRL +0,030.
- **Caso contrário:** "a variância de execução é da ordem do efeito", e a nota de limitação do P1 reporta as magnitudes, por sistema.
- **Reportados sem critério:** as células com maior SD_exec, por sistema; quais sistemas são determinísticos (SD_exec = 0); a razão SD_exec / SD_sementes.

## 5. Regras

- Nenhum número é inspecionado antes do fim. Análise pré-escrita (`scripts/d2_variancia_p1.py analisar`).
- O resultado alimenta só a nota de limitação do P1 (roadmap §6.4). Nenhum número do P1 é substituído.
