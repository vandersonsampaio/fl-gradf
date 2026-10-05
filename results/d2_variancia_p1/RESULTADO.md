# Resultado — D2: variância de execução dos números do P1

**Data:** 2026-10-05.
- Plano: `PLANO.md`, commit `7c0311b`.
- Código: tag `p1.0.0` (3beb0f8), via `git worktree`, sem alterações.
- Grade: 9 jobs (sementes 42–44 × 3 repetições), de 05/10 00h12 a 05h57. Sem falhas.
- A execução original do P1 conta como a 4ª repetição.
- Saídas: `analise.txt`, `sd_exec_por_celula.csv`, `delta_agregado_por_execucao.csv`, `grade_raw.csv`.

## Regra de leitura (PLANO §4): **CONCLUSÃO DO P1 ROBUSTA À VARIÂNCIA DE EXECUÇÃO**

- **Sinais:** em todas as 12 combinações execução × semente, GRADF − Random < 0, FedStrategist − Random < 0 e AdaAggRL − Random > 0.
- **Magnitude:** o SD_exec do Δ agregado fica abaixo de ½ |Δ publicado| nos três:

| sistema | Δ publicado (Tab. 4) | SD_exec do Δ agregado | ½ \|Δ\| | SD entre as 10 sementes do P1 |
|---|---|---|---|---|
| GRADF | −0,042 | **0,0177** | 0,021 | 0,0335 |
| FedStrategist | −0,044 | 0,0005 | 0,022 | 0,0184 |
| AdaAggRL | +0,030 | 0,0001 | 0,015 | 0,0271 |

## Variância por sistema (entre as 4 execuções, por célula × semente)

| sistema | SD médio | SD máx. | amplitude média | amplitude máx. |
|---|---|---|---|---|
| AdaAggRL | 0 | 0 | 0 | 0 |
| Random | 0,0001 | 0,002 | 0,0002 | 0,005 |
| Oracle | 0,0003 | 0,012 | 0,0006 | 0,023 |
| FedStrategist | 0,0007 | 0,013 | 0,0014 | 0,025 |
| **GRADF** | **0,093** | **0,31** | **0,19** | **0,63** |

## Leitura

- **AdaAggRL é determinístico** entre execuções. Random, Oracle e FedStrategist variam pouco: no máximo 1–2 p.p. numa célula, e só a ordem do RNG global os afeta.
- **O GRADF é fortemente não-determinístico célula a célula:** SD médio de 9,3 p.p., e a mesma célula com a mesma semente chega a variar **63 p.p.** entre execuções. A fonte é o DQN (TF) do seletor, como no B2.7.
- **No agregado sobre as 21 células, a variância cai muito:**
  - o SD_exec do Δ do GRADF é 1,8 p.p., ~metade da variação entre sementes (3,4 p.p.), e o sinal negativo se mantém em todas as execuções;
  - a conclusão do P1 (a seleção discreta não supera o Random; o AdaAggRL supera) **não depende da execução**.

## Nota de limitação para o P1 (pronta para a revisão)

> Os resultados do GRADF por célula (Tabela 3) têm variância de execução não reportada: o seletor DQN é não-determinístico entre execuções com a mesma semente (SD médio de 9,3 p.p. por célula; amplitude de até 63 p.p.). A comparação agregada da Tabela 4 é robusta a essa variância. Em três sementes e quatro execuções, a diferença GRADF − Random permaneceu negativa em todas as combinações, com desvio entre execuções (1,8 p.p.) menor que a metade do efeito publicado (−4,2 p.p.) e menor que a variação entre sementes (3,4 p.p.). FedStrategist, Random e Oracle variam ≤ 2,5 p.p. por célula; o AdaAggRL é determinístico.

## Pendente

- **"Depois do D1":** medir de novo após tornar o DQN determinístico (D1, ainda não implementado).
