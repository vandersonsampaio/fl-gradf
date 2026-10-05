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

> Os resultados do GRADF por célula (Tabela 3) têm variância de execução não reportada: o seletor DQN é não-determinístico entre execuções com a mesma semente (SD médio de 9,3 p.p. por célula; amplitude de até 63 p.p.). A comparação agregada da Tabela 4 é robusta a essa variância: em três sementes e quatro execuções, a diferença GRADF − Random permaneceu negativa em todas as combinações. Como cada semente do P1 corresponde a uma execução, a variância de execução já está contida na variação entre sementes usada nos testes da Tabela 4; os testes permanecem válidos, apenas mais ruidosos que o necessário. FedStrategist, Random e Oracle variam ≤ 2,5 p.p. por célula; o AdaAggRL é determinístico.

**O que sustenta a conclusão** é o **sinal negativo de GRADF − Random em todas as 12 combinações**. A razão SD/efeito é secundária e, no GRADF, tem folga pequena: SD_exec de 1,77 p.p. contra o limiar de ½|Δ| = 2,1 p.p., com só 3 sementes.

**Uso da nota:**
- não contatar o editor agora, porque a conclusão do P1 não muda;
- **incluir a nota na resposta à revisão, mesmo que os revisores não perguntem**;
- no P2, a limitação do DQN no B2.7 cita estes números em vez de "alguns p.p.".

## Afirmações do P1 que dependem do GRADF por célula ou por ataque (releitura de 05/10)

Nenhuma frase do texto destaca o GRADF numa célula específica ("vence em X", "colapsa em Y"). Estes pontos, porém, usam valores do GRADF por célula ou por ataque e herdam a variância de execução. Pelo D2, o SD de execução esperado numa média de 10 sementes é de ~2,9 p.p. por célula (9,3/√10) e de ~1,7 p.p. por ataque (30 célula × semente):

| onde | o que depende do GRADF por célula/ataque | risco | ação na revisão |
|---|---|---|---|
| **Tabela 3**, coluna GRADF | acurácia média por célula (10 sementes) | ~2,9 p.p. por célula de ruído de execução | nota de rodapé: valores do GRADF por célula têm variância de execução; não comparar células isoladas |
| **Figura 2** e texto ("*In terms of headroom count, … none by GRADF*") | contagem de células com headroom capturado (critério por célula) | uma célula pode entrar ou sair da contagem por ruído de execução | trocar por "nenhuma de forma consistente", ou recalcular a contagem com médias entre execuções |
| **Tabela 5**, colunas do GRADF, e texto ("*Neither of the two surpasses the Random-selector in any individual attack type*") | Δ, p e d por tipo de ataque | o maior Δ do GRADF por ataque é +0,014 (`sign_flipping`, p = 0,40). A afirmação é provável, mas não está garantida contra ~1,7 p.p. de ruído | manter, com a ressalva de que os valores por ataque do GRADF incluem variância de execução; os testes continuam válidos |
| **Figura 3** (por tipo de ataque) | posição do GRADF por ataque | igual à Tabela 5 | mesma ressalva |

## D2 encerrado

A remedição "depois do D1" **não** é mais item do P1: ela só mostraria que o D1 funcionou. Ela vira o **critério de aceitação do D1 no P3**: *|Δ| = 0 entre execuções com a mesma semente para o GRADF v1 determinístico.*
