# Resultado — C0b: espaço restante em H = 150 contra o melhor método existente

**Data:** 2026-10-04.
- Plano: `PLANO.md`, commit `d676ec3`.
- Adendo 1 (código, registro por cliente, verificações): commit `183a585`.
- Grade: 190 jobs de célula (19 células × sementes 72–81 × 8 sistemas, horizontes 15/50/150 aninhados) + 30 tetos, de 03/10 09h25 a 04/10 16h58. Sem falhas.
- Saídas: `analise.txt`, `espaco_restante.csv`, `grade_raw.csv`, `tetos_raw.csv`, `concordancia_sinais.csv`, `raw/` (inclui `scores_*.csv.gz`, com S_R, cos_server, máscara e peso por cliente e rodada).

## (i) Mapa de espaço restante: **0/19 células com espaço contra o melhor existente, em qualquer horizonte**

Critério do C0: gap ao oráculo FedAvg-8 > 2 p.p. com IC95 > 0.

| horizonte | contra o melhor existente (8 sistemas) | contra o esqueleto de referência (`sr_only`) |
|---|---|---|
| H = 15 | **0/19** | 6/19 |
| H = 50 | **0/19** | 5/19 |
| H = 150 | **0/19** | 5/19 |

- **Nenhuma célula tem espaço contra o melhor método existente.** A única que chega perto é `label_flipping` α 0,05: o melhor é o FLTrust (85,1%), a +1,3 p.p. do oráculo (IC95 +0,7 a +1,9), abaixo do limiar de 2 p.p.
- **Contra o esqueleto de referência, o espaço é grande** em 5 células, todas de rótulo ou backdoor:
  - `label_flipping` α 0,05: +17,2 p.p.;
  - `label_flipping` α 0,1: +5,9;
  - `label_flipping` α 0,5: +20,5;
  - `low_mag_backdoor` α 0,05: +3,3;
  - `sign_flipping` α 0,05: +2,0.
- **O oráculo FedAvg-8 (pré-registrado) subestima o teto com α ≤ 0,1** (corrigido em 2026-10-04; ver a seção de sensibilidade e `VERIFICACOES.md`).
  - Em 13 das 19 células (todas as 6 de α 0,1), o melhor existente fica acima dele; em α 0,1 a diferença é de 3,9 a 5,6 p.p.
  - **Causa:** o `FedAvgStrategy` do framework pondera **por tamanho de amostra** (de ~10² a ~10⁴ exemplos por cliente com α 0,1), enquanto a métrica é a **média uniforme** das acurácias nos test sets dos clientes. O oráculo com ponderação uniforme dos 8 honestos reproduz o melhor método (ex.: `sr_bin` sob `gaussian_noise` α 0,1: 89,5% contra 89,5%).
  - **Correção da leitura anterior:** o que leva ao teto é excluir os atacantes e **ponderar os honestos de forma uniforme**, não "ponderar os honestos melhor que a média uniforme", como estava escrito. Não é aproveitamento dos atacantes (massa ≈ 0 em 4 das 6 células de α 0,1; a exceção é `low_mag_backdoor`, `massa_atacantes_a0.1.csv`).
  - Não há bug no código do oráculo: com a máscara toda em 1, ele reproduz exatamente o FedAvg-10 (`VERIFICACOES.md` §1).
- **Portão C0 em H = 150:**
  - pelo critério pré-registrado (oráculo por amostra), 0/19;
  - pela sensibilidade com o oráculo uniforme, **1/19: `label_flipping` α 0,05**, com espaço de +2,8 p.p. (IC95 +2,2 a +3,4) contra o FLTrust.
  - Fora essa célula, não sobra espaço contra o melhor existente. O diferencial do P3 fica em R3/R4/R5 (privacidade, custo, robustez adaptativa), com um alvo estreito de R1 em `label_flipping` α 0,05.

## (iii) Melhor variante do esqueleto contra a melhor regra estática (H = 150)

- **A variante vence em 12 das 19 células, com IC95 > 0,** por +0,14 a +2,7 p.p. São sobretudo os ataques de modelo (`gaussian_noise`, `krum_collusion`, `trim_attack` e `low_mag_backdoor` α 0,1). A melhor variante costuma ser `sr_b025` ou `sr_bin`.
- **A regra vence em 1:** `trim_attack` α 0,1, Clustering, por −0,12 p.p.
- **Empate (IC95 contém 0) em 6:** `label_flipping` α 0,05 e 0,1, `sign_flipping` α 0,05 e 0,5, `low_mag_backdoor` α 0,05 e `fltrust_aligned` α 0,5. Nos ataques de rótulo, a melhor regra (FLTrust ou Clustering) está à frente na média, sem significância.
- **Leitura:** a granularidade por cliente, com a variante ajustada (limiar b = 0,25 ou máscara binária), é o melhor existente na maioria das células, mas por margens pequenas (≤ 2,7 p.p.). O FLTrust é praticamente igual ao melhor sinal na média geral: 87,35% contra 87,55% do `cosserver_only`.

## (ii) Teto da seleção de sinais em H = 150

Média sobre as 19 células, por semente (n = 10), em %:

| estimativa | acurácia | IC95 |
|---|---|---|
| máx. por semente (viesado) | 89,18 | 88,83–89,52 |
| escolha do sinal por célula, in-sample | 88,92 | 88,48–89,36 |
| **escolha por célula, LOSO (referência)** | **88,79** | 88,37–89,21 |
| `cosserver_only` | 87,55 | 87,02–88,09 |
| FLTrust | 87,35 | 87,16–87,55 |
| `sr_b025` | 87,10 | 87,01–87,19 |
| `sr_only` | 86,04 | 85,58–86,49 |
| `sr_bin` | 85,39 | 84,83–85,96 |
| Clustering / Trimmed-Mean / Median | 82,46 / 75,58 / 74,41 | — |

- **Ganho da escolha por célula (LOSO) sobre o melhor sinal único por semente: +1,24 p.p. (IC95 +0,98 a +1,49).** Em 15 rodadas (A0, B2.6) era +1,75 p.p. **O headroom da direção 1 do P3 encolhe com o horizonte e é modesto.**
- O cos_server é escolhido em 10/19 células (`label_flipping` e `sign_flipping` em todos os α, `low_mag_backdoor` com α ≤ 0,1, `krum_collusion` e `trim_attack` com α 0,05), e o S_R em 9 (ruído e ataques de modelo, sobretudo com α 0,1 e 0,5).

## Concordância entre sinais por cliente (registro do `sr_only`, 10 sementes, rodadas ≥ 2)

Confirma o A0(c), que tinha 1 semente:
- **Os sinais são quase ortogonais:** o Spearman médio por rodada entre S_R e cos_server vai de −0,13 a +0,40.
- **As AUCs são complementares:**
  - o **S_R** separa `gaussian_noise` (0,99–1,0), `krum_collusion` (0,86–0,94) e `trim_attack` (0,80–0,90);
  - o **cos_server** separa `label_flipping` (0,81–0,98), `sign_flipping` (0,86–0,96), `trim_attack` (0,87–0,95) e `krum_collusion` (0,82–0,89);
  - **nenhum dos dois separa `low_mag_backdoor`** (AUC ≤ 0,60);
  - no `fltrust_aligned`, o cos_server é **invertido** (AUC 0,001), como esperado para um ataque desenhado contra o sinal do servidor (R5).
- **Leitura:** um seletor não aprendido baseado na separação de cada sinal tem base empírica. O teto que ele pode capturar é o do item (ii): ~+1,2 p.p.

## Implicações

1. **P2:** com horizonte suficiente, o melhor método existente fecha o espaço restante em quase todas as células, inclusive `label_flipping` α 0,1. A exceção é `label_flipping` α 0,05: +2,8 p.p. contra o oráculo uniforme (sensibilidade). O espaço continua grande contra o AdaAggRL e o esqueleto de referência nos ataques de rótulo e backdoor (9/19 células contra o oráculo uniforme).
2. **P3 (R1, Portão C0b):** o alvo de acurácia contra o melhor global é **estreito**: só `label_flipping` α 0,05 (~+2,8 p.p.). O diferencial principal é privacidade (os dois sinais fortes têm custo), custo computacional e robustez adaptativa (`fltrust_aligned` inverte o cos_server; `low_mag_backdoor` escapa dos dois sinais).
3. **Teto do C0/B2.7/C0b em α ≤ 0,1:** o oráculo FedAvg-8 por amostra não deve ser usado como teto nessa métrica. Usar o oráculo uniforme, ou trocar a métrica por uma ponderada por amostra, de forma coerente, nos próximos experimentos.
4. **Direção 1 (seleção de sinais):** headroom real, mas pequeno (+1,2 p.p.) e decrescente com o horizonte. Um seletor não aprendido deve bastar (R7).

## Sensibilidade post hoc: tetos com ponderação uniforme (2026-10-04)

Declarada como post hoc em `VERIFICACOES.md` §3. Script `scripts/c0b_sensibilidade_oraculo_uniforme.py`, saídas em `sensibilidade_oraculo_uniforme/`. É o mesmo código do `run_ceiling`, com a ponderação do FedAvg uniforme, nas sementes 72–81.

| α | oráculo-8 por amostra (pré-reg.) | **oráculo-8 uniforme** | FedAvg-10 por amostra | FedAvg-10 uniforme |
|---|---|---|---|---|
| 0,05 | 86,46% | **87,92%** | 89,47% | 90,44% |
| 0,1 | 84,99% | **89,46%** | 89,39% | 91,23% |
| 0,5 | 91,67% | **91,74%** | 91,93% | 92,07% |

(H = 150.)

| critério do C0 (gap > 2 p.p., IC95 > 0) | H = 15 | H = 50 | H = 150 |
|---|---|---|---|
| espaço contra o **melhor existente** (oráculo uniforme) | 0/19 | 1/19 | **1/19** (`label_flipping` α 0,05: +2,79 p.p., IC95 +2,19 a +3,39) |
| espaço contra o **esqueleto de referência** (oráculo uniforme) | 7/19 | 9/19 | 9/19 |

- **Em α 0,1, o melhor existente atinge o oráculo uniforme** em todas as células (gaps de −1,1 a +0,6 p.p.): excluir os atacantes e ponderar os honestos por igual é exatamente o que os melhores métodos fazem.
- **Em α 0,05, a única célula aberta é `label_flipping`**, em que o melhor existente (FLTrust, 85,1%) fica 2,8 p.p. abaixo do oráculo uniforme (87,9%).
- O FedAvg-10 uniforme fica 0,7–5,3 p.p. acima do melhor existente em α ≤ 0,1. É o ganho de usar também os dados dos 2 clientes que são atacantes, o que nenhuma defesa pode recuperar sem aproveitar os atacantes. Por isso ele não é o teto relevante para o critério.

## Ressalvas

- Descritivo; o melhor existente e o melhor sinal por célula são escolhidos em retrospecto (otimista para o existente, ou seja, conservador para "espaço").
- MNIST logístico, 10 clientes, 2 bizantinos, root 100; as regras estáticas são aplicadas como no B2.7.
- O teto pré-registrado (oráculo por amostra) subestima o teto real em α ≤ 0,1. A sensibilidade com o oráculo uniforme é post hoc, motivada por uma verificação feita depois dos resultados.
- Antes da grade houve exposição de acurácias de uma célula da semente 72 em H = 15 (declarada no ADENDO1).
