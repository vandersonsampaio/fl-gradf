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
- **O oráculo FedAvg-8 não é teto com α ≤ 0,1, mesmo em H = 150.**
  - Em 13 das 19 células (todas as 6 de α 0,1), o melhor existente fica acima do oráculo; em α 0,1 a diferença é de 3,9 a 5,6 p.p.
  - Contra o FedAvg-10 sem ataque, os gaps em α 0,1 ficam entre −1,2 e +0,5 p.p.
  - Com o A0(a) (massa ≈ 0 nos atacantes), a leitura é que **ponderar os honestos melhor que a média uniforme** (filtro por cliente e regras robustas) rende mais que excluir os atacantes e fazer a média do resto. Não é aproveitamento dos atacantes.
- **Confirma o Portão C0 em H = 150:** em acurácia, não sobra espaço contra o melhor existente. O diferencial do P3 fica em R3/R4/R5 (privacidade, custo, robustez adaptativa). O R1 só tem alvo contra o esqueleto de referência e o AdaAggRL.

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

1. **P2:** com horizonte suficiente, o melhor método existente fecha o espaço restante em todas as células, inclusive `label_flipping` α ≤ 0,1, que era o espaço do C0. O espaço continua grande contra o AdaAggRL e o esqueleto de referência nos ataques de rótulo e backdoor.
2. **P3 (R1, Portão C0b):** não há alvo de acurácia contra o melhor global. O diferencial é privacidade (os dois sinais fortes têm custo), custo computacional e robustez adaptativa (`fltrust_aligned` inverte o cos_server; `low_mag_backdoor` escapa dos dois sinais).
3. **Direção 1 (seleção de sinais):** headroom real, mas pequeno (+1,2 p.p.) e decrescente com o horizonte. Um seletor não aprendido deve bastar (R7).

## Ressalvas

- Descritivo; o melhor existente e o melhor sinal por célula são escolhidos em retrospecto (otimista para o existente, ou seja, conservador para "espaço").
- MNIST logístico, 10 clientes, 2 bizantinos, root 100; as regras estáticas são aplicadas como no B2.7.
- Antes da grade houve exposição de acurácias de uma célula da semente 72 em H = 15 (declarada no ADENDO1).
