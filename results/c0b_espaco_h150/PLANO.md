# Plano — C0b: espaço restante em H = 150 contra o melhor método existente

**Status:** FINAL antes de qualquer run (hash em `PLANO.sha256`). **Descritivo**, com critério fixado agora. O código ainda não existe: ele será escrito, testado e terá o hash registrado em adendo **antes** do disparo, sem mudar nada deste plano.
**Data:** 2026-10-02
**Roadmap:** `references/roadmap_tese_gradf_v4_1.md` §5.0 (C0b) e §7.2 (item 2). Define os alvos de R1 no P3.

## 1. Motivo

O C0 tinha dois defeitos:
1. comparou o oráculo com o `sr_only`, a **pior** variante do esqueleto (B2.6);
2. usou **15 rodadas**, em que o teto não converge: o oráculo FedAvg-8 ganha 8–10 p.p. até H = 150 com α ≤ 0,1 (B2.7).

O B2.7 indica que, em H = 150, uma regra estática fecha `label_flipping` α ≤ 0,1.

## 2. Grade

- **Células:** as **19 válidas** do C0 (3 α × 7 ataques, menos `fltrust_aligned` em α 0,05 e 0,1, artefato do ataque).
- **Horizontes aninhados** 15 / 50 / **150**, como no B2.7. A primária é H = 150.
- **Sementes:** **72–81** (novas, reservadas no roadmap para o C0b).
- **Regime:** idêntico ao C0/B2.6/B2.7: MNIST logístico, 10 clientes, bizantinos [0, 1], root 100.
- **Sistemas (8):**
  - **variantes do esqueleto** (learner do B2.6, `scripts/b26_decomposicao.py`): `sr_only` (referência), `sr_bin`, `sr_b025`, `cosserver_only`;
  - **regras estáticas** (aplicadas como no B2.7, `PlainRuleLearner` de `scripts/b27_horizonte.py`: mesmos updates, `server_update` no root a cada rodada): FLTrust, Trimmed-Mean, Clustering, Median.
- **Tetos** por α × semente: FedAvg-10 sem ataque e oráculo FedAvg-8 (só os 8 honestos, avaliado nos test sets dos 10), como no B2.7.
- **RNG:** `keras.utils.set_random_seed(seed)` antes de cada sistema e de cada teto (correção do B2.7, PLANO §6.3).
- **Total:** 19 × 10 × 8 = 1.520 runs + 60 tetos = **~1.580 runs**, ~25–35 h de CPU com 10 processos (`nice 19`, 2 threads, sem GPU).

## 3. Critério e saídas (fixados agora)

**Métrica:** acurácia do modelo global ao fim da rodada H, média sobre os test sets dos 10 clientes (a mesma do C0). **Unidade:** semente, n = 10.

**(i) Mapa de espaço restante em H = 150 (principal):**
- **Melhor método existente por célula** = o sistema com maior acurácia média entre os 8, escolhido em retrospecto. É otimista para o existente, ou seja, conservador para "espaço".
- **gap_global** = oráculo FedAvg-8 − melhor existente, pareado por semente.
- **Célula com espaço:** gap_global > **2 p.p.** com IC95 inteiro acima de 0 (o critério do C0).
- **Também reportados:**
  - o mesmo gap com o FedAvg-10 como teto;
  - **gap contra o esqueleto de referência** (`sr_only`);
  - os gaps em H = 15 e 50 (tendência).

**(ii) Teto da seleção de sinais em H = 150** (repete o A0(b) com os dados do C0b), média sobre as 19 células por semente:
- max por semente entre `sr_only` e `cosserver_only` (viesado para cima);
- escolha do sinal por célula, in-sample;
- escolha por célula deixando a semente de fora (LOSO, **referência**);

tudo comparado ao melhor sinal único por semente e a cada sistema.

**(iii) Tabela esqueleto × regras estáticas:** por célula e H, a melhor variante do esqueleto contra a melhor regra estática (Δ, IC95).

## 4. Verificação

Repetir um job (célula `label_flipping` α 0,05, semente 72) em dois processos. As acurácias dos 8 sistemas devem ser idênticas. Esse teste também cobre o determinismo, já que o C0b não usa o DQN.

## 5. Leitura (expectativa do roadmap, não critério)

- **Contra o melhor global:** quase nenhuma célula com espaço, o que confirma o Portão C0. Nesse caso, o diferencial do P3 fica em R3/R4/R5.
- **Contra o esqueleto de referência e o AdaAggRL:** espaço grande em ataques de rótulo e backdoor.
- **O teto da seleção de sinais em H = 150** dimensiona a direção 1 do P3.

## 6. Regras

- Análise pré-escrita (script com hash no adendo), rodada uma vez com a grade completa.
- Nenhuma acurácia é inspecionada antes do fim da grade. O monitoramento reporta só saúde e progresso.
- Se o B2.7c for disparado (condicional, B2.7a/b), o C0b é pausado e retomado depois. O lançador é retomável.
- Logs não versionados.
