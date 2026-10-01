# Plano — B2.7: o ganho do aprendizado cresce com o horizonte? (15 / 50 / 150 rodadas)

**Status:** FINAL antes da grade (hash em `PLANO.sha256`). **Exploratório.** Mudanças depois disso só como adendo datado e hasheado.
**Data:** 2026-10-01
**Roadmap:** `references/roadmap_tese_gradf_v4.md` §4.1 (B2.7, prioridade 1): lacuna 0/78 da SLR; separar robustez de convergência nos gaps do C0.
**Código:** `scripts/b27_horizonte.py` (reusa sem alterar `scripts/frente1_ablacao_adaaggrl.py`, `src/experiments/exp10_selector_comparison.py`, `src/experiments/exp1_baseline.py` e `src/fl/gradf_learner.py`).

## 1. Pergunta

O ganho de cada agente sobre a **versão fixa no mesmo espaço de ação** (mesmo sinal, mesma granularidade, mesma memória) cresce com o horizonte?

## 2. Agentes e versões fixas

| agente | versão fixa | referência extra |
|---|---|---|
| **TD3** (`td3_ref`: AdaAggRL, learner da ablação, modo `td3`) | `fixed`: mesmo esqueleto, ação no centro (não a melhor variante do B2.6, para isolar só o aprendizado) | — |
| **LinUCB** (`linucb`: FedStrategist, variante **(b)**, detecção compartilhada do GRADF) | **melhor braço fixo por célula** entre os 7 `arm_rule_<r>` | `rand_rule`: regra uniforme por rodada |
| **DQN** (`dqn`: GRADF v1, pré-treinado com `train_gradf_models`) | **melhor braço fixo por célula** entre os 7 `arm_gradf_<r>` | `rand_gradf`: `RandomSelector`, como no exp10 |

- **Arsenal:** as 7 regras do exp10 (`fedavg`, `fedprox`, `median`, `trimmed_mean`, `fltrust`, `clustering`, `krum`).
- **Mesmo espaço de ação, por agente.** O LinUCB e o DQN aplicam a regra escolhida por caminhos diferentes, e cada braço fixo segue o caminho do seu agente:
  - **LinUCB:** os braços aplicam a regra exatamente como o `FedStrategistGridLearner` (mesmos updates via `compute_param_updates_auto`, `server_update` no root a cada rodada, `_make_arsenal_strategy(regra, n_byz)`), só sem detecção e sem bandit.
  - **DQN:** a regra passa pelo pipeline do GRADF (hardening, camadas 1–5), então os braços são o próprio `GRADFFederatedLearner` com `FixedActionSelector(regra)`.
  - Por isso há **14 braços** (7 + 7), e não 7. **Desvio declarado** em relação ao desenho aprovado: sem ele, a comparação misturaria aprendizado com o efeito do pipeline.
- **O melhor braço é escolhido em retrospecto**, nas mesmas sementes, por célula e por horizonte (maior média). É uma **referência otimista** para o fixo:
  - se o agente ainda assim superar, o resultado é forte;
  - se empatar, a leitura carrega essa ressalva.

## 3. Horizontes

- **H ∈ {15, 50, 150}**, **aninhados**: cada sistema roda 150 rodadas, e a acurácia em H é a do modelo global ao fim da rodada H.
- Nada nos learners depende de `n_rounds` além do laço de treino, então o resultado é idêntico a rodar H rodadas. Isso é **verificado** (§6, item 1).
- Todos os runs são **novos**, com a mesma versão do código; nada é reaproveitado dos runs antigos.

## 4. Células, sementes e regime

**8 células:**

| célula | perfil |
|---|---|
| `label_flipping` α 0,05 e 0,1 | espaço no C0; o S_R não separa |
| `sign_flipping` α 0,05 e 0,1 | oráculo por regra ≥ esqueleto |
| `gaussian_noise` α 0,05 e `krum_collusion` α 0,1 | granularidade robusta |
| `low_mag_backdoor` α 0,05 | o cos_server vence |
| `trim_attack` α 0,5 | controle no teto |

- **Sementes:** 42–51 (exploratórias; consistência com o C0).
- **Regime:** idêntico ao C0/exp10 (MNIST logístico, 10 clientes, bizantinos [0, 1], root 100).
- **Tetos** por α × semente × H, só por α (não por ataque):
  - FedAvg-10 sem ataque, como no C0;
  - oráculo FedAvg-8 (só os 8 honestos, avaliado nos test sets dos 10, como no `c0_teto_oraculo`).
  - Cobre os 3 α das células.

## 5. Critério (exploratório, por agente)

- **Métrica:** acurácia final no horizonte H (média sobre os test sets dos 10 clientes, como no C0).
- **Unidade:** semente, n = 10.
- **Δ = agente − versão fixa**, pareado por semente; IC95 t.
- **"O aprendizado se paga no horizonte H"** se Δ > 0 com IC95 inteiro acima de 0 em **pelo menos 3 das 8 células**. Avaliado para cada agente (3) e cada H (3).
- **Sem correção de multiplicidade** (3 × 3 avaliações), aceitável num exploratório; declarado.
- **Também reportados:**
  - a **tendência** de Δ ao longo de H (Δ médio sobre as células e inclinação por log H, por célula);
  - Δ contra o braço aleatório;
  - gaps aos dois tetos por H;
  - qual braço é o melhor por célula e H.
- **Leitura para o C0:** se os gaps do C0 forem de convergência, eles encolhem com H em todos os sistemas (inclusive os fixos). Se forem de robustez, persistem.

## 6. Verificações (descritivas)

1. **Aninhamento:** na célula `label_flipping` α 0,05, semente 42, rodar com `n_rounds` = 15 e 50 deve reproduzir exatamente (|Δ| < 1e-9) o run de 150 em H = 15 e 50, em 8 sistemas. Se não reproduzir, os horizontes viram runs separados (adendo).
2. **Reprodução em H = 15** contra os runs antigos: `td3_ref` (exp10 AdaAggRL), `fixed` (ablação), `linucb` (exp10 FedStrategist b), `dqn` (exp10 GRADF) e `rand_gradf` (exp10 Random). Reporta-se o máx. |Δ|. Os `arm_rule_*` são comparados ao exp9, que agrega por outro caminho, então só se espera proximidade. Diferenças não invalidam a grade (todos os sistemas usam a mesma versão), mas são reportadas.

3. **RNG global (achado do smoke, corrigido antes do congelamento):** o `seed` dos learners do framework **não** semeia o `np.random` global, que controla a permutação do treino local e o ruído DP. Num processo com vários sistemas, cada um herdava o estado deixado pelo anterior. Só o `td3_ref`/`fixed` escapavam, porque o extrator chama `keras.utils.set_random_seed`. Por isso:
   - o B2.7 chama `keras.utils.set_random_seed(seed)` antes do pré-treino e antes de **cada** sistema e teto;
   - teste com dois processos em ordens opostas: `linucb`, `rand_gradf` e `arm_rule_median` ficam **idênticos**;
   - o `dqn` ainda varia (0,86 p.p. no teste) com a posição, por estado interno do TF. Na grade, a ordem dos sistemas é **fixa e igual** em todas as células e sementes, então os resultados são determinísticos e pareados.
   - **Consequência para os runs antigos** (exp9/exp10/C0): os números dependem da ordem do laço. Por isso os sistemas não-TD3 do B2.7 não reproduzem exatamente o exp10 em H = 15. No smoke (célula `label_flipping` α 0,05, semente 42, antes da correção): `td3_ref`, `fixed`, `arm_rule_median` e `arm_rule_krum` com |Δ| = 0; `linucb` 1,1 p.p.; `dqn` 1,6 p.p.; `rand_gradf` 5,6 p.p.; tetos 0,1–0,6 p.p.

## 7. Custo e execução

- **Por célula × semente:** 20 sistemas × 150 rodadas + pré-treino do GRADF. No smoke (15 rodadas, CPU disputada com o B2.5): ~8 s/rodada para os 2 sistemas com inversão, ~1,1 s/rodada para os outros 18 e 208 s de pré-treino.
- **Estimativa:** ~5.700 s de processo por célula × semente, ~125 h de processo no total. Com 10 processos (20 núcleos, `nice 19`, 2 threads, a GPU ocupada pelo B2.5), **~25–35 h de relógio**. Os tetos (30 jobs) somam < 1 h.
- **Calibração:** com o primeiro lote. Se passar de 40 h, o corte declarado é rodar H = 150 só nas sementes 42–46 (adendo antes de ver resultados).
- **Lançador:** `scripts/run_grid_b27.sh` (retomável, um job por célula × semente e por α × semente para os tetos).

## 8. Regras

- Análise pré-escrita: `python -m scripts.b27_horizonte analisar` (e `verificar`), rodada com a grade completa.
- Nenhuma acurácia é inspecionada antes do fim da grade. O monitoramento reporta só saúde e progresso.
- Jobs que falharem por erro de ambiente são repetidos com a mesma semente e registrados.
- `scripts/frente1_ablacao_adaaggrl.py` é dependência e **não está versionado**. O hash dele entra em `PLANO.sha256`.
