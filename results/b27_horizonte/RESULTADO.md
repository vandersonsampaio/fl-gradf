# Resultado — B2.7: o ganho do aprendizado cresce com o horizonte?

**Data:** 2026-10-02. Plano congelado em `PLANO.md` (hash em `PLANO.sha256`, commit `8ab747f`) antes da grade.
**Grade:** 80 jobs célula × semente (20 sistemas × 150 rodadas) + 30 tetos, de 01/10 12h39 a 02/10 15h56. Sem falhas.
**Saídas:** `analise.txt`, `delta_por_celula.csv`, `grade_raw.csv`, `tetos_raw.csv`, `verificacao.txt`.

## Resposta curta

**Não.** Pelo critério do plano (Δ > 0 com IC95 > 0 em ≥ 3 das 8 células), **nenhum agente se paga em nenhum horizonte**.

| agente | referência fixa | H = 15 | H = 50 | H = 150 | Δ médio sobre as células (15 → 50 → 150) |
|---|---|---|---|---|---|
| TD3 (AdaAggRL) | `fixed`, mesmo esqueleto | 0/8 | 0/8 | 1/8 | −1,65 → −1,12 → **+1,33** p.p. |
| LinUCB (FedStrategist b) | melhor `arm_rule` em retrospecto | 0/8 | 0/8 | 0/8 | −11,55 → −8,23 → −8,62 p.p. |
| DQN (GRADF v1) | melhor `arm_gradf` em retrospecto | 0/8 | 0/8 | 0/8 | −13,27 → −11,61 → −9,54 p.p. |

## Leitura por agente

**TD3: tendência positiva, concentrada em duas células e sem passar no critério.**
- A única célula que passa é `label_flipping` α 0,05 em H = 150: +6,65 p.p., IC95 (+2,84; +10,46). Em H = 15 e 50, a mesma célula estava em −5,8 e −5,3 p.p.
- A virada do Δ médio vem de 3 células:
  - `label_flipping` α 0,05: inclinação +5,3 p.p. por log H;
  - `label_flipping` α 0,1: +2,2, com H = 150 em +2,7 e IC95 cruzando zero;
  - `low_mag_backdoor` α 0,05: +2,4, com H = 150 em +3,1 e IC95 cruzando zero.
- Nas outras 5 células, Δ fica em ±0,1 p.p. (`gaussian_noise`, `krum_collusion`, `trim_attack`) ou oscila (`sign_flipping` α 0,05: −2,0 em H = 150).
- **Leitura:** há um indício de que o TD3 do esqueleto passa a usar o sinal com horizonte longo nos ataques em que o S_R não separa. É exploratório, 1/8 células, sem correção de multiplicidade. É compatível com o sd_estados crescente do B2.3b (A0 d). Merece confirmação em sementes novas antes de entrar no P2 como afirmação.

**LinUCB: muito abaixo do melhor braço fixo em todos os horizontes.**
- Δ entre −0,6 e −26,4 p.p. A diferença encolhe com H em 4 células e cresce em outras (`sign_flipping` α 0,05: −19,4 → −26,4).
- Contra o braço aleatório, Δ fica perto de zero (de −8 a +7 p.p.). O LinUCB não se distingue de escolher a regra ao acaso.

**DQN: muito abaixo do melhor braço fixo e, na maioria das células, abaixo do próprio braço aleatório.**
- Δ de −1,2 a −24,0 p.p. A inclinação por log H é positiva em 7 de 8 células (Δ médio −13,3 → −9,5), mas fica longe de zero.
- Contra o `rand_gradf`, Δ é negativo em 22 das 24 combinações célula × H. Em `low_mag_backdoor` α 0,05, chega a −18 a −23 p.p.: a política aprendida é **pior que escolher ao acaso**.

**Ressalva (prevista no plano):** o melhor braço é escolhido em retrospecto e é otimista para o fixo. Mas o LinUCB e o DQN perdem também para o braço aleatório ou empatam com ele, então a conclusão não depende dessa ressalva.

## Robustez contra convergência (pergunta do C0)

Os tetos sobem com o horizonte, sobretudo o oráculo FedAvg-8 em α baixo. Valores do oráculo **corrigidos** em 2026-10-02 (ver a correção no fim deste arquivo):

| α | FedAvg-10 sem ataque (15 → 50 → 150) | oráculo FedAvg-8 (15 → 50 → 150) |
|---|---|---|
| 0,05 | 85,3 → 88,5 → 89,5% | 77,1 → 84,0 → 86,5% |
| 0,1 | 85,5 → 88,5 → 89,4% | 79,0 → 82,5 → 85,0% |
| 0,5 | 90,5 → 91,6 → 91,9% | 89,9 → 91,1 → 91,7% |

- **O oráculo FedAvg-8 não converge em 15 rodadas com α ≤ 0,1.** Ele ganha 6,0–9,4 p.p. até H = 150. Por isso parte dos gaps negativos do C0 (métodos sob ataque acima do oráculo em 15 rodadas) é **artefato de convergência do teto**:
  - TD3 em `gaussian_noise` α 0,05: gap ao oráculo −3,2 → −1,4 p.p.
- Em outras células o gap negativo **persiste** com o horizonte:
  - TD3 em `krum_collusion` α 0,1: −4,5 → −4,4;
  - TD3 em `sign_flipping` α 0,1: −1,2 → −2,7.
  - Nelas, o esqueleto pondera os honestos melhor que a média uniforme, o que é consistente com o A0(a): massa ≈ 0 nos atacantes.
- **`label_flipping` α ≤ 0,1, a única região com espaço no C0, continua com gap grande em H = 150** (TD3: +18,0 p.p. ao oráculo em α 0,05; +10,9 em α 0,1). O espaço restante é de **robustez**, não de convergência.

## Verificações (`verificacao.txt`)

**1. Aninhamento:** idêntico (|Δ| = 0) para 7 dos 8 sistemas testados (`td3_ref`, `fixed`, `linucb`, `rand_rule`, `rand_gradf`, `arm_rule_median`, `arm_gradf_median`). O `dqn` diferiu 4,15 p.p. (H = 15) e 3,74 p.p. (H = 50).
- **Causa diagnosticada:** não é dependência do horizonte. Dois processos com configuração idêntica (15 rodadas, mesma ordem de sistemas) dão a mesma diferença de 4,15 p.p. no `dqn` e |Δ| = 0 nos demais.
- O **DQN do GRADF é não-determinístico de um run para outro**, mesmo com a semente fixada (provavelmente estado ou operações não determinísticas do TF no treino online do seletor).
- **Desvio declarado:** o plano (§6.1) previa refazer os horizontes como runs separados se o aninhamento falhasse. Isso não foi feito, porque a causa não é o aninhamento e runs separados não removeriam o não-determinismo.
- **Consequência:** o `dqn` carrega ruído extra entre runs (alguns p.p.), absorvido na variância entre sementes. Os Δ do DQN (−9 a −13 p.p. em média) são muito maiores que esse ruído; a conclusão não muda.

**2. Reprodução em H = 15 contra os runs antigos:**
- `td3_ref` e `fixed`: idênticos (80/80).
- `linucb`: média |Δ| 0,9 p.p. (máx. 23,6).
- `dqn`: média |Δ| 17,4 p.p. (máx. 61,9).
- `rand_gradf`: média |Δ| 1,6 p.p.
- Braços contra o exp9: média |Δ| ≤ 2,2 p.p.
- As diferenças vêm da dependência de ordem do RNG global (corrigida no B2.7, PLANO §6.3) e, no DQN, do não-determinismo acima. **Os números de DQN/GRADF dos experimentos antigos (exp10 etc.) têm, portanto, variância de execução não reportada.**

## Implicações

1. **P2:** "o aprendizado não se paga" se estende a horizontes 10× maiores (150 rodadas) para LinUCB e DQN, contra a versão fixa no mesmo espaço de ação e contra o aleatório. Para o TD3 do esqueleto, a afirmação fica **delimitada**: há uma tendência positiva em `label_flipping` e `low_mag_backdoor`, que passa no critério em 1 de 8 células em H = 150.
2. **C0:** parte dos gaps negativos era convergência do teto; o espaço em `label_flipping` α ≤ 0,1 é de robustez e persiste.
3. **Reprodutibilidade:** o DQN do GRADF v1 é não-determinístico entre runs, e o RNG global do framework depende da ordem de execução. Isso deve ser corrigido antes do GRADF-v2 (P3) e declarado como limitação dos resultados do P1.

## Correção do teto oráculo FedAvg-8 (2026-10-02)

- **Bug:** `run_ceiling` gravava o oráculo FedAvg-8 quando `round_num + 1 ∈ {15, 50, 150}`, mas o `train` numera as rodadas a partir de 1. Os valores rotulados H = 15, 50 e 150 eram os das rodadas 14, 49 e 149.
- **O teto FedAvg-10 e todos os sistemas estavam corretos**, porque usam `results[H − 1]`.
- **Correção:** `if round_num in HORIZONS` em `scripts/b27_horizonte.py`. Os 30 jobs de teto foram refeitos em `raw_teto_corrigido/`; os arquivos com o bug ficam em `raw/teto_*` e a análise anterior em `v1_teto_bug_*`.
- **Efeito:**
  - o FedAvg-10 sai idêntico (|Δ| = 0);
  - o oráculo muda +0,25 p.p. em média em H = 15 (máx. 0,56), +0,05 em H = 50 e +0,01 em H = 150;
  - os gaps ao oráculo mudam no máximo 0,46 p.p.;
  - **os Δ dos agentes e todos os veredictos do critério ficam idênticos.**
- **Verificação:** o oráculo corrigido em H = 15 fica a 0,05 p.p. (média; máx. 0,13) do teto oráculo do C0, que foi calculado em outro processo.
