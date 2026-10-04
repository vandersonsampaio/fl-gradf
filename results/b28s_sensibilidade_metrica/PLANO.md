# Plano — B2.8s: o veredito da granularidade (B2.8) depende da métrica?

**Status:** FINAL antes de qualquer run (hash em `PLANO.sha256`). Sensibilidade **post hoc**, motivada pela descoberta de que o teto oráculo ponderado por amostra subestima o teto sob a métrica uniforme (`results/c0b_espaco_h150/VERIFICACOES.md` §3).
**Data:** 2026-10-04
**Antecedente:** B2.8 (`results/b28_oraculo_por_regra/`). Veredito: "a granularidade explica" (8/14 células-alvo; robusto em α ≤ 0,1 com ataques de modelo).

## 1. Pergunta

A conclusão "nem a melhor regra por rodada alcança o filtro por cliente" se mantém quando a métrica deixa de ser a **média uniforme** das acurácias nos test sets dos clientes e passa a ser a **média ponderada pelo tamanho do test set** de cada cliente (≈ acurácia no teste agregado)?

**Por que isso importa:** a máscara binária do esqueleto equivale a pesos uniformes entre os honestos, e a métrica uniforme premia isso. No arsenal do oráculo por rodada, FedAvg e FedProx ponderam por amostra; as outras 5 regras não (inventário no `VERIFICACOES.md` do C0b e na resposta de 04/10).

## 2. Desenho

Rodar de novo todas as peças do B2.8 **gravando as acurácias por cliente**, para que as duas métricas saiam dos mesmos runs:
- **oráculo por rodada `oracle_u`:** escolhe a cada rodada a melhor das 7 regras pela métrica **uniforme**, como no B2.8;
- **oráculo por rodada `oracle_w`:** idem, mas escolhendo pela métrica **ponderada**, para que o oráculo seja ótimo na métrica em que é avaliado;
- **esqueleto:** `fixed`, `sr_only` e `td3_ref` (learner da ablação, como no B2.8);
- **regras estáticas:** FLTrust, Krum, Median e Trimmed-Mean, com os learners do exp9 (`InformedAttackedFederatedLearner` nos ataques informados), como no B2.8.

**Regime:** idêntico ao B2.8: MNIST, 10 clientes, bizantinos [0, 1], 15 rodadas, root 100, **sementes 42–51**, 21 células. Total: 9 sistemas × 21 células × 10 sementes = **1.890 runs**, um job por célula × semente, ~1,5–3 h de CPU com 10 processos.

**RNG:** `keras.utils.set_random_seed(seed)` antes de cada sistema, como no B2.7/C0b. O B2.8 original não ressemeava, então os números da métrica uniforme **não precisam reproduzir** o B2.8 bit a bit. A comparação entre métricas é **dentro desta grade**.

**Métricas** (do modelo global final, rodada 15):
- **uniforme:** média das acurácias dos 10 clientes nos seus `X_test`;
- **ponderada:** Σᵢ nᵢ·accᵢ / Σᵢ nᵢ, com nᵢ = tamanho do `X_test` do cliente i.

## 3. Quantidades e critério (fixados agora)

Para cada métrica m, com o oráculo correspondente (`oracle_u` para a uniforme, `oracle_w` para a ponderada), por célula (média pareada por semente, IC95):
- **Q** = oráculo − melhor esqueleto;
- **W** = melhor esqueleto − melhor regra estática;
- **G** = oráculo − melhor regra estática.

O melhor de cada grupo é escolhido por célula, na própria métrica. As células-alvo, o veredito e as 19 células válidas seguem o **PLANO §4 do B2.8**.

**Critério da sensibilidade:**
- **"Granularidade blindada"** se o veredito do B2.8 for o **mesmo** nas duas métricas **e**, entre as células-alvo comuns às duas, o sinal de Q com IC95 (abaixo / acima / inconclusivo) coincidir em **pelo menos 75%**.
- **"Depende da métrica"** se o veredito mudar. Nesse caso, o P2 precisa dizer que parte da vantagem do filtro por cliente vem de ponderar os honestos uniformemente sob a métrica uniforme.
- Caso intermediário (veredito igual, concordância < 75%): **"blindada no veredito, sensível por célula"**, reportando as células que mudam.

**Reportados sem critério:**
- se o veredito da métrica uniforme nesta grade reproduz o do B2.8 original;
- a frequência das regras escolhidas por `oracle_u` e `oracle_w` (FedAvg/FedProx ganham espaço na ponderada?).

## 4. Regras

- Código com hash em adendo antes do disparo; análise pré-escrita, rodada uma vez com a grade completa.
- Nenhuma acurácia é inspecionada antes do fim da grade.
- Logs não versionados.
