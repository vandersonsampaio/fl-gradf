# Resultado — B2.7a/b: o TD3 em H = 150 depende do estado ou só achou uma constante melhor?

**Data:** 2026-10-03.
- Plano: `PLANO.md`, commit `d676ec3`.
- Adendo 1 (código do B2.7a): commit `53c4722`.
- Adendo 2 (B2.7b explicativo, escrito depois de ver o B2.7a): commit `14dbefd`.

**Grades:**
- B2.7a: 30 runs, de 02/10 20h36 a 22h19.
- B2.7b: 120 runs, de 02/10 23h03 a 03/10 04h20.
- Sem falhas.

**Saídas:** `b27a_analise.txt`, `b27a_por_run.csv`, `a_td3.csv`, `b27b_analise.txt`, `b27b_deltas.csv`, `b27b_grade_raw.csv`.

## Resposta curta

**O TD3 do esqueleto não aprende uma política, e o pouco que aprende é uma constante pior que a melhor constante simples.**
- A política é independente do estado (B2.7a).
- O ganho sobre o centro não se distingue de uma constante deslocada (critério 2 atendido).
- O ganho também não se distingue de um TD3 sem aprendizado do ator (td3_ref ≈ td3_frozen nas 3 células).
- Um limiar fixo b = 0,25 supera o TD3 em **4,4 a 6,9 p.p. nas 3 células**, com IC95 inteiro abaixo de 0.

## B2.7a: mecanismo (descritivo)

- **Verificação:** reproduz o `td3_ref` do B2.7 exatamente (máx. |Δacc| = 5,6e-17).

| célula | drift (π₁₅₀ − π₀) | sd_estados π₁₅₀ | sd_estados π₀ | distância ao centro |
|---|---|---|---|---|
| `label_flipping` α 0,05 | 0,064 | 0,0005 | 0,0004 | 0,066 |
| `label_flipping` α 0,1 | 0,061 | 0,0004 | 0,0004 | 0,073 |
| `low_mag_backdoor` α 0,05 | 0,061 | 0,0004 | 0,0004 | 0,068 |

- A dependência do estado é ~300× menor que σ_a = 0,15 e igual à do ator inicial.
- O ator desliza até uma **constante** perto do centro: a ≈ (0,46; 0,56; 0,50; 0,50), **b ≈ 0,45** (centro 0,5).
- **Condição 1 do B2.7c: 0/3 → o B2.7c não roda.**

## B2.7b: de onde vem o Δ (H = 150, sementes 42–51, n = 10 por célula)

Acurácia média por sistema:

| célula | fixed (b 0,5) | **fixed_b025** | fixed_b075 | fixed_td3mean | td3_frozen | td3_ref |
|---|---|---|---|---|---|---|
| `label_flipping` α 0,05 | 61,8 | **75,3** | 54,7 | 67,4 | 66,1 | 68,5 |
| `label_flipping` α 0,1 | 71,4 | **78,5** | 47,3 | 72,3 | 73,2 | 74,1 |
| `low_mag_backdoor` α 0,05 | 80,6 | **88,2** | 71,5 | 82,9 | 83,3 | 83,7 |

Δ pareados (p.p., IC95):

| comparação | LF α 0,05 | LF α 0,1 | LMB α 0,05 |
|---|---|---|---|
| td3_ref − td3_frozen (só o aprendizado) | +2,3 (−1,2; +5,9) | +0,9 (−1,7; +3,5) | +0,4 (−2,2; +3,0) |
| td3_frozen − fixed (ruído + aquecimento, sem aprendizado) | +4,3 (−0,04; +8,6) | +1,8 (−2,4; +6,1) | +2,7 (−0,8; +6,3) |
| td3_ref − fixed_td3mean (a constante aprendida) | +1,1 (−5,6; +7,7) | +1,8 (−2,3; +5,8) | +0,9 (−2,8; +4,5) |
| fixed_td3mean − fixed (só deslocar a constante) | +5,6 (−0,05; +11,2) | +0,9 (−3,6; +5,5) | +2,2 (−2,6; +7,0) |
| **td3_ref − fixed_b025 (melhor constante)** | **−6,9 (−10,7; −3,1)** | **−4,4 (−7,3; −1,5)** | **−4,4 (−6,8; −2,1)** |

**Critérios do ADENDO2 (pré-registrados):**
1. **"O ganho é exploração, não aprendizado": NÃO atendido.**
   - A 1ª parte vale: o IC95 de td3_ref − td3_frozen contém 0 nas 3 células.
   - A 2ª parte falha: td3_frozen − fixed é positivo nas 3 células, mas o IC95 toca 0 em todas (em `label_flipping` α 0,05 o limite inferior é −0,04 p.p.), e o critério exigia IC95 > 0 em ≥ 1.
   - *Leitura não pré-registrada:* com a regra literal "Δ > 0" (só a média), o critério seria atendido (3/3 positivos). O resultado fica no limite.
2. **"Constante deslocada" (fixed_td3mean ≈ td3_ref): ATENDIDO.** O IC95 contém 0 nas 3 células. A TOST ±1 p.p. não fecha em nenhuma, então é "não distinguível", não "equivalente".
3. **Secundário (td3_ref > melhor constante): 0/3.** Pelo contrário, o TD3 **perde** para b = 0,25 nas 3 células, com IC95 inteiro abaixo de 0.

## Leitura para o P2

- O sinal positivo do TD3 em `label_flipping` H = 150 (B2.7, +6,65 p.p. sobre o centro) **não é aprendizado de política**:
  - a política não depende do estado;
  - tirar o aprendizado do ator não muda o resultado de forma detectável;
  - a constante aprendida reproduz o TD3.
- O Δ sobre o centro decompõe-se, sem separação estatística entre as partes, em um pequeno deslocamento da constante (b 0,5 → ~0,45) e no ruído de exploração.
- **"O aprendizado equivale, no máximo, a ajustar o limiar, e ajusta mal."** A direção certa (b menor) é muito mais longe do que o TD3 vai: b = 0,25 ganha 4–7 p.p. sobre o TD3. Isso é coerente com o B2.6 (b = 0,25 é a melhor variante, +1,6 p.p.) e reforça o limiar como o componente decisivo.
- **Limitações:**
  - 3 células escolhidas pela inclinação positiva no B2.7 (as mais favoráveis ao TD3);
  - n = 10;
  - B2.7b explicativo (desenhado depois do B2.7a);
  - o b = 0,25 é escolhido em retrospecto.
