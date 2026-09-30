# Resultado — B2.1 + B2.2: replicação confirmatória no AdaAggRL oficial

**Data:** 2026-09-30
**Pré-registro:** `PREREGISTRO.md` (hash e hashes do código em `PREREGISTRO.sha256`; `analisar_b21.py` e `analisar.py` conferidos contra o hash antes da execução: SUCESSO).
**Grade:** 40/40 runs (sementes 105–114 × LMP/EB × fixed/td3; MNIST, q=0,5, 500 rodadas), 2026-09-27 13:30 → 2026-09-30 03:37, 0 falhas. 40 `obs`, 20 checkpoints finais do ator.
**Análise:** `scripts/passo2_oficial/analisar_b21.py`, rodada uma vez → `analise.txt`, `resumo_runs.csv`, `b22_mecanismo.csv`.

---

## 1. B2.1 — resultado confirmatório

Métrica primária: mediana da acurácia nas rodadas 401–500. D = fixed − td3, n = 20 pares (ataque × semente).

| | Δ (p.p.) | IC95 | d | TOST ±1,0 p.p. | Wilcoxon |
|---|---|---|---|---|---|
| **pooled (primária)** | **−0,22** | (−1,01; +0,57) | −0,13 | **p = 0,027** | p = 0,064 |
| LMP | +0,10 | (−0,02; +0,22) | +0,60 | — | p = 0,11 (Holm 0,21) |
| EB | −0,54 | (−2,26; +1,17) | −0,23 | — | p = 0,38 (Holm 0,38) |
| pooled (secundária: média 451–500) | −0,12 | (−0,67; +0,43) | −0,10 | p = 0,002 | p = 0,097 |

→ **VEREDITO PRÉ-REGISTRADO: EQUIVALÊNCIA CONFIRMADA.** A ação fixa e o TD3 publicado são equivalentes dentro de ±1,0 p.p. no código e no horizonte oficiais.

**Nota sobre o IC:** o TOST com α = 0,05 corresponde ao **IC 90%**, que é (−0,87; +0,43) p.p. e fica inteiro dentro de ±1,0. O IC 95% da tabela toca −1,01 e é reportado por completude.

## 2. B2.2 — hipóteses mecanísticas (Holm sobre H3–H5)

σ_a = 0,0475 (ruído de exploração do SB3 em unidades de ação).

| hipótese | estatística | resultado |
|---|---|---|
| **H3** corr. das ações EB × LMP > 0,9 | mediana r = 0,991 (por semente: 0,993 · 0,991 · 0,994 · 0,990 · 0,922 · **0,899** · 0,989 · 0,988 · 0,993 · 0,992) | Holm p = 0,002 → **CONFIRMADA** |
| **H4** drift \|π₅₀₀ − π₀\| < σ_a | mediana 0,0097; máx. 0,0147 (≈ 1/5 do ruído) | Holm p < 0,001 → **CONFIRMADA** |
| **H5** S_swap \|π₅₀₀(s) − π₅₀₀(s′)\| < σ_a | mediana 0,0055; máx. 0,0062 | Holm p < 0,001 → **CONFIRMADA** |

Referências (sem teste): S_shuffle = 0,0061; **S_swap do ator inicial π₀ = 0,0055**, idêntico ao do ator final; |π₀ − centro| = 0,0265.

**Leitura:** a sensibilidade da política final à entrada é a mesma de uma rede **recém-inicializada** (0,0055 contra 0,0055). Em 500 rodadas, a política deriva cerca de 1/5 do próprio ruído de exploração e não passa a responder ao estado. Trocar a observação inteira pela de outro ataque muda a ação em ~0,006 numa escala de 0–0,95.

## 3. Achados exploratórios (não confirmatórios)

- **O único par discrepante vem de reset tardio, de novo.** Em EB, semente 110, o `fixed` resetou na rodada 392. A recuperação ocupou **52% da janela 401–500** (rodadas com acurácia < 0,9), acima do ponto de ruptura da mediana. A premissa do pré-registro ("um reset tardio contamina ~30 rodadas") falhou neste caso. Mesmo com o par incluído, o TOST fecha. Sem ele (sensibilidade, **não** pré-registrada): Δ = +0,15 p.p., sd 0,24.
- **Resets extras** (média por run): EB td3 2,2 contra fixed 0,2; LMP td3 0,6 contra fixed 0,8. Os resets do td3 concentram-se no início (por exemplo s110: rodadas 28, 41, 87), na fase de aquecimento aleatório, como no Passo 2.
- **Massa de peso nos atacantes** ≤ 0,0001 em todas as condições: o esqueleto fixo exclui os atacantes com ou sem TD3.
- A menor correlação de H3 (s110, 0,899) coincide com o par em que os dois runs tiveram histórias de reset diferentes. Isso é coerente com o fato de a correlação vir da sequência de ruído compartilhada, perturbada pelos resets.

## 4. Consequências (roadmap §4, Portão B-b)

- **B-b fechado a favor da tese do P2:** no código e no horizonte publicados, o TD3 do AdaAggRL é **equivalente** a uma ação fixa no centro do espaço de ações (confirmatório, sementes novas 105–114).
- **Mecanismo confirmatório:** a política não se afasta da inicialização e **não depende da entrada** (H4, H5), com sensibilidade igual à de uma rede aleatória.
- **O que ainda não se pode afirmar:** que "o RL não se paga" em geral. Isso depende do steelman (B2.3, em execução, Portão B-a): esta é a configuração **publicada**.
- **Lição metodológica para os próximos pré-registros:** métricas "robustas a reset" por janela fixa podem falhar quando a recuperação é lenta. Vale considerar uma métrica de área sob a curva ou a exclusão pré-declarada de janelas pós-reset com sensibilidade.
