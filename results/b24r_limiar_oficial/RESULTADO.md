# Resultado — B2.4r: varredura do limiar a₅ no AdaAggRL oficial (EB)

**Data:** 2026-10-04.
- Plano: `PLANO.md`, commit `d676ec3`.
- Adendo 1: commit `53c4722`. O centro foi refeito porque a GPU não é bit-reprodutível.
- Grade: 25 runs (a₅ ∈ {0; 0,1; 0,25; 0,475; 0,95} × sementes 100–104, 500 rodadas), de 02/10 20h36 a 04/10 10h50. Sem falhas.
- Saídas: `analise.txt`, `resumo_runs.csv`, `delta_vs_centro.csv`, `raw/`.

## Respostas (critérios do plano; exploratório, n = 5)

- **Q1, o limiar tem efeito causal no código publicado: SIM.** Abaixo de a₅ ≈ 0,25 o AdaAggRL colapsa sob EB:

| a₅ | primária (mediana 401–500) | secundária (média 1–500) | resets por run | atacantes excluídos por rodada (de ~2) | Δ primária contra o centro (IC95) |
|---|---|---|---|---|---|
| 0 | 34,4% | 34,6% | 56–66 | 0,45 | **−62,2 p.p.** (−64,2; −60,2) |
| 0,1 | 78,1% | 65,8% | 14–15 | 1,41 | **−18,5 p.p.** (−24,3; −12,8) |
| 0,25 | 96,6% | 92,2% | 0–2 | 1,96 | −0,05 (−0,38; +0,28) |
| **0,475 (centro)** | **96,7%** | **92,8%** | 0–2 | 1,98 | — |
| 0,95 | 96,3% | 90,3% | 0–3 | 1,95 | −0,38 (−0,64; −0,12) |

- **Q2, existe uma constante melhor que o centro: NÃO.**
  - Nenhum a₅ supera o centro.
  - a₅ = 0,25 empata (Δ −0,05 p.p.).
  - a₅ = 0,95 fica ligeiramente abaixo (−0,38 p.p., com IC95 que exclui 0, mas abaixo do limiar de 2 p.p.). Ele exclui ~4,9 clientes por rodada, contra ~2,6 no centro, então descarta honestos.

## Mecanismo

- O efeito é um **degrau**, não uma curva suave:
  - **com a₅ ≤ 0,1**, o filtro deixa atacantes EB passarem (0,45 e 1,41 excluídos de ~2 por rodada). O modelo diverge, e o ambiente oficial reinicia em cascata (14 a 66 resets por run);
  - **de 0,25 a 0,95**, há um platô: os 2 atacantes por rodada são excluídos quase sempre e a acurácia fica a ≤ 0,4 p.p. do centro (primária).
- Mesmo com a₅ = 0, o mecanismo oficial ainda exclui 1 cliente por rodada: o de menor score tem k = 0 após o min-max.
- Massa de peso nos atacantes ≈ 0 em todos os a₅ ≥ 0,25.

## Leitura para o P2

- **O limiar é causal também no código publicado.** Ele é o único componente da ação que, sozinho, leva o AdaAggRL do colapso (34%) ao teto (97%). Isso fecha, na Parte I, a evidência do B2.6 (framework próprio) e dos colapsos acidentais do B2.3, em que cantos com a₅ ≈ 0 colapsaram sob EB.
- **Não há o que aprender sobre o limiar sob EB no código oficial.** O centro já está no platô ótimo, então um TD3 que ajustasse a₅ não teria ganho a capturar. Isso é coerente com B2.1/B2.2 (TD3 ≡ fixed, política parada no centro): o regime publicado não oferece gradiente útil de recompensa na direção do limiar.
- **Contraste com o framework próprio** (B2.6/B2.7b): lá, b = 0,25 é melhor que o centro (até +4–7 p.p. sobre o TD3 em `label_flipping`). A posição ótima do limiar depende do ataque e do regime; sob EB/MNIST/q = 0,5 o centro já é ótimo. É mais um argumento para o limiar como parâmetro central e explicável (R6), e para protegê-lo (R5): o colapso fica a um passo (a₅ = 0,1).

## Ressalvas

- n = 5, uma célula (EB, MNIST, q = 0,5), exploratório, sem correção de multiplicidade.
- O centro foi refeito nesta grade. O centro do Passo 2 (mesma configuração, outra execução na GPU) deu primária 94,9% contra 96,7%: a diferença entre execuções da mesma configuração chega a ~1,8 p.p. na média de 5 sementes.
- Os resets com a₅ ≤ 0,1 dominam as duas métricas, inclusive a secundária, que não exclui janelas pós-reset. Mas eles são consequência da ação e contam como resultado (regra do plano).
