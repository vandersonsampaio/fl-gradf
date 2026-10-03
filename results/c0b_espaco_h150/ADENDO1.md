# Adendo 1 — C0b: código, registro por cliente e verificações

**Data:** 2026-10-03 09:25:37 -0300, antes de disparar a grade. O plano (`PLANO.md`, commit `d676ec3`) e seus critérios **não mudam**.

## Implementação

- **`scripts/c0b_espaco_h150.py`**, com um job por célula × semente e `keras.utils.set_random_seed(seed)` antes de cada sistema:
  - variantes: `DecompLearner` do B2.6;
  - regras estáticas: `PlainRuleLearner` do B2.7;
  - tetos: `run_ceiling` do B2.7, já com a correção do oráculo FedAvg-8.
- **Lançador:** `scripts/run_grid_c0b.sh`, 12 processos, 190 jobs de célula + 30 de teto.
- **Grade conferida:** 19 células (`fltrust_aligned` só em α 0,5) × 8 sistemas: `sr_only`, `sr_bin`, **`sr_b025`**, `cosserver_only`, `fltrust`, `trimmed_mean`, `clustering` e `median`.

## Acréscimo: registro por cliente (pedido do autor)

- Nas 4 variantes do esqueleto, por rodada × cliente, vão para `raw/scores_<célula>_seed<s>_R150.csv.gz`:
  - `is_byz`;
  - **S_R**, nas variantes que já fazem a inversão; NaN no `cosserver_only`, para não pagar a inversão;
  - **cos_server**, calculado **sempre**, com save/restore do `np.random` como no B2.6;
  - a **máscara** (`incluido` = peso > 0) e o peso normalizado.
- **Para que serve:** o A0 de concordância entre sinais (Spearman S_R × cos_server e AUC por sinal, agora com 10 sementes e 19 células) e o teto da seleção de sinais, sem rodar de novo.
- A análise pré-escrita ganhou uma seção descritiva de concordância (trajetória `sr_only`, rodadas ≥ 2). Os critérios do plano não mudam.

## Verificações (15 rodadas, célula `label_flipping` α 0,05, semente 72)

- **Determinismo (PLANO §4):** o mesmo job em dois processos deu acurácias idênticas nos 8 sistemas (máx. |Δ| = 0).
- **Registro neutro:** o mesmo job sem registro deu acurácias idênticas (máx. |Δ| = 0). Calcular o cos_server em todas as variantes não altera a trajetória.

## Declaração

O teste a seco da análise (`analisar`), feito sobre o smoke acima, **exibiu acurácias** dessa célula e semente (semente 72, `label_flipping` α 0,05, H = 15) antes da grade. O plano e o critério já estavam congelados e commitados (`d676ec3`), e nenhuma escolha de desenho ou de análise foi feita a partir desses números. Esses runs do smoke não entram na grade: ela roda de novo todas as células e sementes.

## Custo revisto

Cada job tem 3 variantes com inversão (~1.600–1.900 s cada com a CPU disputada), o `cosserver_only` e 4 regras (baratos). São ~1,8 h por job e 190 jobs com 12 processos, ou seja, **~28–35 h**.

## Código no congelamento
97b0cdf13b85112927b969d0b770c4fdc25d1cd0b9bf9cbdf2d9f46f40d854fc  scripts/c0b_espaco_h150.py
0d2454805068b74779ffb5bf98af4f60911a0aff5556cba5a43825b8adfe1549  scripts/run_grid_c0b.sh
