# Result — B2.8s: does the granularity verdict depend on the metric?

**Date:** 2026-10-05.
- Plan: `PLANO.md` (English translation in `PLANO.en.md`), commit `c15937d`.
- Grid: 210 jobs (21 cells × seeds 42–51 × 9 systems), from 10/04 19:26 to 10/05 00:12. No failures.
- Outputs: `analise.txt`, `celulas_uniforme.csv`, `celulas_ponderada.csv`, `grade_raw.csv` (with the per-client accuracies).

## The metric sensitivity is vacuous: the two metrics are identical by construction

- **All clients have test sets of the same size (1000 examples)**, at every α. `data/download_datasets.py` splits the global MNIST test set (10,000) **randomly and into equal parts** across the 10 clients ("shared test set split equally across clients").
- Therefore:
  - the mean **weighted by test-set size** is mathematically equal to the **uniform** mean (max. |Δ| = 2e-16 over the 1,890 runs);
  - **both are the accuracy on the global MNIST test set, which is IID and class-balanced.** The C0/B2.7/B2.8/C0b metric **is not a "per-client" mean over local distributions**; it is the global accuracy.
- **The formal criterion was met** ("granularity robust": same verdict, 100% agreement), **but it has no content**: the two lines of the analysis are the same metric. The small differences at α 0.5 (≤ 0.002 p.p.) come from the greedy oracle's tie-breaks, not from the metric.
- **Design error, declared:** the plan started from the (analyst's) premise that the metric was a uniform mean over local test sets of different sizes. The weighting inventory was done; the sizes and origin of the test sets were not checked before the launch.

## What the grid shows anyway

- **Reproduction of B2.8 with per-system reseeding (seeds 42–51):** 14 target cells, 7 with the per-round oracle below the skeleton and 2 above → **"granularity explains"**. The original B2.8 had 8/14 below, with the same verdict. The verdict withstands the change of RNG and of the references' implementation: skeleton and static rules re-run in the same grid.

## Consequences

1. **The underlying question remains open, but reformulated.** The concern was: "part of the per-client filter's advantage comes from weighting the honest clients uniformly, and the metric rewards that". With an IID global test set, the mechanism that favors uniform weighting is not a per-client metric. It is the **class imbalance** that sample-size weighting introduces when the large clients have a skewed distribution (Dirichlet α ≤ 0.1), evaluated on a balanced test set. Testing this properly would require a metric on **local test sets with each client's label distribution**, which do not exist in the current pipeline (they would have to be built).
2. **C0b-iii, as planned, would be equally vacuous on the metric side:** the U and P pairs differ only in the ceiling, which was already measured in the C0b sensitivity. That is why **the queue was stopped before launching it** (10/05 00:53), pending the author's decision. D2 continues.
3. **The texts of 10/04 need correcting** (`VERIFICACOES.md` §3, C0's `CORRECAO_2026-10-04.md`, the C0b, A0 and B2.7 RESULTADOs) where they say "the metric is the uniform mean of the accuracies on the clients' test sets". The correct statement: the metric is the accuracy on the IID global test set; the sample-weighted oracle loses because, with per-client label skew, it gives too much weight to the classes of the large clients. **The numerical conclusions (uniform oracle as the ceiling, C0b sensitivity) do not change;** the explanation of the mechanism does.
