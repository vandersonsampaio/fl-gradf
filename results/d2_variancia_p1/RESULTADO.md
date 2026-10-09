# Result — D2: run-to-run variance of the P1 numbers

**Date:** 2026-10-05.
- Plan: `PLANO.md` (English translation in `PLANO.en.md`), commit `7c0311b`.
- Code: tag `p1.0.0` (3beb0f8), via `git worktree`, unmodified.
- Grid: 9 jobs (seeds 42–44 × 3 repetitions), from 10/05 00:12 to 05:57. No failures.
- The original P1 execution counts as the 4th repetition.
- Outputs: `analise.txt`, `sd_exec_por_celula.csv`, `delta_agregado_por_execucao.csv`, `grade_raw.csv`.

## Reading rule (PLANO §4): **P1 CONCLUSION ROBUST TO RUN-TO-RUN VARIANCE**

- **Signs:** in all 12 execution × seed combinations, GRADF − Random < 0, FedStrategist − Random < 0 and AdaAggRL − Random > 0.
- **Magnitude:** the SD_exec of the aggregate Δ is below ½ |published Δ| for all three:

| system | published Δ (Tab. 4) | SD_exec of the aggregate Δ | ½ \|Δ\| | SD across P1's 10 seeds |
|---|---|---|---|---|
| GRADF | −0.042 | **0.0177** | 0.021 | 0.0335 |
| FedStrategist | −0.044 | 0.0005 | 0.022 | 0.0184 |
| AdaAggRL | +0.030 | 0.0001 | 0.015 | 0.0271 |

## Variance per system (across the 4 executions, per cell × seed)

| system | mean SD | max SD | mean range | max range |
|---|---|---|---|---|
| AdaAggRL | 0 | 0 | 0 | 0 |
| Random | 0.0001 | 0.002 | 0.0002 | 0.005 |
| Oracle | 0.0003 | 0.012 | 0.0006 | 0.023 |
| FedStrategist | 0.0007 | 0.013 | 0.0014 | 0.025 |
| **GRADF** | **0.093** | **0.31** | **0.19** | **0.63** |

## Reading

- **AdaAggRL is deterministic** across executions. Random, Oracle and FedStrategist vary little: at most 1–2 p.p. in one cell, and only the order of the global RNG affects them.
- **GRADF is strongly non-deterministic cell by cell:** a mean SD of 9.3 p.p., and the same cell with the same seed varies by up to **63 p.p.** across executions. The source is the selector's DQN (TF), as in B2.7.
- **In the aggregate over the 21 cells, the variance drops a lot:**
  - GRADF's Δ SD_exec is 1.8 p.p., ~half the variation across seeds (3.4 p.p.), and the negative sign holds in every execution;
  - P1's conclusion (discrete selection does not beat Random; AdaAggRL does) **does not depend on the execution**.

## Limitation note for P1 (ready for the revision)

> GRADF's per-cell results (Table 3) have unreported run-to-run variance: the DQN selector is non-deterministic across executions with the same seed (mean SD of 9.3 p.p. per cell; range up to 63 p.p.). The aggregate comparison in Table 4 is robust to this variance: across three seeds and four executions, the GRADF − Random difference remained negative in every combination. Since each P1 seed corresponds to one execution, the run-to-run variance is already contained in the across-seed variation used in the Table 4 tests; the tests remain valid, only noisier than necessary. FedStrategist, Random and Oracle vary by ≤ 2.5 p.p. per cell; AdaAggRL is deterministic.

**What supports the conclusion** is the **negative sign of GRADF − Random in all 12 combinations**. The SD/effect ratio is secondary and, for GRADF, has a small margin: SD_exec of 1.77 p.p. against the ½|Δ| = 2.1 p.p. threshold, with only 3 seeds.

**Use of the note:**
- do not contact the editor now, because P1's conclusion does not change;
- **include the note in the response to the revision, even if the reviewers do not ask**;
- in P2, the DQN limitation in B2.7 cites these numbers instead of "a few p.p.".

## P1 claims that depend on GRADF per cell or per attack (re-reading of 10/05)

No sentence in the text singles out GRADF in a specific cell ("wins in X", "collapses in Y"). These points, however, use GRADF values per cell or per attack and inherit the run-to-run variance. By D2, the expected execution SD of a 10-seed mean is ~2.9 p.p. per cell (9.3/√10) and ~1.7 p.p. per attack (30 cell × seed):

| where | what depends on GRADF per cell/attack | risk | action in the revision |
|---|---|---|---|
| **Table 3**, GRADF column | mean accuracy per cell (10 seeds) | ~2.9 p.p. of execution noise per cell | footnote: GRADF's per-cell values have run-to-run variance; do not compare isolated cells |
| **Figure 2** and text ("*In terms of headroom count, … none by GRADF*") | count of cells with captured headroom (per-cell criterion) | a cell can enter or leave the count due to execution noise | change to "none consistently", or recompute the count with means across executions |
| **Table 5**, GRADF columns, and text ("*Neither of the two surpasses the Random-selector in any individual attack type*") | Δ, p and d per attack type | GRADF's largest per-attack Δ is +0.014 (`sign_flipping`, p = 0.40). The claim is likely, but not guaranteed against ~1.7 p.p. of noise | keep, with the caveat that GRADF's per-attack values include run-to-run variance; the tests remain valid |
| **Figure 3** (per attack type) | GRADF's position per attack | same as Table 5 | same caveat |

## D2 closed

The "after D1" re-measurement is **no longer** a P1 item: it would only show that D1 worked. It becomes **D1's acceptance criterion in P3**: *|Δ| = 0 across executions with the same seed for the deterministic GRADF v1.*
