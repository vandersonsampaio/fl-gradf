# Result — B2.4r: sweep of the threshold a₅ in the official AdaAggRL (EB)

**Date:** 2026-10-04.
- Plan: `PLANO.md` (English translation in `PLANO.en.md`), commit `d676ec3`.
- Addendum 1 (`ADENDO1.md`, English translation in `ADENDO1.en.md`): commit `53c4722`. The center was re-run because the GPU is not bit-reproducible.
- Grid: 25 runs (a₅ ∈ {0; 0.1; 0.25; 0.475; 0.95} × seeds 100–104, 500 rounds), from 10/02 20:36 to 10/04 10:50. No failures.
- Outputs: `analise.txt`, `resumo_runs.csv`, `delta_vs_centro.csv`, `raw/`.

## Answers (plan criteria; exploratory, n = 5)

- **Q1, the threshold has a causal effect in the published code: YES.** Below a₅ ≈ 0.25 AdaAggRL collapses under EB:

| a₅ | primary (median 401–500) | secondary (mean 1–500) | resets per run | excluded attackers per round (of ~2) | primary Δ vs. the center (CI95) |
|---|---|---|---|---|---|
| 0 | 34.4% | 34.6% | 56–66 | 0.45 | **−62.2 p.p.** (−64.2; −60.2) |
| 0.1 | 78.1% | 65.8% | 14–15 | 1.41 | **−18.5 p.p.** (−24.3; −12.8) |
| 0.25 | 96.6% | 92.2% | 0–2 | 1.96 | −0.05 (−0.38; +0.28) |
| **0.475 (center)** | **96.7%** | **92.8%** | 0–2 | 1.98 | — |
| 0.95 | 96.3% | 90.3% | 0–3 | 1.95 | −0.38 (−0.64; −0.12) |

- **Q2, there is a constant better than the center: NO.**
  - No a₅ beats the center.
  - a₅ = 0.25 ties (Δ −0.05 p.p.).
  - a₅ = 0.95 is slightly below (−0.38 p.p., with a CI95 that excludes 0, but below the 2 p.p. threshold). It excludes ~4.9 clients per round, vs. ~2.6 at the center, so it discards honest clients.

## Mechanism

- The effect is a **step**, not a smooth curve:
  - **with a₅ ≤ 0.1**, the filter lets EB attackers through (0.45 and 1.41 excluded of ~2 per round). The model diverges, and the official environment resets in cascade (14 to 66 resets per run);
  - **from 0.25 to 0.95**, there is a plateau: the 2 attackers per round are excluded almost always and the accuracy stays within ≤ 0.4 p.p. of the center (primary).
- Even with a₅ = 0, the official mechanism still excludes 1 client per round: the lowest-scoring one has k = 0 after the min-max.
- Weight mass on attackers ≈ 0 for all a₅ ≥ 0.25.

## Reading for P2

- **The threshold is causal in the published code too.** It is the only component of the action that, alone, takes AdaAggRL from collapse (34%) to the ceiling (97%). This closes, in Part I, the evidence from B2.6 (in-house framework) and from the accidental collapses of B2.3, in which corners with a₅ ≈ 0 collapsed under EB.
- **There is nothing to learn about the threshold under EB in the official code.** The center is already on the optimal plateau, so a TD3 that tuned a₅ would have no gain to capture. This is consistent with B2.1/B2.2 (TD3 ≡ fixed, policy stuck at the center): the published regime offers no useful reward gradient in the direction of the threshold.
- **Contrast with the in-house framework** (B2.6/B2.7b): there, b = 0.25 is better than the center (up to +4–7 p.p. over TD3 on `label_flipping`). The optimal position of the threshold depends on the attack and the regime; under EB/MNIST/q = 0.5 the center is already optimal. It is one more argument for the threshold as a central, explainable parameter (R6), and for protecting it (R5): collapse is one step away (a₅ = 0.1).

## Caveats

- n = 5, one cell (EB, MNIST, q = 0.5), exploratory, no multiplicity correction.
- The center was re-run in this grid. The Step 2 center (same configuration, another GPU execution) gave a mean primary of 94.9% vs. 96.7%. The ~1.8 p.p. difference comes from a single seed (next section).
- The resets at a₅ ≤ 0.1 dominate both metrics, including the secondary, which does not exclude post-reset windows. But they are a consequence of the action and count as results (plan rule).

## Check of the center's seed 100 (2026-10-04, after the grid)

Comparison of the three executions of the same configuration (`fixed` a₅ = 0.475, EB, seed 100, same torch 2.3.0+cu121, same device cuda:0):
- Step 2;
- the 25-round check from ADENDO1;
- the center re-run in this grid.

| | Step 2 | re-run center |
|---|---|---|
| initial state (acc/loss of the initial reset) | 0.1221 / 358.66 | identical |
| first round with Δacc ≠ 0 | — | **round 1** |
| extra resets | **1, at round 421** | 0 |
| primary (median 401–500) | **88.3%** | 96.9% |
| secondary (mean 1–500) | 90.4% | 93.7% |

- The three executions start from the same initial state and diverge already at round 1. Max. |Δacc| = 10.3 p.p. between the check and the re-run center over the first 25 rounds; 8.4 p.p. between the check and Step 2. This confirms the ADENDO1 diagnosis: **the official code on GPU is not bit-reproducible across executions**, and the seed only fixes the initialization.
- **The mean difference of 1.8 p.p. between the two centers comes entirely from seed 100.**
  - In Step 2, that seed had a late reset (round 421) inside the primary-metric window. It is the same "late reset" that made Step 2's H1 inconclusive.
  - In the other 4 seeds, the centers differ by −0.40, −0.09, +0.15 and +0.65 p.p. (seeds 101–104).
- **Implication:** whether a late reset happens is itself random across executions of the same configuration. In a confirmatory experiment on the official code, the relevant variance includes this reset lottery, not just the accuracy variation across seeds. This is why there is a metric with AUC sensitivity and why resets count as results, which B3.1 should inherit.
- **No effect on the B2.4r conclusions:** Q1 and Q2 use the re-run center, in the same grid and with the same code version. Sensitivity with the Step 2 center (descriptive):
  - Q1 remains YES: a₅ = 0 gives −60.4 p.p. (−66.3; −54.6) and a₅ = 0.1 gives −16.7 p.p. (−26.0; −7.5) on the primary.
  - Q2 remains NO: a₅ = 0.25 gives +1.7 p.p. (−2.9; +6.3) on the primary and +0.3 (−0.4; +1.0) on the secondary; a₅ = 0.95 gives +1.4 (−3.2; +6.0) and −1.6 (−4.9; +1.6). The CI95s contain 0, and the positive Δ comes only from the seed-100 reset in the Step 2 center.
