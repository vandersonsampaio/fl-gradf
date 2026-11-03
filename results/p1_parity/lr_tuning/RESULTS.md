# Result — global learning rate tuning per dataset

**Date:** 2026-11-03. **Status:** setting chosen under the frozen rule, **with the margin failing on both datasets.** Seeds 0–2, outside the confirmatory block. It confirms nothing. Adoption is the author's decision (`ADDENDUM.md`, Sec. 5).

**Pre-registration:** `ADDENDUM.md` (hash and code hashes check out). **Grid:** 72/72, 0 failures, from 15:13 to 20:36.

**Deviations from the addendum:**
- W1 was resumed by the author at 16:36 and overlapped until the end of the grid. The addendum provides for this and asks for it to be reported. It affects timing only: the grid runs on CPU with fixed seeds.
- The cost was about 5.4 h against the estimated 4.6 h, because of that overlap: a wave took 38 minutes before it and up to about 64 minutes during it.

## Reading under the frozen rule

| dataset | chosen | margin of 1.0 p.p. | largest gap at the chosen factor | reading |
|---|---|---|---|---|
| MNIST | **factor 1** | **failed** | 7.54, FLTrust | smallest largest gap |
| CIFAR-10 | **factor 1/3** | **failed** | 8.94, FedAvg | smallest largest gap |

No factor was admissible on either dataset. Under Sec. 4, step 3, of the addendum, the chosen factor is the one with the smallest largest gap, and this is reported as a failure of the margin.

## Training error

Mean of the last 5 rounds, seed means. The gap to each rule's own best is in parentheses.

| dataset | factor | FedAvg | trimmed mean | FLTrust | largest gap |
|---|---|---|---|---|---|
| MNIST | 1 | 15.91 (0) | 25.29 (0) | 23.80 (7.54) | **7.54** |
| | 1/3 | 24.89 (8.98) | 41.77 (16.48) | 16.26 (0) | 16.48 |
| | 1/5 | 29.88 (13.97) | 49.44 (24.15) | 19.56 (3.30) | 24.15 |
| | 1/10 | 35.03 (19.12) | 55.86 (30.57) | 28.56 (12.30) | 30.57 |
| CIFAR-10 | 1 | 50.85 (0) | 69.83 (0) | 72.36 (9.82) | 9.82 |
| | 1/3 | 59.79 (8.94) | 75.71 (5.88) | 64.97 (2.43) | **8.94** |
| | 1/5 | 64.69 (13.84) | 78.66 (8.83) | 66.33 (3.79) | 13.84 |
| | 1/10 | 74.14 (23.29) | 82.03 (12.20) | 62.54 (0) | 23.29 |

## What the failure means

**The rules do not share a best rate.** FedAvg and the trimmed mean are best at factor 1 on both datasets. FLTrust is best at 1/3 on MNIST and at 1/10 on CIFAR-10. The conflict is between FLTrust and the other two, and no factor in this set resolves it.

**The best rate of FedAvg and the trimmed mean is at the edge of the grid.** Factors above 1 were not run, so their true best may be higher, which would widen the conflict.

**The CIFAR-10 choice is close.** The largest gaps of factors 1/3 and 1 differ by 0.88 p.p., less than the margin itself. Applied to each seed alone, the same rule picks 1/3 on seeds 1 and 2 and factor 1 on seed 0.

## The factor sets which rule leads

FLTrust minus trimmed mean, test accuracy on the last 5 rounds, no attack, seed means. Descriptive, not used in the choice.

| dataset | factor 1 | 1/3 | 1/5 | 1/10 |
|---|---|---|---|---|
| MNIST | +0.92 | +26.88 | +30.81 | +27.21 |
| CIFAR-10 | **−2.98** | +10.04 | +12.24 | +18.69 |

On CIFAR-10 the sign changes between factor 1 and factor 1/3. Part of this difference comes from the trimmed mean being penalized by every reduced factor. **A common factor is therefore not neutral between these two rules, whichever value it takes.**

## FLTrust remains unstable at the chosen factors

Mean absolute change of the training error per round, second half of the run, no attack, p.p.:

| dataset | factor | FedAvg | trimmed mean | FLTrust |
|---|---|---|---|---|
| MNIST | 1 | 0.62 | 1.22 | **9.39** |
| CIFAR-10 | 1 | 0.88 | 0.66 | 13.00 |
| | 1/3 | 0.31 | 0.23 | **11.82** |
| | 1/10 | 0.14 | 0.09 | 0.51 |

The oscillation therefore does not need an attack. As fixed in Sec. 5 of the addendum, this is a result to report and is not a reason to re-choose the factor from these numbers.

## Limits

- 3 seeds, one α, root 100 only.
- The horizons are short: FedAvg's training error is still 15.91 on MNIST and 50.85 on CIFAR-10 at the end of its best run.
- "First round within the margin of the final error" (`analysis/analysis.txt`) is hard to read for an oscillating run. An early round can fall within the margin by chance, as with FLTrust at factor 1.
- The cause of the conflict between FLTrust and the other rules was not tested.

## Consequences if the author adopts the rule's choice

- **MNIST at factor 1:** `../mnist_fixed` and `../mnist_rq3` do not need rerunning on account of the factor.
- **CIFAR-10 at factor 1/3:** all rules of the campaign run at 1/3. The verification under attack needs a separate addendum.

## Outputs

- `analysis/analysis.txt`, `analysis/runs.csv`.
- `raw/`: runs, one JSON and one log each.

## Decision of the author (2026-11-03)

The rule's factors are adopted: factor 1 on MNIST and factor 1/3 on CIFAR-10. CIFAR-10 at factor 1 becomes a pre-registered sensitivity analysis, with E1 at both factors, and the extra machine hours are accepted. See `../factor_attack_check/ADDENDUM.md`.
