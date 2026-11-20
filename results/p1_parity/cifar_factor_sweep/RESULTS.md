# Result — FLTrust factor sweep on CIFAR-10

**Date:** 2026-11-03. **Status:** descriptive pilot on seeds 0–2, outside the confirmatory block. It confirms nothing and does not choose the campaign's factor.

**Pre-registration:** `ADDENDUM.md` (hash and code hashes check out). **Grid:** 24/24, 0 failures, from 11:46 to 14:49.

**Deviations from the addendum:**
- The addendum says the grid runs while W1 is paused. W1 was resumed by the author and overlapped from 11:59 to 13:19. This affects timing only: the sweep runs on CPU and its seeds are fixed.
- The estimated cost was about 2 h. It took 3 h 03 min, at 24.9 s per round against the 15 s measured in the pilot at 4 processes.

## Reading under the frozen rule

Stability is the mean absolute change between consecutive rounds, rounds 51–110. A factor stabilizes if this is at most half of factor 1's in all 4 cells.

| factor | stability / factor 1, per cell | cells at most half | reading | accuracy, mean of 4 cells |
|---|---|---|---|---|
| 1 | | | reference | 21.20 |
| 1/3 | 0.73, 0.47, 1.43, 0.30 | 2 of 4 | partly | 37.31 |
| 1/5 | 0.49, 0.13, 0.62, 0.09 | 3 of 4 | partly | 34.46 |
| 1/10 | 0.09, 0.05, 0.10, 0.03 | 4 of 4 | **stabilizes** | 29.68 |

**Best for FLTrust under the rule: factor 1/10,** the only factor that stabilizes.

That reading should not be taken without the next two sections: factor 1/10 has the lowest accuracy of the three reduced factors, 7.63 p.p. below 1/3, and it is still climbing at round 110.

## Accuracy and stability per cell

Seed means. Accuracy is the trailing mean of rounds 106–110.

| attack | root | accuracy: 1 | 1/3 | 1/5 | 1/10 | stability: 1 | 1/3 | 1/5 | 1/10 |
|---|---|---|---|---|---|---|---|---|---|
| `sign_flipping` | 100 | 23.11 | 35.53 | 32.87 | 30.76 | 11.19 | 8.17 | 5.49 | 0.96 |
| | 2,500 | 18.25 | 40.10 | 36.05 | 27.51 | 4.05 | 1.89 | 0.52 | 0.21 |
| `trim_attack` | 100 | 25.18 | 35.03 | 32.19 | 33.07 | 6.89 | 9.88 | 4.28 | 0.72 |
| | 2,500 | 18.26 | 38.58 | 36.75 | 27.40 | 7.60 | 2.27 | 0.69 | 0.25 |

FedAvg at factor 1, same attacks and seeds: 37.40 on `sign_flipping`, 26.03 on `trim_attack`. It is not a fair reference for the reduced factors, because it runs at a different rate. The rate-matched runs are in `../cifar_factor_rules/`.

## The root contrast

Root 2,500 minus root 100, per attack:

| factor | `sign_flipping` | `trim_attack` | seeds with a positive sign, of 6 |
|---|---|---|---|
| 1 | −4.86 | −6.92 | 0 |
| 1/3 | +4.57 | +3.55 | 6 |
| 1/5 | +3.18 | +4.57 | 6 |
| 1/10 | −3.25 | −5.67 | 1 |

**The sign changes twice along the factor.** It is negative at 1, positive at 1/3 and 1/5, and negative again at 1/10. So the direction of E1's hypothesis on this block does not follow from the size of the factor alone.

## The horizon at factor 1/10

Descriptive, not in the reading rule. Accuracy by window, seed means:

| factor | attack | root | rounds 71–90 | rounds 91–110 | change |
|---|---|---|---|---|---|
| 1/10 | `sign_flipping` | 100 | 28.43 | 29.88 | +1.46 |
| | | 2,500 | 24.29 | 26.54 | +2.25 |
| | `trim_attack` | 100 | 30.56 | 32.56 | +2.00 |
| | | 2,500 | 23.82 | 26.46 | +2.64 |

At factor 1/10 the model is still learning at R = 110, and root 2,500 is climbing faster than root 100. The negative contrast may therefore be partly a horizon effect. This was not tested, and the addendum fixed in advance that the reading concerns this horizon and not convergence.

## Limits

- 2 cells of 21, one α, 3 seeds.
- The spread across seeds is large relative to some differences. On `sign_flipping`, root 100, factor 1/10 gives 35.38, 28.52 and 28.36.
- The stability criterion favours small factors by construction: a smaller step moves the curve less, whether or not the model is better. This was visible only after the fact and is not a change to the rule.

## What this does not decide

The campaign's factor needs the trimmed mean and FedAvg at the same factors (`../cifar_factor_rules/ADDENDUM.md`), because the paper's rate is common to all rules.

## Outputs

- `analysis/analysis.txt`, `analysis/runs.csv`, `analysis/curves.csv`.
- `scripts/p1/analyze_cifar_factor_sweep.py`.
