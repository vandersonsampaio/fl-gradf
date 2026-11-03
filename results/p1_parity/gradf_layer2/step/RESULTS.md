# Result — a noise-free arm that shrinks the server's step

**Date:** 2026-11-03. **Status: complete, not confirmatory.** Plan: `ADDENDUM.md`, frozen before the grid. 10/10 jobs, 0 failures, 09:51–10:02, 8 processes, with the W1 rerun paused. Analysis: `scripts/p1/analyze_gradf_step.py`, output in `analysis/`.

**In short:** multiplying FLTrust's aggregate by 1/3, with no noise, reproduces the gain of the noise. By the frozen rule the reading is **step alone goes further**. On MNIST the noise of Layer 2 therefore acts through the size of the step, not as a defense. Fixed FLTrust at factor 1 swings about 10 p.p. per round under parity, and this inflates the gains measured at the final round.

## 1. Frozen contrasts

18 confirmatory cells, final round, p.p.; arm means: F000 74.92, N010 83.65, S010 84.29.

| contrast | mean (CI95) | category |
|---|---|---|
| S010 − F000 | +9.37 (+7.80; +10.94) | superiority |
| N010 − S010 | −0.65 (−1.01; −0.28) | inferiority |

**Reading: step alone goes further.**

- **The difference between the two is small.** −0.65 p.p. also passes the TOST at the 1.0 p.p. margin (p = 0.027). `inference.describe` checks inferiority first, so the category is inferiority. In practice the arms are close.
- **The factor was tuned** on these seeds (`ADDENDUM.md`), which favours S010 in this comparison.
- **The gain falls with α as under noise:** +11.83 at α 0.05, +10.52 at 0.10, +2.31 at 0.50.

## 2. Descriptive, not in the addendum

| 18 cells, p.p. | mean (CI95) | category |
|---|---|---|
| S010 − F111 | +0.76 (+0.46; +1.07) | superiority |
| GRADF − S010, first run | −1.49 (−2.11; −0.86) | inferiority |
| GRADF − S010, new 1 | −1.93 (−2.75; −1.10) | inferiority |
| GRADF − S010, new 2 | −1.60 (−2.94; −0.26) | inferiority |

With the mean of the last 5 rounds instead of the final round:

| 18 cells, p.p. | mean (CI95) | category |
|---|---|---|
| S010 − F000 | +6.19 (+5.43; +6.95) | superiority |
| N010 − F000 | +6.02 (+5.45; +6.59) | superiority |
| N010 − S010 | −0.17 (−0.46; +0.12) | equivalence |

Mean absolute change between consecutive rounds over the last 5 rounds: **F000 9.79 p.p., N010 1.44, S010 0.61.**

- **Fixed FLTrust at factor 1 is unstable on MNIST too,** alternating between about 75% and 80% by round. The 15th round falls on a low point, so the final-round gain of +9.37 is about a third larger than the +6.19 of the trailing mean.
- **The MNIST metric stays the final round** under `../../ADDENDUM1.md`. These numbers show that it is sensitive for this rule.
- **All arms are still climbing at round 15** except F000, so the size of the gain depends on the horizon.

## 3. What is not established

- **Other rules.** Only FLTrust was scaled. It is not known whether other rules, or GRADF and Random, gain from a smaller step.
- **CIFAR-10.** The FLTrust swing of about 8 p.p. per round there looks similar, but no scaled arm has run on the CNN.
- **The value of the factor.** 1/3 is one value on seen seeds, not a tuned or validated setting.
- **Faithfulness.** FLTrust's global learning rate in Cao et al. should be checked against the paper before this is written up. The implementation here applies the aggregate at factor 1.

## 4. Proposals for the author (nothing changes on its own)

- **Do not present Layer 2 as a defense.** On MNIST its effect on FLTrust is matched by a smaller step without noise.
- **Before the pre-registration, a short CIFAR-10 pilot** of FLTrust with the same factor on block C (seeds 0–2, outside the confirmatory block), to see whether it stabilizes the rule. If it does, the campaign has a design decision to make: fixed FLTrust at factor 1 would be a weak reference, and E1's and E3's comparisons depend on it.
- **Report the MNIST results with both metrics** where FLTrust's swing affects them.

## Outputs

`analysis/analysis.txt`, `contrasts.csv`, `contrasts_by_alpha.csv`, `contrasts_by_attack_type.csv`, `per_cell.csv`, `accuracy_by_round.csv`; `raw/`.
