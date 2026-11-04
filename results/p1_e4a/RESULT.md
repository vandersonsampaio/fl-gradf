# E4a result: temporal headroom, metric check on MNIST

**Date:** 2026-11-04. Plan: `PLAN.md`, frozen before the grid; all hashes matched at analysis time. Grid complete: 10 of 10 seeds (42–51), 0 failures. Analysis run once, on the complete grid: `analysis/analysis.txt`, `headroom.csv`, `phases.csv`, `curves.csv`. Figure: `analysis/curves.png` and `.pdf`, from `scripts/p1/plot_e4a.py`, which was not in the plan's hash list.

**The headroom here is constructed to validate the metric. This is not a confirmatory claim.** The phases and their rules came from the MNIST rerun on these same seeds.

## 1. Result

**At α = 0.1, the main scenario, the CI of H_T lies above zero, so under the reading fixed in the plan the metric detects the constructed headroom.**

| α | best fixed rule | A(G), % | A(best fixed), % | H_T, p.p. | CI95 | Wilcoxon p | seeds with H_T > 0 |
|---|---|---|---|---|---|---|---|
| 0.1 | FLTrust | 80.77 | 79.13 | **+1.65** | (+0.82; +2.48) | 0.0020 | 10 of 10 |
| 0.5 | trimmed mean | 89.29 | 88.53 | **+0.75** | (+0.73; +0.77) | 0.0020 | 10 of 10 |

- **Cross-fitted reference:** the same rule was chosen on all 10 folds at each α, so the cross-fitted H_T equals the in-sample one.
- **α = 0.5:** the gain is small and precise. It reflects little contrast between rules, as the plan expected.
- **Rough expectation at α = 0.1:** the plan's 1.5 p.p. is close to the result, but the agreement hides a different composition (Sec. 2).

## 2. Where the headroom comes from

Mean of each phase's last 5 rounds, %:

| α | phase | attack | G's rule | G | FLTrust | median | trimmed mean | Krum |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 1 | `sign_flipping` | FLTrust | 74.62 | 74.62 | 41.27 | 42.77 | 19.26 |
| 0.1 | 2 | `gaussian_noise` | TM | 87.04 | 81.29 | 66.26 | 81.41 | 20.50 |
| 0.1 | 3 | `trim_attack` | FLTrust | 80.66 | 81.46 | 52.81 | 49.74 | 21.17 |
| 0.5 | 1 | `sign_flipping` | Krum | 88.15 | 85.00 | 85.35 | 86.81 | 88.15 |
| 0.5 | 2 | `gaussian_noise` | TM | 90.58 | 85.62 | 89.74 | 90.01 | 87.39 |
| 0.5 | 3 | `trim_attack` | TM | 89.13 | 86.67 | 88.30 | 88.78 | 88.45 |

**At α = 0.1, all of H_T comes from phase 2.** G against fixed FLTrust, per phase:

| phase | G − FLTrust, p.p. | seeds with G ahead |
|---|---|---|
| 1 | 0.00 (same rule and state) | none, tied by construction |
| 2 | +5.75 | 10 of 10 |
| 3 | −0.80 | 3 of 10 |

**Carry-over, measured:**

- **Favourable on the switch to trimmed mean.** In phase 2, G (87.04) is 5.63 p.p. above trimmed mean run from round 1 (81.41). Starting from the state left by FLTrust helps more than the rule's own ranking suggests. G's phase 2 gain over FLTrust is 5.75 p.p., against the 4.64 p.p. of the isolated ranking.
- **Slightly unfavourable on the switch back to FLTrust.** G ends phase 3 at −0.80 p.p. against FLTrust run throughout, with mixed signs across seeds. So the sequence of static winners is not the best sequence in every phase, which confirms that G is a reference and not a ceiling.
- **Every phase's rule still wins its phase among the fixed arms,** on both α. At α = 0.1, phase 2, that win is by 0.12 p.p. (81.41 against 81.29), and at α = 0.5, phase 3, it is the expected near-tie.

## 3. Secondary metrics

| metric | α | reference | H_T, p.p. | CI95 | Wilcoxon p |
|---|---|---|---|---|---|
| Trailing mean, rounds 41–45 | 0.1 | FLTrust | −0.80 | (−2.29; +0.68) | 0.23 |
| Trailing mean, rounds 41–45 | 0.5 | trimmed mean | +0.35 | (+0.30; +0.39) | 0.0020 |
| Final round | 0.1 | FLTrust | +0.96 | (−5.11; +7.04) | 0.90 |
| Final round | 0.5 | Krum | −0.42 | (−1.20; +0.36) | 0.084 |

**The end-of-run metric misses the headroom at α = 0.1,** as the plan anticipated: the trailing mean sees only the last phase, and the CI includes zero. This supports using the phase score when the best rule changes within a run.

## 4. Sensitivity to the window, not in the plan

FLTrust oscillates between consecutive rounds, by about 6 p.p. in the mean curve at α = 0.1 in phase 3 (`analysis/curves.png`), and a 5-round window holds an unequal number of high and low rounds. In the fixed FLTrust arm this window takes 3 high rounds in phase 2 and 3 low rounds in phases 1 and 3.

This was noticed after seeing the curves, so the check below is post hoc. It is descriptive and does not replace the planned metric. With the last 6 rounds of each phase, which balance high and low rounds:

| α | H_T, p.p. | CI95 | G − best fixed per phase, p.p. |
|---|---|---|---|
| 0.1 | +1.80 | (+1.09; +2.52) | 0.00, +6.55, −1.15 |
| 0.5 | +0.79 | (+0.77; +0.81) | +1.43, +0.59, +0.35 |

The conclusion does not change. The planned window understates the phase 2 gain and the phase 3 loss at α = 0.1. **For oscillating rules, the final round and odd windows are sensitive to parity.** The final-round CI at α = 0.1 is the widest in this table set, so that metric is not informative here.

## 5. Limits

- Built by design, on seeds whose results were known, so it says nothing about how often such regimes occur.
- One attack order, one phase length, one partition per α, MNIST only.
- G is not an upper bound, so H_T does not measure the most an adaptive system could gain.
- Seeds 42–51 are E4b's development seeds, so E4a cannot serve as E4b's confirmation.
- The scenario was not retuned. Any other sequence gets a new name and its own addendum, and counts as E4b development.

## 6. Feed to gate P-2

This covers only E4a's part of question 4 in `results/p1_plan/PLAN.md`, Sec. 6: the metric responds to constructed headroom, and the end-of-run metric does not at α = 0.1. **Gate P-2 itself stays open,** since it also needs E1–E3 on CIFAR-10 from the campaign.
