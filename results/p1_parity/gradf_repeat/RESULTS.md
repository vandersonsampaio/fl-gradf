# Result — repetition of GRADF on MNIST under step parity

**Date:** 2026-11-02. **Status: complete, not confirmatory.** Plan: `PLAN.md`, frozen before the runs. Code hashes match the plan. 20/20 jobs, 0 failures, 06:51–16:06, 5 processes, alongside the W1 rerun. Analysis: `scripts/p1/analyze_gradf_repeat.py`, output in `analysis/`.

**In short:** GRADF − Random repeats, and by the frozen rule it is robust to run-to-run variance. It does not show that selection adapts. GRADF votes FLTrust in 87% of rounds, and Random also beats the fixed rules, so GRADF − B₋ₛ cannot be read as selection gain.

## 1. Regression

Random equals `../mnist_rq3` in 209 of 210 pairs, in both executions. The remaining pair (α 0.5, `low_mag_backdoor`, seed 48) differs by 1.11e-16.

## 2. Repetition (18 confirmatory cells, p.p.)

| execution | GRADF − Random (CI95) | GRADF − B₋ₛ (CI95) |
|---|---|---|
| first run | +3.48 (+2.13; +4.84) | +7.89 (+5.86; +9.92) |
| new 1 | +3.04 (+2.13; +3.95) | +7.45 (+6.15; +8.75) |
| new 2 | +3.37 (+1.88; +4.86) | +7.78 (+6.11; +9.44) |

Random − B₋ₛ is +4.41 (+2.81; +6.01) in all three, since Random repeats.

- **Criterion 1:** the CI95 is above zero in all 3 executions. Met.
- **Criterion 2:** SD across executions 0.23, below ½|mean| = 1.65 (mean +3.30). Met.
- **Verdict: robust to run-to-run variance.**

**Limits of this verdict:**
- The executions are far from independent. GRADF differs from the first run in only 64 and 51 of 210 pairs (mean |d| 2.27 and 1.87 p.p.). The SD of 0.23 is well below the 1.77 D2 measured on the earlier code.
- Robustness here covers the DQN's non-determinism on the same seeds and partitions (one partition per α). It is not generalization to new seeds or partitions.
- Some seeds vary more: seed 42 gives +5.15, +2.08 and +0.79.

## 3. Mechanism (descriptive, new executions)

| | GRADF | Random |
|---|---|---|
| FLTrust vote share | 86.9% | 17.2% |
| Largest other vote share | 3.3% | 19.0% |
| Rounds rejected by Layer 4 | 7.6% | 21.5% |
| Cells above the best of the 4 D_R rules (21 cells) | 14 | 12 |
| Cells above the best of all 7 fixed rules (21 cells) | 9 | 4 |

- **GRADF is close to a constant vote for FLTrust**, with 64–100% per cell.
- **Fixed FLTrust is well below GRADF even where GRADF nearly always votes FLTrust.** In the 8 cells with at least 90% FLTrust votes, GRADF is +5.81 p.p. above fixed FLTrust on average, and above it in all 8. Fixed rules do not pass through Layers 1, 2 and 4.
- **Random is also above the fixed rules.** It is 12.0 p.p. above the vote-weighted mean of the fixed rules on the 18 cells. This line was not in the plan's list. It cannot separate the layers from the effect of switching rules across rounds.
- **Layer 1 rejects no honest client.** It acts almost only on `gaussian_noise` (both attackers in every round) and `trim_attack` (about 0.3 per round).
- **Layer 4 rejects 2.8 times as many of Random's rounds**, which fits GRADF's votes being better aggregates, on a root of 100 samples.
- **The gain over Random is concentrated:** up to +13.1 p.p. at low α under `sign_flipping`, `krum_collusion` and `label_flipping`; under +3 p.p. in every α 0.5 cell.
- **`fltrust_aligned` goes the other way:** −4.3 to −6.6 p.p. against Random. That attack targets the rule GRADF favours, and it lies outside the 18 cells.

## 4. What this means for the pre-registration

- **The comparison against Random is like for like.** Both systems pass through the same layers, so GRADF − Random reflects the votes.
- **The comparison against B₋ₛ is not.** It mixes selection with the hardening layers. This plan does not separate them, so "capture" should not be claimed from it.
- **Open question, not answered here:** does GRADF beat a fixed FLTrust that passes through the same layers? If not, its gain is the choice of one good rule plus hardening, not adaptation. An ablation needs its own plan.
- **Other systems:** FedStrategist and AdaAggRL do not pass through these layers. Each has its own `_run_round`, and `exp10` never uses `HardeningPipeline`. So FedStrategist − Random and AdaAggRL − Random in `../RESULTS.md` (Sec. 4) also mix selection with hardening.

## Outputs

`analysis/analysis.txt`, `repetition_gradf_vs_random.csv`, `vote_shares.csv`, `mechanism_per_cell.csv`, `gradf_vs_fixed_fltrust.csv`; `raw/`.
