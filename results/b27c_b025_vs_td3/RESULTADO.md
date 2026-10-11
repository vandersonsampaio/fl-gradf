# Result — reformulated B2.7c: does the fixed threshold b = 0.25 beat TD3 at H = 150?

**Date:** 2026-10-11.

**Pre-registration:**
- Plan: `PLANO.md` (frozen 2026-10-11 11:15:51 -0300 before the code and any run; hash in `PLANO.sha256`; English translation in `PLANO.en.md`).
- Addendum 1: `ADENDO1.md`, with the code hashes and the smoke test. It was frozen at 11:18:56, before any run on seeds 82–91 (`ADENDO1.en.md`).
- The frozen originals are in Portuguese; the `.en.md` files are translations added afterwards. **Not committed yet.**

**Grid:**
- 60 runs (2 systems × 3 cells × seeds 82–91, H = 150), from 11:18:56 to 13:31:17.
- **No failures and no re-runs.** No accuracy was inspected before 60/60.

**Outputs:** `analise.txt`, `deltas.csv`, `grade_raw.csv`, `verificacao.txt`.

## Short answer

**Yes. A fixed threshold b = 0.25, chosen in advance, beats the skeleton's TD3 in all 3 cells on fresh seeds.**
- The pre-registered primary hypothesis is **confirmed in all cells**: Holm rejects H0 in 3/3, with D = +6.0 to +9.1 p.p.
- The secondary pooled hypothesis is also confirmed: +8.1 p.p., one-sided p = 0.0005.
- The constant wins in **30 of 30 seed × cell pairs**.

This settles the open point from B2.7b, where b = 0.25 had been chosen in hindsight. Here, choosing it in advance did not shrink the effect; the effect came out larger.

## Check (PLANO §3)

A 15-round smoke test on seed 42, `label_flipping` α 0.05, through the new code:
- td3_ref reproduces B2.7's value exactly: 0.7194, |Δ| = 0;
- fixed_b025 reproduces B2.7b's value exactly: 0.7254, |Δ| = 0.

## Primary result (H = 150, seeds 82–91, n = 10 per cell)

Mean accuracy (%):

| cell | td3_ref | **fixed_b025** |
|---|---|---|
| `label_flipping` α 0.05 | 66.7 | **75.8** |
| `label_flipping` α 0.1 | 69.7 | **78.8** |
| `low_mag_backdoor` α 0.05 | 81.9 | **87.9** |

D = fixed_b025 − td3_ref, paired by seed (p.p.). The p-values are one-sided; Holm is applied at family α = 0.05:

| cell | D | CI95 | d_z | p (t, one-sided) | Holm | p (Wilcoxon, one-sided) | seeds with D > 0 |
|---|---|---|---|---|---|---|---|
| `label_flipping` α 0.05 | **+9.06** | (+3.39; +14.73) | 1.14 | 0.0028 | rejects | 0.0010 | 10/10 |
| `label_flipping` α 0.1 | **+9.09** | (+2.44; +15.73) | 0.98 | 0.0064 | rejects | 0.0010 | 10/10 |
| `low_mag_backdoor` α 0.05 | **+6.03** | (+2.75; +9.32) | 1.31 | 0.0012 | rejects | 0.0010 | 10/10 |

**Pre-registered verdicts:**
- **H1 (primary): CONFIRMED in all cells.** Holm rejects in 3/3, and no mean D is negative.
- **H1b (secondary, pooled):** the per-seed mean of D over the 3 cells is +8.06 p.p. (n = 10), one-sided p = 0.0005. **CONFIRMED.**

## Descriptive (not pre-registered as verdicts)

**Variance (where the difference comes from):**
- TD3 is far more variable across seeds than the constant:
  - sd(td3_ref) = 8.0 / 8.8 / 4.6 p.p.;
  - sd(fixed_b025) = 1.1 / 2.9 / 0.4 p.p.
- The worst seed is 51.9 / 47.8 / 72.1% for TD3, against 73.9 / 74.9 / 87.4% for the constant.
- Per seed, D ranges from ~0 (min +0.01 p.p.) to +33.6 p.p.
- So the gap combines two things. The constant has a better center, and TD3 has a heavy lower tail: some seeds lose 15–34 p.p.

**Shorter horizons (D in p.p.):**

| cell | H = 15 | H = 50 | H = 150 |
|---|---|---|---|
| `label_flipping` α 0.05 | +5.5 | +8.4 | +9.1 |
| `label_flipping` α 0.1 | −0.7 | +3.4 | +9.1 |
| `low_mag_backdoor` α 0.05 | +8.4 | +7.2 | +6.0 |

- The advantage is not transient: in label_flipping it **grows** with the horizon. More rounds of TD3 learning do not close the gap.
- The only negative value (−0.7 p.p., `label_flipping` α 0.1, H = 15) is not significant.

**Comparison with B2.7b** (seeds 42–51, b chosen in hindsight):

| cell | B2.7b | this experiment |
|---|---|---|
| `label_flipping` α 0.05 | +6.86 | +9.06 |
| `label_flipping` α 0.1 | +4.42 | +9.09 |
| `low_mag_backdoor` α 0.05 | +4.45 | +6.03 |

The hindsight-selection worry (winner's curse) does not materialise: the effect is larger on fresh seeds, not smaller. Mostly this reflects worse TD3 runs on seeds 82–91, which fits the TD3 variance noted above.

## Reading for P2

- Together, B2.7a/b and this experiment close the question for the own-framework skeleton at H = 150:
  - TD3 does not learn a state-dependent policy (B2.7a);
  - its gain over the center is no larger than a shifted constant (B2.7b);
  - **a simple constant fixed in advance (b = 0.25) beats it by 6–9 p.p., confirmed with a pre-registered test on fresh seeds.**
- **What P2 can state:**
  - "In the cells most favourable to TD3, a fixed detection threshold chosen a priori beats the learned agent by 6–9 p.p. (Holm-corrected one-sided tests, 3/3 cells, 30/30 paired runs)."
  - The threshold is the decisive component (B2.6); learning neither finds the right threshold nor makes up for not having it.
- TD3's seed-to-seed instability (sd 5–9 p.p. vs 0.4–3 p.p. for the constant) is a second practical cost on top of the lower mean.

## Limitations

- **Scope:**
  - own-framework skeleton (ablation learner), logistic MNIST, 10 clients, 2 Byzantine;
  - only 3 cells, chosen for being the most favourable to TD3 in B2.7. That makes the test conservative for H1, but it does not generalise to other attacks or α values.
- **b = 0.25 is not shown to be optimal.** It was fixed a priori from B2.6. A finer threshold sweep could find a better constant, which would only widen the gap.
- **n = 10 per cell.** The CI95s are wide (e.g. +2.4 to +15.7 p.p.), so the size of the effect is imprecise even though its direction is clear.
- **TD3 is compared in its reference configuration (`td3_ref`).** B2.3/B2.3b's steelman variants were run in the official code and were not re-tested here.
- **RNG caveat (B2.7):** results are reproducible via `keras.utils.set_random_seed(seed)` before each system, as verified by the smoke test. The framework's learner `seed` alone does not seed the global RNG.
