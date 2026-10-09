# Result — B2.1 + B2.2: confirmatory replication in the official AdaAggRL

**Date:** 2026-09-30
**Pre-registration:** `PREREGISTRO.md` (English translation in `PREREGISTRO.en.md`; hash and code hashes in `PREREGISTRO.sha256`; `analisar_b21.py` and `analisar.py` checked against the hash before running: SUCCESS).
**Grid:** 40/40 runs (seeds 105–114 × LMP/EB × fixed/td3; MNIST, q=0.5, 500 rounds), 2026-09-27 13:30 → 2026-09-30 03:37, 0 failures. 40 `obs`, 20 final actor checkpoints.
**Analysis:** `scripts/passo2_oficial/analisar_b21.py`, run once → `analise.txt`, `resumo_runs.csv`, `b22_mecanismo.csv`.

---

## 1. B2.1 — confirmatory result

Primary metric: median accuracy over rounds 401–500. D = fixed − td3, n = 20 pairs (attack × seed).

| | Δ (p.p.) | CI95 | d | TOST ±1.0 p.p. | Wilcoxon |
|---|---|---|---|---|---|
| **pooled (primary)** | **−0.22** | (−1.01; +0.57) | −0.13 | **p = 0.027** | p = 0.064 |
| LMP | +0.10 | (−0.02; +0.22) | +0.60 | — | p = 0.11 (Holm 0.21) |
| EB | −0.54 | (−2.26; +1.17) | −0.23 | — | p = 0.38 (Holm 0.38) |
| pooled (secondary: mean 451–500) | −0.12 | (−0.67; +0.43) | −0.10 | p = 0.002 | p = 0.097 |

→ **PRE-REGISTERED VERDICT: EQUIVALENCE CONFIRMED.** The fixed action and the published TD3 are equivalent within ±1.0 p.p. in the official code and horizon.

**Note on the CI:** TOST with α = 0.05 corresponds to the **90% CI**, which is (−0.87; +0.43) p.p. and lies entirely within ±1.0. The 95% CI in the table touches −1.01 and is reported for completeness.

## 2. B2.2 — mechanistic hypotheses (Holm over H3–H5)

σ_a = 0.0475 (SB3 exploration noise in action units).

| hypothesis | statistic | result |
|---|---|---|
| **H3** corr. of EB × LMP actions > 0.9 | median r = 0.991 (per seed: 0.993 · 0.991 · 0.994 · 0.990 · 0.922 · **0.899** · 0.989 · 0.988 · 0.993 · 0.992) | Holm p = 0.002 → **CONFIRMED** |
| **H4** drift \|π₅₀₀ − π₀\| < σ_a | median 0.0097; max. 0.0147 (≈ 1/5 of the noise) | Holm p < 0.001 → **CONFIRMED** |
| **H5** S_swap \|π₅₀₀(s) − π₅₀₀(s′)\| < σ_a | median 0.0055; max. 0.0062 | Holm p < 0.001 → **CONFIRMED** |

References (no test): S_shuffle = 0.0061; **S_swap of the initial actor π₀ = 0.0055**, identical to the final actor's; |π₀ − center| = 0.0265.

**Reading:** the final policy's sensitivity to its input is the same as that of a **freshly initialized** network (0.0055 vs. 0.0055). Over 500 rounds, the policy drifts about 1/5 of its own exploration noise and does not start responding to the state. Replacing the whole observation with that of another attack changes the action by ~0.006 on a 0–0.95 scale.

## 3. Exploratory findings (not confirmatory)

- **The only discrepant pair comes from a late reset, again.** In EB, seed 110, `fixed` reset at round 392. The recovery took **52% of the 401–500 window** (rounds with accuracy < 0.9), above the median's breakdown point. The pre-registration premise ("a late reset contaminates ~30 rounds") failed in this case. Even with the pair included, TOST closes. Without it (sensitivity, **not** pre-registered): Δ = +0.15 p.p., sd 0.24.
- **Extra resets** (mean per run): EB td3 2.2 vs. fixed 0.2; LMP td3 0.6 vs. fixed 0.8. The td3 resets concentrate at the beginning (e.g. s110: rounds 28, 41, 87), in the random warm-up phase, as in Step 2.
- **Weight mass on attackers** ≤ 0.0001 in all conditions: the fixed skeleton excludes the attackers with or without TD3.
- The lowest H3 correlation (s110, 0.899) coincides with the pair in which the two runs had different reset histories. This is consistent with the correlation coming from the shared noise sequence, perturbed by the resets.

## 4. Consequences (Gate B-b)

- **B-b closed in favor of the P2 thesis:** in the published code and horizon, AdaAggRL's TD3 is **equivalent** to a fixed action at the center of the action space (confirmatory, new seeds 105–114).
- **Confirmatory mechanism:** the policy does not move away from its initialization and **does not depend on its input** (H4, H5), with a sensitivity equal to that of a random network.
- **What cannot be claimed yet:** that "RL does not pay off" in general. That depends on the steelman (B2.3, Gate B-a): this is the **published** configuration.
- **Methodological lesson for the next pre-registrations:** fixed-window "reset-robust" metrics can fail when recovery is slow. It is worth considering an area-under-the-curve metric, or a pre-declared exclusion of post-reset windows with a sensitivity analysis.
