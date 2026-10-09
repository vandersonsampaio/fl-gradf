> English translation of `ADENDO2.md`. The Portuguese original is the frozen record (its hash is in `ADENDO2.sha256`); if the two ever disagree, the original prevails.

# Addendum 2 — B2.7b revised: does TD3's Δ come from a shifted constant or from the noise?

**Date:** 2026-10-02 23:02:53 -0300, before any B2.7b run.
**Declaration:** this addendum was written **after seeing B2.7a** (`b27a_analise.txt`): the TD3 policy at H = 150 does not depend on the state (sd_estados ~0.0005 ≪ σ_a = 0.15, equal to the initial actor). The actor drifts to a constant near the center (drift ~0.06). Therefore B2.7b becomes **explanatory**, not a test of "TD3 beats the best constant". B2.7c is already ruled out (condition 1 of PLANO §4 not met, 0/3).

## Question

Where does Δ(td3_ref − fixed) at H = 150 come from: a **shifted constant** or the **exploration noise** (TD3 executes π(s) + N(0; 0.15) every round)?

## Systems (they replace the arms of PLANO §3)

3 cells (`label_flipping` α 0.05 and 0.1; `low_mag_backdoor` α 0.05) × seeds 42–51 × H = 150:

| system | what it isolates |
|---|---|
| **td3_frozen** | **main control:** the same agent, same warm-up and same noise process, **without learning the actor**. `train_step` runs normally (critic, buffer and RNG consumed as in td3_ref) and the actor (W, b and targets) is restored to the initial one after each step. This is equivalent to actor lr = 0. The actor update does not consume RNG, so the noise sequence is the same as td3_ref's. |
| **fixed_td3mean** | the effect of the shifted constant: constant, deterministic action = **per-cell** mean of a_TD3 (π₁₅₀ on the states of rounds 101–150, `a_td3.csv`) |
| **fixed_b025**, **fixed_b075** | threshold sensitivity: a = [0.5]×4, b = 0.25 and 0.75, as in the plan |
| fixed (center) and td3_ref | already exist in B2.7 (`results/b27_horizonte/grade_raw.csv`); td3_ref was exactly reproduced in B2.7a |

- **Grid:** 4 × 3 × 10 = **120 runs**. With the GPU busy with B2.4r, the runs with inversion cost ~1,600–2,000 s, which gives **~5–6 h wall clock with 12 processes**. The ~2 h estimate underestimates the CPU contention.
- **Checks (15-round smoke test):**
  - the constant [0.5]×5 through the new path reproduces B2.7's `fixed` (|Δ| = 0);
  - in td3_frozen, the actor stays identical to the initial one after 8 training steps.

## Criteria (fixed now, before any B2.7b run)

Unit: seed, n = 10 per cell. Paired Δ, t CI95, H = 150.

1. **"The gain is exploration, not learning"** if:
   - the CI95 of Δ(td3_ref − td3_frozen) **contains 0 in the 3 cells**, **and**
   - td3_frozen − fixed > 0 in **at least 1** of them.

   Operationalized as Δ > 0 with CI95 > 0, the same significance rule as B2.7.
2. **"Shifted constant"** if fixed_td3mean ≈ td3_ref, operationalized as: the CI95 of Δ(td3_ref − fixed_td3mean) **contains 0 in the 3 cells**. The ±1 p.p. TOST is reported as descriptive.

The two criteria are not mutually exclusive and are reported together.

**Secondary (the original PLANO §3 criterion):** td3_ref vs. the best constant in hindsight among {fixed, fixed_b025, fixed_b075, fixed_td3mean}, with Δ > 0 and CI95 > 0 in ≥ 2 of 3 cells.

**Also reported:** fixed_td3mean − fixed; td3_ref vs. b = 0.25 and b = 0.75.

## Code at freeze time
1c30f5bc8ff69ca495971ca8c3d02ac7298939f0be38465c436c002a5d06143f  scripts/b27b_explicativo.py
5cc052c70579c1fa4e86649ba8c2e96215cb8b829fe9d836f80c6838a9d4d747  scripts/run_grid_b27b.sh
