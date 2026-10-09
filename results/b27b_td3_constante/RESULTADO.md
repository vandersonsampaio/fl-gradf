# Result — B2.7a/b: does TD3 at H = 150 depend on the state, or did it just find a better constant?

**Date:** 2026-10-03.
- Plan: `PLANO.md` (English translation in `PLANO.en.md`), commit `d676ec3`.
- Addendum 1 (B2.7a code; `ADENDO1.en.md`): commit `53c4722`.
- Addendum 2 (explanatory B2.7b, written after seeing B2.7a; `ADENDO2.en.md`): commit `14dbefd`.

**Grids:**
- B2.7a: 30 runs, from 10/02 20:36 to 22:19.
- B2.7b: 120 runs, from 10/02 23:03 to 10/03 04:20.
- No failures.

**Outputs:** `b27a_analise.txt`, `b27a_por_run.csv`, `a_td3.csv`, `b27b_analise.txt`, `b27b_deltas.csv`, `b27b_grade_raw.csv`.

## Short answer

**The skeleton's TD3 does not learn a policy, and the little it learns is a constant worse than the best simple constant.**
- The policy is independent of the state (B2.7a).
- The gain over the center is indistinguishable from a shifted constant (criterion 2 met).
- The gain is also indistinguishable from a TD3 without actor learning (td3_ref ≈ td3_frozen in the 3 cells).
- A fixed threshold b = 0.25 beats TD3 by **4.4 to 6.9 p.p. in the 3 cells**, with the whole CI95 below 0.

## B2.7a: mechanism (descriptive)

- **Check:** exactly reproduces B2.7's `td3_ref` (max. |Δacc| = 5.6e-17).

| cell | drift (π₁₅₀ − π₀) | sd_estados π₁₅₀ | sd_estados π₀ | distance to the center |
|---|---|---|---|---|
| `label_flipping` α 0.05 | 0.064 | 0.0005 | 0.0004 | 0.066 |
| `label_flipping` α 0.1 | 0.061 | 0.0004 | 0.0004 | 0.073 |
| `low_mag_backdoor` α 0.05 | 0.061 | 0.0004 | 0.0004 | 0.068 |

- The state dependence is ~300× smaller than σ_a = 0.15 and equal to the initial actor's.
- The actor slides to a **constant** near the center: a ≈ (0.46; 0.56; 0.50; 0.50), **b ≈ 0.45** (center 0.5).
- **B2.7c condition 1: 0/3 → B2.7c does not run.**

## B2.7b: where Δ comes from (H = 150, seeds 42–51, n = 10 per cell)

Mean accuracy per system:

| cell | fixed (b 0.5) | **fixed_b025** | fixed_b075 | fixed_td3mean | td3_frozen | td3_ref |
|---|---|---|---|---|---|---|
| `label_flipping` α 0.05 | 61.8 | **75.3** | 54.7 | 67.4 | 66.1 | 68.5 |
| `label_flipping` α 0.1 | 71.4 | **78.5** | 47.3 | 72.3 | 73.2 | 74.1 |
| `low_mag_backdoor` α 0.05 | 80.6 | **88.2** | 71.5 | 82.9 | 83.3 | 83.7 |

Paired Δs (p.p., CI95):

| comparison | LF α 0.05 | LF α 0.1 | LMB α 0.05 |
|---|---|---|---|
| td3_ref − td3_frozen (learning only) | +2.3 (−1.2; +5.9) | +0.9 (−1.7; +3.5) | +0.4 (−2.2; +3.0) |
| td3_frozen − fixed (noise + warm-up, no learning) | +4.3 (−0.04; +8.6) | +1.8 (−2.4; +6.1) | +2.7 (−0.8; +6.3) |
| td3_ref − fixed_td3mean (the learned constant) | +1.1 (−5.6; +7.7) | +1.8 (−2.3; +5.8) | +0.9 (−2.8; +4.5) |
| fixed_td3mean − fixed (just shifting the constant) | +5.6 (−0.05; +11.2) | +0.9 (−3.6; +5.5) | +2.2 (−2.6; +7.0) |
| **td3_ref − fixed_b025 (best constant)** | **−6.9 (−10.7; −3.1)** | **−4.4 (−7.3; −1.5)** | **−4.4 (−6.8; −2.1)** |

**ADENDO2 criteria (pre-registered):**
1. **"The gain is exploration, not learning": NOT met.**
   - The 1st part holds: the CI95 of td3_ref − td3_frozen contains 0 in the 3 cells.
   - The 2nd part fails: td3_frozen − fixed is positive in the 3 cells, but the CI95 touches 0 in all of them (on `label_flipping` α 0.05 the lower bound is −0.04 p.p.), and the criterion required CI95 > 0 in ≥ 1.
   - *Non-pre-registered reading:* with the literal rule "Δ > 0" (mean only), the criterion would be met (3/3 positive). The result is borderline.
2. **"Shifted constant" (fixed_td3mean ≈ td3_ref): MET.** The CI95 contains 0 in the 3 cells. The ±1 p.p. TOST does not close in any of them, so it is "indistinguishable", not "equivalent".
3. **Secondary (td3_ref > best constant): 0/3.** On the contrary, TD3 **loses** to b = 0.25 in the 3 cells, with the whole CI95 below 0.

## Reading for P2

- TD3's positive signal on `label_flipping` H = 150 (B2.7, +6.65 p.p. over the center) **is not policy learning**:
  - the policy does not depend on the state;
  - removing actor learning does not change the result detectably;
  - the learned constant reproduces TD3.
- The Δ over the center breaks down, with no statistical separation between the parts, into a small shift of the constant (b 0.5 → ~0.45) and the exploration noise.
- **"Learning amounts, at most, to tuning the threshold, and it tunes it badly."** The right direction (a smaller b) is much farther than TD3 goes: b = 0.25 gains 4–7 p.p. over TD3. This is consistent with B2.6 (b = 0.25 is the best variant, +1.6 p.p.) and reinforces the threshold as the decisive component.
- **Limitations:**
  - 3 cells chosen for their positive slope in B2.7 (the most favorable to TD3);
  - n = 10;
  - explanatory B2.7b (designed after B2.7a);
  - b = 0.25 is chosen in hindsight.
