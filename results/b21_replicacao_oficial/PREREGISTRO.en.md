> English translation of `PREREGISTRO.md`. The Portuguese original is the frozen record (its hash is in `PREREGISTRO.sha256`); if the two ever disagree, the original prevails.

# Pre-registration — B2.1 + B2.2: confirmatory replication in the official AdaAggRL

**Status:** FINAL, frozen before any grid run (hash in `PREREGISTRO.sha256`).
**Date:** 2026-09-27
**Scope:** experiments B2.1 (confirmatory) and B2.2 (mechanistic hypotheses, confirmatory together with B2.1).
**Background:** Step 2 (`results/frente1_passo2_oficial/`, seeds 100–104): H1 inconclusive because of a late reset; the TD3 policy was, on an exploratory basis, independent of its input.
**Record:** local, with a hash; committing is up to the author. Changes after the grid starts only as a dated, hashed addendum.

## 1. Questions

- **B2.1:** in the published code and horizon, is the fixed action equivalent (±1.0 p.p.) to AdaAggRL's TD3?
- **B2.2:** is the learned TD3 policy essentially the initial one and independent of its input?

## 2. Code

Identical to Step 2 (official code in `external/AdaAggRL`, commit `27b9c18`; venv `external/.venv_adaaggrl`; minimal adjustments listed in `scripts/passo2_oficial/run_oficial.py`). New runner `scripts/passo2_oficial/run_b21.py`, which reuses the Step 2 runner without changing it and only adds logging: the states observed before each action and TD3 actor checkpoints at step 0 and every 50 steps. Conditions and hyperparameters identical to Step 2 (`fixed` = [0.475]×5; `td3` as in the official `main.py`).

## 3. Choice of the primary metric (declared)

The primary metric was **chosen after inspecting the already-used seeds 100–104**, to neutralize the late-reset artifact seen in Step 2 (the official environment re-initializes the model when the reward is < −80). On seeds 100–104, the standard deviation of the fixed − td3 differences was:

| metric | sd of the differences | TOST (already-used seeds, **not confirmatory**) |
|---|---|---|
| mean 451–500 (Step 2) | 1.84 p.p. | p = 0.18 |
| median 451–500 | 1.14 p.p. | p = 0.03 |
| **median 401–500** | **0.32 p.p.** | p < 0.001 |

Justification, independent of the result: the 100-round median does not condition on resets (which are a consequence of the action and cannot be excluded without bias), and a late reset typically contaminates ~30 of the 100 rounds, below the median's breakdown point. Confirmation comes **only** from the new seeds 105–114.

## 4. Grid

- MNIST, q = 0.5, 500 rounds, 100 clients, 10% per round, 20 attackers (same as Step 2).
- Attacks: LMP and EB. Conditions: `td3` and `fixed`.
- **Seeds: 105–114** (new, reserved for confirmation in the official code).
- Total: 10 × 2 × 2 = **40 runs**, 6 in parallel, ~60 h wall clock (measured in Step 2: ~9 h per batch of 6).
- Truncation at 500 steps (SB3's td3 runs 501, Step 2 Addendum 1).

## 5. B2.1: hypothesis and criteria

**Primary metric:** median test accuracy over rounds 401–500.
**Unit:** (attack, seed) pair, n = 20. D = metric(fixed) − metric(td3).

**H1.** The fixed action is equivalent to TD3.
- Paired TOST (t), margin ±1.0 p.p., α = 0.05 → both nulls rejected: **equivalence confirmed**.
- Paired Wilcoxon p < 0.05 with D < 0 → **TD3 contributes**.
- Wilcoxon p < 0.05 with D > 0 → the fixed action is better; report as "does not contribute", without claiming "hurts" without replication.
- None → **inconclusive**; report Δ, CI95 and d.
- Per attack: Δ, CI95, d, Wilcoxon with Holm (descriptive).

**Secondary:** the same procedure with the mean over 451–500 (Step 2 metric, for continuity); number of resets and weight mass on attackers per condition (descriptive).

## 6. B2.2: mechanistic hypotheses

σ_a = 0.1 × 0.95 / 2 = **0.0475**: standard deviation of SB3's exploration noise converted to action units. Family H3–H5, **Holm**, α = 0.05, one-sided tests.

**H3 (replication of the exploratory finding).** For each seed, r = Pearson correlation between the actions **executed** by td3 under EB and under LMP, rounds 101–500, 5 dimensions flattened. Confirmed if the one-sided Wilcoxon on atanh(r) − atanh(0.9) > 0 has p < 0.05 (n = 10).
*Declared limitation:* the executed actions include the same noise sequence in both conditions (same seed). H3 replicates the finding, but the clean test of input independence is H5.

**H4 (the policy does not move away from the initial one).** For each td3 run, drift = mean of |π₅₀₀(s) − π₀(s)| over the states observed in rounds 401–500 and the 5 dimensions, with the deterministic π (no noise) rebuilt from the checkpoints. Confirmed if the one-sided Wilcoxon on drift − σ_a < 0 has p < 0.05 (n = 20).

**H5 (the policy does not depend on its input).** For each td3 run, S_swap = mean of |π₅₀₀(s_t) − π₅₀₀(s′_t)|, where s′_t is the state of the same round in the run of the **other attack** with the same seed, t in 401–500. Confirmed if the one-sided Wilcoxon on S_swap − σ_a < 0 has p < 0.05 (n = 20).

**Reported without a test:** S_shuffle (state from another randomly drawn round of the same run), S_swap of the initial actor π₀ and the distance of π₀ to the center, as references.

## 7. Gate (B-b)

- H1 with equivalence → the P2 thesis ("TD3 does not contribute") becomes confirmatory in the official code.
- H1 inconclusive → report "no evidence of contribution", with B2.2 as the main evidence.
- H4 and H5 confirmed → "the policy does not learn and does not depend on its input" becomes confirmatory.
- The steelman (B2.3) remains necessary, whatever the result.

## 8. Rules

- Pre-written analysis in `scripts/passo2_oficial/analisar_b21.py`, run once, with the 40 runs complete.
- Runs that fail due to an environment error are repeated with the same seed and recorded. Collapses and resets count as results.
- No accuracy is inspected before the grid ends. Monitoring reports only health and progress.
