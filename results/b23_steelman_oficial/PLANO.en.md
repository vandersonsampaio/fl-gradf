> English translation of `PLANO.md`. The Portuguese original is the frozen record (its hash is in `PLANO.sha256`); if the two ever disagree, the original prevails.

# Plan — B2.3: TD3 steelman in the official AdaAggRL

**Status:** FINAL before the data (hash in `PLANO.sha256`). Exploratory.
**Date:** 2026-09-28
**Scope:** B2.3 ("exploratory, essential") and Gate B-a.
**Record:** local, with a hash; committing is up to the author.

## 1. Question

The published TD3 configuration (lr 1e-5, 100 rounds of random warm-up, ~66 actor updates in 500 rounds) does not learn (Step 2, exploratory; B2.2 being confirmed). Given much more favorable conditions, does TD3 start learning something that depends on its input and beat the fixed action?

## 2. Configuration (a single one, fixed before the data)

| parameter | official | steelman |
|---|---|---|
| learning_rate (actor and critic) | 1e-5 | **1e-3** |
| learning_starts | 100 | **10** |
| others (MlpPolicy [256,128], buffer 1000, batch 64, train_freq 3, noise N(0; 0.1), γ 0.99) | — | same |

Justification: a 100× larger lr gives update steps of the order used in standard TD3 (SB3's default is 1e-3); a short warm-up removes the random phase that caused most of the resets under EB and increases the number of updates. A single configuration, for cost reasons (~16 h); if the result is ambiguous, the 2×2 grid is the next step.

## 3. Grid

- Seeds **100–104** (already used; exploratory), LMP and EB → **10 runs** (~16 h, after B2.1 ends).
- Pairing by (attack, seed) with the Step 2 `fixed` and `td3` runs (same environment, same attackers and partition; client sequence paired by the seed). The baselines do not need to be re-run.
- Runner `scripts/passo2_oficial/run_b23.py` (reuses `run_b21.py` without changing it; saves states and actor checkpoints every 50 steps).

## 4. Analysis (pre-written in `scripts/passo2_oficial/analisar_b23.py`)

- Primary: median accuracy over rounds 401–500 (the same as B2.1). Secondary: mean 451–500.
- C1: steelman − fixed; C2: steelman − official td3. n = 10 pairs, two-sided Wilcoxon, Δ, CI95, d; Holm per attack (descriptive).
- Mechanism: drift of the deterministic policy across checkpoints (π_t vs. π_0) and S_swap (state replaced by the other attack's), compared with σ_a = 0.0475.

## 5. Gate B-a criterion (fixed before the data)

"The steelman learns and beats the fixed action" requires **all three** conditions:
1. C1 with Δ > 0 and Wilcoxon p < 0.05 (primary);
2. median drift_500 > σ_a (the policy moved away from the initial one);
3. median S_swap > σ_a (the policy depends on its input).

- If all three hold → P2 becomes "the published configuration does not learn; well tuned, RL delivers X" (Gate B-a).
- If C1 does not hold → the P2 thesis becomes strong ("not even the steelman beats the fixed action").
- If C1 holds without 2–3 → investigate (a gain without input-dependent learning, e.g. from the shorter random warm-up).

## 6. Declared limitations

- Exploratory and with n = 10 pairs; the Step 2 baselines have already been inspected by the author. A confirmatory claim would require new seeds.
- A single configuration does not exhaust the hyperparameter space; "the steelman does not beat it" means "this favorable configuration does not beat it", not "no configuration beats it".
