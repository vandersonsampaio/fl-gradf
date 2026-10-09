> English translation of `PLANO.md`. The Portuguese original is the frozen record (its hash is in `PLANO.sha256`); if the two ever disagree, the original prevails.

# Plan — B2.3b: last TD3 steelman (normalized reward)

**Status:** FINAL before the data (hash in `PLANO.sha256`). Exploratory.
**Date:** 2026-09-30
**Scope:** definitive Gate B-a.
**Record:** local, with a hash; committing is up to the author.

## 1. Question

In B2.3 (lr 1e-3, `learning_starts` 10) the TD3 policy left its initialization, but saturated at a **constant corner** of the Box, independent of the state, and collapsed under EB when a₅ ≈ 0. The conditioning hypothesis is the scale of the official reward: the **sum** of the loss over ~156 test batches, with tens to hundreds per round and no normalization. With a well-conditioned reward, does TD3 learn a **state-dependent** policy and beat the fixed action?

## 2. Configuration (a single one, fixed before the data)

| parameter | official | B2.3 | **B2.3b** |
|---|---|---|---|
| reward | raw | raw | **normalized** (`VecNormalize`: `norm_reward=True`, `gamma=0.99`, `clip_reward=10`; `norm_obs=False`) |
| learning_rate (actor and critic) | 1e-5 | 1e-3 | **1e-4** |
| learning_starts | 100 | 10 | 10 |
| others (MlpPolicy [256,128], buffer 1000, batch 64, train_freq 3, noise N(0; 0.1), γ 0.99) | — | same | same |

- **Observations are not normalized**, so the actor can still be evaluated with the B2.2 tools.
- The reward recorded in the JSON is the **raw** one. SB3 stores the raw reward in the replay buffer and normalizes it when sampling for training; checked on 2026-09-30: sd 86 → 0.5.
- `learning_starts = 10` is kept equal to B2.3 so that the only differences are the normalization and the lr.

## 3. Grid

- Seeds **100–104** (already used; exploratory), LMP and EB → **10 runs** (~15–16 GPU hours).
- Pairing by (attack, seed) with Step 2's `fixed` and `td3` and with the B2.3 steelman.
- Runner `scripts/passo2_oficial/run_b23b.py`, which reuses `run_b21.py` without changing it; launcher `run_grid_b23b.sh` (resumable, 6 processes).

## 4. Analysis (pre-written in `scripts/passo2_oficial/analisar_b23b.py`)

- **Metrics:** primary, median accuracy over rounds 401–500; secondary, mean 451–500.
- **Comparisons** (n = 10 pairs, two-sided Wilcoxon, Δ, CI95, d; Holm per attack, descriptive):
  - C1: B2.3b − fixed;
  - C2: B2.3b − official td3;
  - C3: B2.3b − B2.3 steelman.
- **Mechanism:**
  - drift_500 = mean |π₅₀₀(s) − π₀(s)|;
  - **sd_estados** = mean, over the 5 dimensions, of the standard deviation of π₅₀₀(s) across the states of rounds 401–500. It is the constancy statistic: S_swap saturates with the tanh;
  - S_swap, reported without a criterion;
  - drift curve per checkpoint.

## 5. Definitive Gate B-a criterion (fixed before the data)

"The steelman learns and beats the fixed action" requires **all three** conditions:
1. C1 with Δ > 0 and Wilcoxon p < 0.05 (primary);
2. median drift_500 > σ_a = 0.0475;
3. median sd_estados > σ_a = 0.0475 (the policy depends on the state).

- **All three hold** → confirm in **B2.3c** (seeds 115–124, confirmatory, its own pre-registration).
- **C1 does not hold** → **B-a definitive**: neither steelman beats the fixed action. B2.3c does not run.
- **C1 holds without 2–3** → investigate (a gain without a state-dependent policy). It **does not trigger** B2.3c.

## 6. Termination clause

B2.3b is the **last steelman**. Whatever the result, there is no other TD3 configuration in this line. **Both** steelmen (B2.3 and B2.3b) are reported in P2.

## 7. Declared limitations

- Exploratory, n = 10 pairs, Step 2 baselines already inspected.
- Two changes relative to B2.3 (normalization and lr). The design does not separate the effects, because the question is whether **any** well-conditioned configuration learns, not which factor matters.
