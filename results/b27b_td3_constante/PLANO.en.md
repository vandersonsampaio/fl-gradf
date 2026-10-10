> English translation of `PLANO.md`. The Portuguese original is the frozen record (its hash is in `PLANO.sha256`); if the two ever disagree, the original prevails.

# Plan — B2.7a/b: does TD3 at H = 150 depend on the state, or did it just find a better constant?

**Status:** FINAL before any run and **before seeing B2.7a** (hash in `PLANO.sha256`). **Exploratory.** The code does not exist yet: it will be written, tested and have its hash recorded in an addendum **before** the launch, without changing anything in this plan.
**Date:** 2026-10-02
**Background:** B2.7 (`results/b27_horizonte/`). TD3 (`td3_ref`) passed the criterion in only 1 of 8 cells at H = 150 (`label_flipping` α 0.05, +6.65 p.p.), among 72 uncorrected tests. Moreover, the comparison was against the **center** (`fixed`, b = 0.5), while LinUCB and DQN were compared against the **best constant in hindsight**.

## 1. Cells, horizon and seeds

- **Cells:** the 3 with a positive slope of Δ(TD3 − fixed) per log H in B2.7:
  - `label_flipping` α 0.05;
  - `label_flipping` α 0.1;
  - `low_mag_backdoor` α 0.05.
- **H = 150**, seeds **42–51** (B2.7's, for pairing).
- **Regime:** identical to B2.7: the ablation learner, logistic MNIST, 10 clients, Byzantine [0, 1], root 100, `keras.utils.set_random_seed(seed)` before each system.

## 2. B2.7a: TD3 mechanism (descriptive)

B2.7 did not record actions or actor parameters. So `td3_ref` is re-run on the 3 cells × 10 seeds (**30 runs**), with per-round logging of:
- the mean state (`mean_state`);
- the executed action;
- the actor parameters (`W_actor`, `b_actor`) at rounds 0, 50, 100 and 150.

**Check:** the accuracy at H = 15, 50 and 150 must exactly reproduce (|Δ| < 1e-9) that of `td3_ref` in B2.7's `grade_raw.csv`. If it does not, B2.7a is invalidated and reported as such.

**Measures per run**, over the states observed in rounds 101–150, with the deterministic π (no noise):
- **drift** = mean of |π₁₅₀(s) − π₀(s)|, over states and the 5 dimensions;
- **sd_estados** = mean, over the 5 dimensions, of the standard deviation of π₁₅₀(s) across states;
- **mean learned action** a_TD3 = mean of π₁₅₀(s) over those states (5-dimensional vector), and its distance to the center [0.5; 0.5; 0.5; 0.5; 0.5].

**Reference:** σ_a = **0.15**, the standard deviation of the exploration noise of the in-house framework's TD3 agent (`exploration_sigma`), already in [0; 1] action units. It is different from the official code's σ_a = 0.0475.

## 3. B2.7b: TD3 vs. the best constant in hindsight

**Constant arms**, on the same 3 cells, H = 150, seeds 42–51. It is the same skeleton as `fixed`, with the full detector, a constant action and no TD3:
- `fixed_b025`: a = [0.5; 0.5; 0.5; 0.5], b = 0.25;
- `fixed_b075`: a = [0.5; 0.5; 0.5; 0.5], b = 0.75;
- `const_td3`: constant action = a_TD3 from the **same** B2.7a run (cell × seed), i.e. the constant that TD3 itself learned in that run.

The center (`fixed`, b = 0.5) comes from B2.7. Total: 3 × 30 = **90 runs**.

**Best constant in hindsight:** per cell, the arm with the highest mean accuracy at H = 150 among {`fixed`, `fixed_b025`, `fixed_b075`, `const_td3`}. It is an optimistic reference for the fixed side, the same asymmetry required of LinUCB and DQN in B2.7.

**Criterion (fixed now, before B2.7a):** **"TD3 learns something beyond a constant"** if Δ = TD3 − best constant (paired by seed, t CI95) is > 0 with the whole CI95 above 0 in **at least 2 of the 3 cells**.

**Also reported:**
- Δ vs. each arm;
- which arm is the best per cell;
- Δ(`const_td3` − `fixed`), which indicates whether the learned constant already explains TD3's gain over the center.

## 4. B2.7c (conditional)

B2.7c (confirmation: 3 cells, H = 150, **seeds 82–91**, pre-registered separately) only runs if **both** conditions hold:
1. **B2.7a:** the median across seeds of sd_estados > σ_a = 0.15 in at least 2 of the 3 cells;
2. **B2.7b:** the §3 criterion is met.

If B2.7c runs, it takes priority over C0b (C0b is paused).

## 5. Reading for P2

- **B2.7b not met:** report "TD3's gain at a long horizon amounts to tuning a constant (the threshold)", with B2.7a as the mechanism.
- **B2.7b met, but sd_estados ≤ σ_a:** TD3 found a better action without depending on the state. Report as an exploratory finding, without confirmation.
- **Both conditions:** run B2.7c before any claim.

## 6. Cost and rules

- **Cost:** B2.7a, 30 runs of ~600 s; B2.7b, 90 runs. In total ~1.5–2 h wall clock with 10 CPU processes (`nice 19`, 2 threads, no GPU).
- **Mandatory order:**
  1. this plan committed;
  2. code with its hash in the addendum;
  3. B2.7a;
  4. B2.7b, which depends on B2.7a's a_TD3.

  The B2.7b criterion does not change after seeing B2.7a.
- No B2.7b accuracy is inspected before the grid ends.
- Logs not versioned.
