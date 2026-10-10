> English translation of `PREREGISTRO.md`. The Portuguese original is the frozen record (its hash is in `PREREGISTRO.sha256`); if the two ever disagree, the original prevails.

# Pre-registration: Step 2, does TD3 contribute in the official AdaAggRL?

**Status:** FINAL. The grid was fixed on 2026-09-25, after the cost measurement (seed 0, discarded) and before any confirmatory run.
**Date:** 2026-09-25
**Record:** local and uncommitted at the time, by the author's decision. The file must not be edited after the grid starts; any later change goes in as a dated addendum at the end.

## 1. Question

In the published AdaAggRL code (`github.com/yjEugenia/AdaAggRL`, commit `27b9c18`), at the published horizon (500 rounds), does the TD3 policy produce better accuracy than a fixed action and than a random action?

## 2. Code and adjustments

- Official code untouched in `external/AdaAggRL`, isolated venv `external/.venv_adaaggrl` with the versions from the official `requirements.txt` (torch 2.3.0, SB3 2.3.2, numpy 1.26.4, gym 0.26.2), plus the gymnasium 0.29.1 that SB3 requires.
- Runner: `scripts/passo2_oficial/run_oficial.py`. The minimal adjustments are listed in its docstring: `inversefed` shim, Gymnasium adapter, seeds, no-op SummaryWriter and `--dataset MNIST`. No line of the official logic (environment, attacks, reward, TD3) is changed.
- The official reward uses the test set. It is kept, because the question is about the published method. The leak can only favor TD3.

## 3. Conditions

| condition | action |
|---|---|
| `td3` | SB3 TD3 exactly as in `main.py` (MlpPolicy [256,128], lr 1e-5, buffer 1000, batch 64, `train_freq` 3, noise N(0; 0.1), γ 0.99, `learning_starts` 100) |
| `fixed` | [0.475]×5, the center of the Box [0; 0.95]^5: a constant a[:4] gives a uniform softmax, and a₅ = 0.475 is the mean of TD3's own actions in the random phase |
| `random` | uniform in [0; 0.95]^5 for the 500 rounds |

## 4. Grid

- MNIST dataset, 500 rounds, 100 clients, 10% per round, 20 attackers (official defaults).
- **q = 0.5** (moderate non-IID, one of the paper's three values). The 42–51 ablation regime is non-IID, and the `main.py` default q = 0.1 is IID.
- **Attacks:** LMP and EB. The official IPM is **left out**: it is a null update, and it would cost about 22 h more.
- **Seeds:** 100, 101, 102, 103, 104 (new, never used in the project).
- **Total:** 2 attacks × 3 conditions × 5 seeds = **30 runs**.
- **Measured cost** (seed 0, EB/LMP/IPM, runs discarded): about 27 s per round with 1 process and about 64 s per round per process with 6 processes in parallel (3 threads each, RTX 3050 GPU at 98%). That is about 1.5 machine-hours per run and **about 44 h** for the grid.
- **Execution order:** all combinations for seeds 100–102 first, then 103–104. This is only scheduling: all 5 seeds are committed and **no confirmatory analysis is done before the grid is complete**.
- **Parallelism:** 6 processes, `OMP_NUM_THREADS=3`. Does not affect the logic.

## 5. Metrics

- **Primary:** mean test-set accuracy over the last 50 rounds (451–500), from the official `history['acc']`.
- **Secondary:** accuracy at round 500; full curve; weight mass assigned to real attackers per round; number of resets (reward < −80); firings of the `sim_lc ≥ 0.9` rule; trajectory of the TD3 actions.

## 6. Hypotheses and criteria

Let D = metric(`fixed`) − metric(`td3`), per (attack, seed) pair.

**H1 (central thesis).** TD3 does not beat the fixed action.
- Primary test: paired TOST (t) over the (attack, seed) pairs of the main attacks, with margin **±1.0 p.p.** and α = 0.05.
  - TOST rejects both null hypotheses → **equivalence**; the central thesis holds for the published method.
  - Paired Wilcoxon with p < 0.05 and D < 0 → **TD3 contributes** at the long horizon; the paper changes to "RL only pays off after N rounds".
  - Wilcoxon with p < 0.05 and D > 0 → the fixed action is better. Report as "does not contribute"; do **not** claim that TD3 hurts without replication.
  - None of the above → **inconclusive**; report Δ, 95% CI and d, without declaring equivalence.
- Per attack: Δ, 95% CI, d and Wilcoxon with Holm over the attacks (descriptive, given the small n).

**H2 (secondary).** `random` − `td3`, with the same procedure. It tells whether the learned policy is distinguishable from no policy at all.

**Exploratory, non-confirmatory analysis:**
- the curve of D over the rounds (TD3 only acts by policy after round 100);
- the trajectory of the TD3 actions and the **mean distance to the Box center (0.475)**. In the smoke test (seed 0, `learning_starts=2`, discarded), the untrained policy already emits actions near 0.47, because the tanh at initialization is ≈ 0. The `fixed` action therefore coincides with TD3 "before learning", and the question becomes whether the policy moves away from its starting point with about 66 updates at lr 1e-5;
- the weight mass on real attackers, per condition.

## 7. Rules

- No parameter of the official code is tuned after seeing grid results.
- Runs that fail due to an environment error are repeated with the same seed and recorded. Runs with NaN or collapse count as results.
- Resets are counted, not discarded.

---

## Addendum 1 (2026-09-25, 21:35): truncation at 500 rounds

**Reason:** the first completed `td3` run (EB, seed 100) has **501 steps**. SB3 2.3.2 collects in blocks of `train_freq=3`, and `learn(total_timesteps=500)` only stops at the end of a block (498 → 501). The official `main.py` behaves the same way. The `fixed` and `random` conditions have exactly 500 steps.
**Change:** in the analysis, all runs are truncated to the **first 500 steps** before computing any metric. For all conditions, the primary metric stays on the window of rounds 451–500, as pre-registered. The analysis records `n_steps_raw` for transparency.
**Information seen before the addendum:** only the step counts and run times. **No accuracy or comparison between conditions was looked at.**
**File changed:** `scripts/passo2_oficial/analisar.py` (function `load`).
