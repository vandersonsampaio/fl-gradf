> English translation of `PLANO.md`. The Portuguese original is the frozen record (its hash is in `PLANO.sha256`); if the two ever disagree, the original prevails.

# Plan — D2: run-to-run variance of the P1 numbers

**Status:** FINAL before any run (hash in `PLANO.sha256`). Descriptive, with a reading rule fixed now.
**Date:** 2026-10-04
**Motivation:** B2.7 showed that GRADF v1's DQN is non-deterministic across executions (~4 p.p. in one cell) and that the framework's global RNG depends on the execution order. The P1 numbers (Tables 3–5 of the P1 paper) therefore have an unreported run-to-run variance.

## 1. Question

Re-running **the same code** as P1 (tag `p1.0.0`), with the **same seeds** and in the **same order**, how much do the numbers change? Does P1's conclusion survive this variance? (GRADF and FedStrategist below Random; AdaAggRL above; Table 4.)

## 2. Design

- **Code:** a `git worktree` of tag `p1.0.0`, without any change. `data/processed` (not versioned) comes in through a symbolic link.
- **Call:** `src.experiments.exp10_selector_comparison.run_selector_comparison_grid(seed=s, variant="b", root_size=100)`, with P1's 5 systems (random, oracle, gradf, fedstrategist, adaaggrl), on the 21 cells and in the code's own order, as in P1.
- **Seeds:** 42, 43 and 44 (3 of P1's 10).
- **Repetitions:** 3 per seed, each in a new process (9 processes, in parallel on the CPU). The original P1 execution (`results/tables/exp10_FULL_variantb_raw.csv` at the tag) counts as the 4th execution.
- **Cost:** ~1.5 h per process (21 cells × GRADF pretraining + 5 systems); ~2–3 h wall clock.
- **"After D1":** D1 (DQN determinism and isolated RNG) has not been implemented yet. The "after" round is left for when it exists, as an addendum. Here only the **before** is measured.

## 3. Measures

1. **Per system × cell × seed:** standard deviation of the accuracy across the 4 executions (SD_exec) and the max − min range.
2. **Table 4's aggregate Δ,** per execution and seed: mean over the 21 cells of (system − Random), for GRADF, FedStrategist, AdaAggRL and Oracle. Then:
   - **SD_exec** of the aggregate Δ, across the 4 executions of each seed, averaged over the 3 seeds;
   - **SD_sementes** of the aggregate Δ across the original P1's 10 seeds, as a reference.

## 4. Reading rule (fixed now)

- **"P1 conclusion robust to run-to-run variance"** if, in the 12 execution × seed combinations (4 × 3):
  - the aggregate Δ of **GRADF − Random** and of **FedStrategist − Random** is **< 0** in all of them;
  - the aggregate Δ of **AdaAggRL − Random** is **> 0** in all of them;
  - and the SD_exec of the aggregate Δ is **< ½ |published Δ|** for all three: GRADF −0.042, FedStrategist −0.044, AdaAggRL +0.030.
- **Otherwise:** "run-to-run variance is of the order of the effect", and P1's limitation note reports the magnitudes, per system.
- **Reported without a criterion:** the cells with the largest SD_exec, per system; which systems are deterministic (SD_exec = 0); the ratio SD_exec / SD_sementes.

## 5. Rules

- No number is inspected before the end. Pre-written analysis (`scripts/d2_variancia_p1.py analisar`).
- The result only feeds P1's limitation note. No P1 number is replaced.
