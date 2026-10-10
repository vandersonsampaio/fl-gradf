> English translation of `PLANO.md`. The Portuguese original is the frozen record (its hash is in `PLANO.sha256`); if the two ever disagree, the original prevails.

# Plan — B2.7: does the gain from learning grow with the horizon? (15 / 50 / 150 rounds)

**Status:** FINAL before the grid (hash in `PLANO.sha256`). **Exploratory.** Changes after that only as a dated, hashed addendum.
**Date:** 2026-10-01
**Scope:** B2.7 (priority 1): a gap in the literature (0 of 78 reviewed studies vary the horizon); separate robustness from convergence in the C0 gaps.
**Code:** `scripts/b27_horizonte.py` (reuses, without changing them, `scripts/frente1_ablacao_adaaggrl.py`, `src/experiments/exp10_selector_comparison.py`, `src/experiments/exp1_baseline.py` and `src/fl/gradf_learner.py`).

## 1. Question

Does each agent's gain over the **fixed version in the same action space** (same signal, same granularity, same memory) grow with the horizon?

## 2. Agents and fixed versions

| agent | fixed version | extra reference |
|---|---|---|
| **TD3** (`td3_ref`: AdaAggRL, the ablation learner, `td3` mode) | `fixed`: same skeleton, action at the center (not B2.6's best variant, to isolate learning only) | — |
| **LinUCB** (`linucb`: FedStrategist, variant **(b)**, GRADF's shared detection) | **best fixed arm per cell** among the 7 `arm_rule_<r>` | `rand_rule`: uniform rule per round |
| **DQN** (`dqn`: GRADF v1, pretrained with `train_gradf_models`) | **best fixed arm per cell** among the 7 `arm_gradf_<r>` | `rand_gradf`: `RandomSelector`, as in exp10 |

- **Arsenal:** exp10's 7 rules (`fedavg`, `fedprox`, `median`, `trimmed_mean`, `fltrust`, `clustering`, `krum`).
- **Same action space, per agent.** LinUCB and DQN apply the chosen rule through different paths, and each fixed arm follows its agent's path:
  - **LinUCB:** the arms apply the rule exactly as `FedStrategistGridLearner` does (same updates via `compute_param_updates_auto`, `server_update` on the root every round, `_make_arsenal_strategy(rule, n_byz)`), only without detection and without the bandit.
  - **DQN:** the rule goes through the GRADF pipeline (hardening, layers 1–5), so the arms are `GRADFFederatedLearner` itself with `FixedActionSelector(rule)`.
  - Hence there are **14 arms** (7 + 7), not 7. **Declared deviation** from the approved design: without it, the comparison would mix learning with the effect of the pipeline.
- **The best arm is chosen in hindsight**, on the same seeds, per cell and per horizon (highest mean). It is an **optimistic reference** for the fixed side:
  - if the agent still beats it, the result is strong;
  - if it ties, the reading carries that caveat.

## 3. Horizons

- **H ∈ {15, 50, 150}**, **nested**: each system runs 150 rounds, and the accuracy at H is that of the global model at the end of round H.
- Nothing in the learners depends on `n_rounds` beyond the training loop, so the result is identical to running H rounds. This is **checked** (§6, item 1).
- All runs are **new**, with the same code version; nothing is reused from the earlier runs.

## 4. Cells, seeds and regime

**8 cells:**

| cell | profile |
|---|---|
| `label_flipping` α 0.05 and 0.1 | headroom in C0; S_R does not separate |
| `sign_flipping` α 0.05 and 0.1 | per-rule oracle ≥ skeleton |
| `gaussian_noise` α 0.05 and `krum_collusion` α 0.1 | robust granularity |
| `low_mag_backdoor` α 0.05 | cos_server wins |
| `trim_attack` α 0.5 | control at the ceiling |

- **Seeds:** 42–51 (exploratory; consistency with C0).
- **Regime:** identical to C0/exp10 (logistic MNIST, 10 clients, Byzantine [0, 1], root 100).
- **Ceilings** per α × seed × H, by α only (not by attack):
  - FedAvg-10 without attack, as in C0;
  - FedAvg-8 oracle (the 8 honest clients only, evaluated on the 10 test sets, as in `c0_teto_oraculo`).
  - Covers the 3 α of the cells.

## 5. Criterion (exploratory, per agent)

- **Metric:** final accuracy at horizon H (mean over the 10 clients' test sets, as in C0).
- **Unit:** seed, n = 10.
- **Δ = agent − fixed version**, paired by seed; t CI95.
- **"Learning pays off at horizon H"** if Δ > 0 with the whole CI95 above 0 in **at least 3 of the 8 cells**. Evaluated for each agent (3) and each H (3).
- **No multiplicity correction** (3 × 3 evaluations), acceptable in an exploratory study; declared.
- **Also reported:**
  - the **trend** of Δ across H (mean Δ over the cells and slope per log H, per cell);
  - Δ vs. the random arm;
  - gaps to both ceilings per H;
  - which arm is the best per cell and H.
- **Reading for C0:** if the C0 gaps are about convergence, they shrink with H in all systems (including the fixed ones). If they are about robustness, they persist.

## 6. Checks (descriptive)

1. **Nesting:** on the `label_flipping` α 0.05 cell, seed 42, running with `n_rounds` = 15 and 50 must exactly reproduce (|Δ| < 1e-9) the 150-round run at H = 15 and 50, in 8 systems. If it does not, the horizons become separate runs (addendum).
2. **Reproduction at H = 15** against the earlier runs: `td3_ref` (exp10 AdaAggRL), `fixed` (ablation), `linucb` (exp10 FedStrategist b), `dqn` (exp10 GRADF) and `rand_gradf` (exp10 Random). The max. |Δ| is reported. The `arm_rule_*` are compared with exp9, which aggregates through another path, so only closeness is expected. Differences do not invalidate the grid (all systems use the same version), but they are reported.

3. **Global RNG (smoke-test finding, fixed before the freeze):** the framework learners' `seed` does **not** seed the global `np.random`, which controls the local-training permutation and the DP noise. In a process with several systems, each one inherited the state left by the previous one. Only `td3_ref`/`fixed` escaped, because the extractor calls `keras.utils.set_random_seed`. Therefore:
   - B2.7 calls `keras.utils.set_random_seed(seed)` before the pretraining and before **each** system and ceiling;
   - test with two processes in opposite orders: `linucb`, `rand_gradf` and `arm_rule_median` are **identical**;
   - `dqn` still varies (0.86 p.p. in the test) with its position, due to TF internal state. In the grid, the order of the systems is **fixed and equal** in all cells and seeds, so the results are deterministic and paired.
   - **Consequence for the earlier runs** (exp9/exp10/C0): the numbers depend on the loop order. That is why the non-TD3 B2.7 systems do not exactly reproduce exp10 at H = 15. In the smoke test (cell `label_flipping` α 0.05, seed 42, before the fix): `td3_ref`, `fixed`, `arm_rule_median` and `arm_rule_krum` with |Δ| = 0; `linucb` 1.1 p.p.; `dqn` 1.6 p.p.; `rand_gradf` 5.6 p.p.; ceilings 0.1–0.6 p.p.

## 7. Cost and execution

- **Per cell × seed:** 20 systems × 150 rounds + GRADF pretraining. In the smoke test (15 rounds, CPU shared with B2.5): ~8 s/round for the 2 systems with inversion, ~1.1 s/round for the other 18 and 208 s of pretraining.
- **Estimate:** ~5,700 process-seconds per cell × seed, ~125 process-hours in total. With 10 processes (20 cores, `nice 19`, 2 threads, the GPU busy with B2.5), **~25–35 h wall clock**. The ceilings (30 jobs) take < 1 h in total.
- **Calibration:** with the first batch. If it exceeds 40 h, the declared cut is to run H = 150 only on seeds 42–46 (addendum before seeing results).
- **Launcher:** `scripts/run_grid_b27.sh` (resumable, one job per cell × seed and per α × seed for the ceilings).

## 8. Rules

- Pre-written analysis: `python -m scripts.b27_horizonte analisar` (and `verificar`), run with the complete grid.
- No accuracy is inspected before the grid ends. Monitoring reports only health and progress.
- Jobs that fail due to an environment error are repeated with the same seed and recorded.
- `scripts/frente1_ablacao_adaaggrl.py` is a dependency and **is not versioned** [at the time of the freeze; it has since been committed]. Its hash goes into `PLANO.sha256`.
