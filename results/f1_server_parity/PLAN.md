# Front 1 under FLTrust step parity — plan

**Date:** 2026-11-15. **Status: descriptive check**, frozen before any parity run of the grid or of step 1 is analysed (`PLAN.sha256` covers this file and `scripts/f1_server_parity.py`).

## Why

Front 1 (`results/frente1_ablacao_adaaggrl/`) computed the server's root update with the native fit: the clients' 5 epochs over the 100-sample root, 20 SGD steps. The clients take a median of 752, 917 and 820 steps at α = 0.05, 0.1 and 0.5.

P1 later found that this couples FLTrust to the root size through the server's step budget, and matched the server's step count to the clients' median (`server_steps="median"`, commit `c8cc2e8`, on the P1 branches only). The P2 code base (`src/` on `exp/review`) never received it.

Two Front 1 quantities use that update:
- **`cos_server`**: the step-1 detector (mean AUC 0.704 against S_R's 0.745) and the exploratory `cosserver_only` variant ("practically at the level of `sr_only`", Δ = +0.0014).
- **The `fltrust_aligned` attacker**, which clones it, norm included. On seed 42 its update has 13–20% of the honest norm.

## Parity fit

Copied from P1's `FederatedLearner._fit_server`:
- `steps` = median over clients of ceil(n_train / 32) × 5;
- `steps × 32` indices drawn with replacement from the root, and one pass;
- it starts from the same global parameters as the native fit.

Every parity fit saves and restores the global NumPy RNG, so the native trajectory is not perturbed. Native fits still run wherever Front 1 ran them.

## Runs (seed 42 for step 1, seeds 42–51 for the grid; MNIST, logistic, R = 15, as Front 1)

| variant | content | cells |
|---|---|---|
| `detector` | step 1 along the stored td3 trajectory; adds `cos_server_parity` | 21 |
| `attack` | td3 with record, `fltrust_aligned` cloning the parity update | 3 |
| `grid` | `cosserver_only` with the score's server update fitted with parity; attack reference native | 210 |

## Checks (passed before freezing, on this PC, outputs in the scratchpad)

- **T1:** the detector variant reproduces Front 1's stored step-1 CSV (`trim_attack`, α = 0.05) exactly, max |diff| = 0.
- **T2:** the native `cosserver_only` reproduces Front 1's stored accuracy exactly on 2 cells (one blind attack, one informed). The parity learner replaces the score's server fit exactly once per round (15/15).
- **T3:** parity steps are 752, 917 and 820.
- **In the analysis:** the detector variant must reproduce all 21 stored step-1 CSVs in Front 1's columns.

## Analysis and reading, fixed now

AUCs follow Front 1's procedure: rounds ≥ 2, pre-declared direction, and `cos_server_parity` taking `cos_server`'s direction.

**Q1, the detector (primary).** Mean AUC over 21 cells (and over the 18 without `fltrust_aligned`) of `cos_server_parity` against `cos_server` and against S_R (unchanged; it does not use the root).
- If |Δ mean AUC| < 0.03 and `cos_server`'s position relative to S_R is unchanged, Front 1's step-1 reading is **unaffected**.
- Otherwise it is **affected**, and the change is reported.

**Q2, the attack (descriptive).**
- Norm ratio of Byzantine to honest updates under the parity attack.
- AUC of every detector in the 3 `fltrust_aligned` cells.
- The 21-cell means with those 3 cells replaced.

**Q3, the grid (secondary, descriptive).** `cosserver_parity − sr_only` per seed, mean over cells, with a 90% seed-bootstrap CI. It is read next to Front 1's native +0.0014 (−0.009; +0.012).
- If the CI lies within ±0.01 or includes 0 with |mean| < 0.01, "practically at the level of `sr_only`" **stands**.
- If the CI lies entirely beyond ±0.01, it **changes**, in that direction.
- In any other case it is **inconclusive**.
- The per-attack differences are reported next to the native ones.

Nothing here changes a frozen Front 1 result. The outcome is reported next to it as a check of the implementation.
