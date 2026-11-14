# Addendum 3 — X5: the action set against the reference set (descriptive)

**Date:** 2026-11-14, after the X1 analysis (`X1_RESULTS.md`). **Decision:** the author's (`docs/p1_x5_conjunto_de_acoes.md`, in `fl-fedmed`), with three adjustments agreed before this freeze (Sec. 6).

**Status: DESCRIPTIVE.** X5 changes no confirmatory result (`RESULTS.md`) and no exploratory result (`X1_RESULTS.md`). It runs on seeds 42–51 to stay paired with them, so it cannot confirm anything. D_R = {FLTrust, median, trimmed mean, Krum} stays the preregistered reference set; the 7-rule reference enters as a sensitivity analysis.

**Freeze.** This addendum is frozen with `ADDENDUM_3.sha256`, and the X5 code with `X5_CODE.sha256`, before the first X5 job on seeds 42–51 and before any X5 output is computed from existing runs.

## 1. Why

In X1, FedStrategist beat the plain d\* by +6.38 p.p. (+5.80; +6.97) at factor 1. An explanation that needs no selection remains open:
- d\* is chosen among the 4 rules of D_R, while FedStrategist (and Random) choose among 7 rules. The 7-rule arsenal adds FedAvg, FedProx and clustering.
- Those three rules never ran as fixed rules at factor 1. `e1_sens_factor` has only FLTrust and the trimmed mean, and `x1_fixed` only the median and Krum.
- In the E0 pilot, which ran at factor 1 but with R = 50 and seeds 0–2, FedAvg was the best rule under `sign_flipping` (31.9%) and clustering under `trim_attack` (32.3%) at α = 0.05, above every rule of D_R. This is a motivation, not evidence.
- FedStrategist's gains concentrate at α ≤ 0.1.

If a fixed rule outside D_R matches FedStrategist, its gain comes from the size of the action space, not from selection.

## 2. Runs

| block | content | jobs |
|---|---|---|
| `x5_fixed` (X5a) | FedAvg, FedProx and clustering as fixed rules at factor 1 and root 100, on the 12 E3 cells | 360 |
| `x5_replay` (X5b) | every FedStrategist run of `e3_main` (factor 1/3) and of `x1_systems` (factor 1), run again with the rule chosen in each round recorded | 240 |
| | **Total** | **600** |

**Setup.** As in X1: R = 110, 2 local epochs, `local_lr` 0.01, `server_steps = median`, τ_CNN = 7.86, root 100, 10 clients of which 2 are Byzantine, seeds 42–51, and the primary metric (mean test accuracy over rounds 106–110).

**Not run** (declared as limits, Sec. 5):
- the optional 270 jobs for the 9 remaining E2 cells;
- hardened versions of the three rules.

**Code.**
- `scripts/p1/x5_job.py` calls `campaign_job.run_job` unchanged. The campaign's frozen hash (`d5ed26d6…`) and `x1_job.py` still hold.
- The X5 hash covers `src/`, `campaign_job.py`, `x1_job.py` and `x5_job.py`.
- On seeds 42–51 a job refuses:
  - a job outside this addendum's lists;
  - R ≠ 110;
  - an output or replay source other than `results/p1_campaign`;
  - a code hash other than `X5_CODE.sha256`;
  - the numerical environment of PREREGISTRATION Sec. 3.
- `run_job`'s guard aborts any run whose factor or server steps differ from its block's.
- **The replay** records the bandit's choice by wrapping `LinUCBAgent.select_action` from outside `src/`. It writes the choices to a new file under `x5_replay/` and never touches the stored run. It also records whether the final parameters' hash and the accuracy curve equal the stored ones.

**Dry run** (`checks/x5_dry/`), seed 0, R = 2, on this PC:
- each of the three rules finished with factor 1 recorded;
- the replay of X1's dry-run FedStrategist job reproduced its parameter hash and accuracy curve exactly.

**Execution.** `scripts/p1/dispatch.py`, on this PC (8 processes) and the 2 VMs (4 each), with free assignment and the one repeat on lost connection of ADDENDUM_1. Job order: `x5_fixed`, then `x5_replay` for `x1_systems`, then for `e3_main`, each grouped by seed.

## 3. Analysis (fixed now, run once)

The unit is the seed. The 7-rule set D_7 is D_R plus FedAvg, FedProx and clustering, all at root 100 and on the 12 E3 cells.
- **Factor 1/3:** all 7 rules from `e1_main`.
- **Factor 1:** FLTrust and the trimmed mean from `e1_sens_factor`, the median and Krum from `x1_fixed`, and the other three from `x5_fixed`.

**X5c, existing runs** (factor 1/3, and factor 1 once X5a is complete): the best of the 7 rules against the best of D_R.
- d\*₇ and d\*_R, in sample and as LOSO selections.
- Cells where a rule outside D_R is the best of the 7.
- Ĥ_LOSO over D_7 next to Ĥ_LOSO over D_R, on the 12 cells, with the 20,000-resample seed bootstrap.

**Main comparison** (per factor): A_FedStrategist − B7_−s.
- B7_−s is the cross-fitted global best of the 7 rules, built exactly like E3's B_−s but over D_7.
- **Reported:**
  - `inference.describe` on the per-seed means: CI95, Wilcoxon, t-test, d_z, TOST at 1.0 p.p., and category;
  - a seed-bootstrap CI95 (20,000 resamples) that repeats the selection of d\*₇ in each resample, leaving every copy of the held-out seed out of the selection, as in `inference.loso_headroom`.
- **Descriptive, alongside:**
  - per cell, FedStrategist against the best of the 7 rules (O7_−s);
  - A_FedStrategist − B_R,−s (the X1 comparison) next to it.
- **No Holm and no confirmatory test.**

**X5b, action frequencies.**
- **FedStrategist:** from the replays, the share of rounds each of the 7 rules is chosen, per cell and factor.
- **Random, attack-aware and GRADF:** from the stored Layer-3 logs.
- Cells where FedStrategist beats the plain d\*_R are flagged.
- In those cells: A_FedStrategist minus the fixed rule it chooses most in that cell.
- **Replay acceptance:** a replay counts only if its parameter hash equals the stored run's. The number of non-identical replays is reported. Their choices are excluded from the frequencies, and if any exist, the frequencies are labelled as covering the identical replays only.

## 4. Reading fixed before running

Primary: the category of `describe` on A_FedStrategist − B7_−s, read with the bootstrap CI. If the two disagree on whether 0 is inside, the result is **inconclusive**.

| result at factor 1 | reading |
|---|---|
| CI95 entirely above 0 | **Exploratory evidence of useful selection** at factor 1, beyond the action set. Reported next to the confirmatory factor-1/3 result, where FedStrategist does not beat d\*. |
| equivalence (TOST at 1.0 p.p.), or CI95 entirely below 0 | FedStrategist's gain in X1 is attributed to the **action set**, not to selection. The paper says so explicitly. |
| no evidence (CI95 includes 0, not equivalent) | **Inconclusive**, stated in those words. The X1 gain is not attributed to selection. |

| result at factor 1/3 (X5c) | reading |
|---|---|
| some rule outside D_R beats D_R on the E3 cells (d\*₇ ∉ D_R, or Ĥ_LOSO over D_7 above Ĥ_LOSO over D_R with non-overlapping CIs) | The same issue affects the confirmatory result. The paper states that headroom and d\* were defined over D_R and shows d\*₇ next to it. |
| no rule outside D_R beats D_R | The confirmatory result is not affected. |

In every case the confirmatory result stands, and D_R stays the preregistered set.

## 5. Limits stated in advance

- X5 is descriptive, and its seeds are the confirmatory ones, so it cannot confirm anything.
- **The three added rules run plain only.**
  - This is enough for FedStrategist, which does not use the layers.
  - It does not test whether GRADF or Random, which pass through the layers and choose among 7 rules, gain from the larger set against a hardened 7-rule reference.
- **E2 is not extended** to the 7 rules on all 21 cells.
- **A replay reproduces one stored run;** it does not add a draw. FedStrategist was deterministic in the dry run.
- **Monitoring:** reads counts, errors and process state only. No X5 accuracy is read before the X5 blocks are complete, and no X5c output is computed before this freeze.

## 6. Adjustments to the author's plan, agreed before freezing

1. **X5b needs new jobs for FedStrategist.** Its stored runs do not record the rule chosen per round, so its choices come from the 240 replays (Sec. 2). Random, attack-aware and GRADF have their choices in the stored Layer-3 logs.
2. **"CI95 includes 0" is no evidence, not a tie.** Sec. 4 separates equivalence, attributed to the action set, from no evidence, which is inconclusive. It uses the campaign's TOST margin.
3. **The 7-rule cross-fitting and the bootstrap that repeats the selection** are new code in the X5 analysis script. The frozen `inference.py` is reused but not modified.
