# Addendum 2 — X1: E2 and E3 at global factor 1 (exploratory)

**Date:** 2026-11-11, after the confirmatory analysis (`RESULTS.md`). **Decision:** the author's (`docs/p1_experimentos_finais.md`, X1, in `fl-fedmed`). **Status: EXPLORATORY.** X1 changes no confirmatory result of the campaign. It runs on the confirmatory seeds 42–51 to stay paired with them, which is allowed because those conclusions are closed and X1 makes no confirmatory claim. This addendum is frozen with `ADDENDUM_2.sha256`, and the X1 code with `X1_CODE.sha256`, before the first X1 job on seeds 42–51.

## 1. Why

E2 and E3 ran at factor 1/3 only, and the factor was tuned in FLTrust's favour (PREREGISTRATION Sec. 10). "No system beats the best fixed rule" is therefore stated against a reference chosen to be strong. X1 checks whether RQ2 and RQ3 hold at factor 1, where FLTrust trails the trimmed mean (E1 sensitivity: V = −8.38 p.p. at root 100).

## 2. Setup

The campaign's setup applies, with only the global factor changed to 1:
- R = 110, 2 local epochs, `local_lr` 0.01;
- `server_steps = median`, τ_CNN = 7.86, root 100;
- 10 clients, 2 of them Byzantine;
- seeds 42–51 and the primary metric (mean test accuracy over rounds 106–110).

Systems pretrain at factor 1, as in the campaign, where pretraining receives the block's factor.

| block | arms | cells | runs |
|---|---|---|---|
| `x1_fixed` (X1a) | median and Krum at root 100 | 21 (3 α × 7 attacks) | 420 |
| `x1_systems` (X1b) | Random, attack-aware, GRADF, FedStrategist (variant a), AdaAggRL (random extractor) | 12, the E3 grid | 600 |
| `x1_hardened` (X1b) | FLTrust, median, trimmed mean and Krum through Layers 1, 2 and 4 | 12 | 480 |
| | **Total** | | **1,500** |

**Runs reused from `e1_sens_factor`.** FLTrust and the trimmed mean at root 100 and factor 1 are already there, on all 21 cells and seeds 42–51 (210 runs each). With `x1_fixed` they complete D_R = {FLTrust, median, trimmed mean, Krum} at factor 1.

**Code.**
- `scripts/p1/x1_job.py` adds the three blocks and calls `campaign_job.run_job` unchanged. The campaign's frozen hash still holds (`d5ed26d6…`).
- The X1 hash covers `src/`, `campaign_job.py` and `x1_job.py`.
- On seeds 42–51 a job refuses:
  - a block, arm or cell outside this addendum;
  - R ≠ 110;
  - an output other than `results/p1_campaign`;
  - a code hash other than `X1_CODE.sha256`;
  - the numerical environment of PREREGISTRATION Sec. 3.
- `run_job`'s guard aborts any run, pretraining included, whose factor or server steps differ from the block's.

**Execution.** `scripts/p1/dispatch.py` runs X1 on this PC (8 processes) and on the 2 VMs (4 each), with free assignment as in the campaign, and with the one repeat on lost connection of ADDENDUM_1. The bit-for-bit check between machines (`checks/SUMMARY.md`) was run at factor 1/3 and is not repeated; the factor does not change the numerical environment. Job order: `x1_fixed`, then `x1_systems`, then `x1_hardened`, grouped by seed.

**Before the first job on seeds 42–51:** a dry run on seed 0 with R = 2 runs one job of each arm kind (`checks/x1_dry/`). Each must finish with the factor recorded as 1 in every training run.

## 3. Analysis (fixed now, run once on the complete blocks)

Same procedure and code as the confirmatory analysis (`scripts/p1/inference.py`): the seed is the unit, with paired Wilcoxon and t-test, CI95, d_z, TOST at 1.0 p.p. and the four categories. Holm is applied within each family.

- **E2 at factor 1:**
  - Ĥ_LOSO with a 20,000-resample seed bootstrap that repeats selection and evaluation;
  - in-sample H, labelled as an optimistic upper bound;
  - d\* and LOSO agreement;
  - on 21 cells and on the 18 without `fltrust_aligned`.
- **E3 at factor 1, on the 12 cells:** d\*, B and O recomputed on them, with B_−s cross-fitted. Two families, each with Holm:
  - A_sys − A_Random, for GRADF, attack-aware, FedStrategist and AdaAggRL;
  - A_sys − B_−s, for the same systems and Random.

  GRADF and Random are compared with the hardened rules, the other systems with the plain rules, as in the campaign. The other reference is reported as descriptive. So is the attack-aware reference against the hardened rules, as in `RESULTS.md`.
- **Side by side:** E3 at factors 1/3 and 1, and d\* and Ĥ_LOSO per factor. Descriptive only, with no test across factors.
- **Exploratory:** cells where each system beats B_−s, capture in headroom cells, and the Layer-1 and Layer-4 rates.

## 4. Reading fixed before running (from the author's X1)

- **If d\* stays FLTrust at factor 1:** the RQ3 conclusions no longer depend on the factor.
- **If d\* changes** (for example to the trimmed mean): RQ3 is reported as dependent on the factor, like RQ1. The text states which system beats which reference at each factor, with no general claim.
- **If a system beats d\* at factor 1:** this is an exploratory result. It does not replace the confirmatory result at factor 1/3, and the two are reported side by side.
- **If the headroom at factor 1 is larger:** the opportunity for adaptation is reported as depending on the global-rate setting, which supports the thesis of measuring before claiming.

**Required in the paper:** the E3 table with both factors side by side, d\* and Ĥ_LOSO per factor, and a sentence in the abstract restricting RQ3 to the factor if the result depends on it.

## 5. Limits stated in advance

- X1 is exploratory, and its seeds are the confirmatory ones, so it cannot confirm anything.
- **R = 110 at factor 1:** FLTrust is unstable at this factor (E1 sensitivity). The primary metric averages 5 rounds, but single runs may still oscillate.
- GRADF's DQN is not deterministic, so each run is one draw.
- **Monitoring:** reads counts, errors and process state only. No X1 accuracy is read before each block is complete.
