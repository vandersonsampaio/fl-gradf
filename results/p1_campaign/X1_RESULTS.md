# Results — X1: E2 and E3 at global factor 1 (exploratory)

**Date:** 2026-11-14. **Status: EXPLORATORY** (`ADDENDUM_2.md`). The analysis ran once on the complete blocks (`scripts/p1/analyze_x1.py`, built on the frozen `scripts/p1/inference.py`). X1 changes no confirmatory result in `RESULTS.md`. It ran on the confirmatory seeds 42–51 to stay paired with them, so it cannot confirm anything.

**Short answer.**
- **The best fixed rule changes with the factor.** At factor 1, d\* is the trimmed mean, not FLTrust, on both the 21-cell E2 grid and the 12-cell E3 grid. By the reading fixed in ADDENDUM_2 Sec. 4, **RQ3 is reported as dependent on the factor, like RQ1.** The text states which system beats which reference at each factor and makes no general claim.
- **RQ2 holds at factor 1 too:** cross-fitted headroom is positive, Ĥ_LOSO = 0.95 p.p. (CI95 0.83 to 1.11) on 21 cells and 1.11 p.p. (0.97 to 1.30) without `fltrust_aligned`. **The headroom is not larger at factor 1:** it is smaller on 21 cells (3.86 at 1/3) and similar on 18 (1.36 at 1/3).
- **One system beats d\* at factor 1: FedStrategist,** by +6.38 p.p. (+5.80; +6.97) against the plain trimmed mean. The attack-aware reference beats it too (+1.77). This is an exploratory result. It does not replace the confirmatory result at factor 1/3, where no system beats FLTrust. The two are reported side by side.
- **The layers matter at factor 1:** hardened FLTrust (41.5%) beats every plain rule (best: trimmed mean, 37.9%). Against that reference, GRADF and Random are worse.

## 0. Execution and checks

- **Size:** 1,500/1,500 jobs from 2026-11-11 22:03 to 2026-11-14 14:34. There were 0 FAIL, 0 INFRA and 0 `SYNC_FAILED`, and no job was repeated.
- **Machines:** this PC 640 runs, vm3 440 runs, vm2 420 runs.
- **Checks passed before any statistic:**
  - every X1 block matches the job list of `x1_job.py` exactly, and so does `e1_sens_factor`, the source of FLTrust and the trimmed mean at factor 1;
  - every X1 result carries the X1 code hash (`19d05b19…`) and the campaign's frozen hash (`d5ed26d6…`);
  - every training run, pretraining included, recorded factor 1.
- **Blindness:** no X1 accuracy was read before the X1 blocks were complete.

Accuracies are the primary metric, the mean test accuracy over rounds 106–110, in p.p. n = 10 seeds.

## 1. E2 at factor 1 — headroom outside the sample (RQ2)

These figures use root 100 and D_R = {FLTrust, median, trimmed mean, Krum}. FLTrust and the trimmed mean come from `e1_sens_factor`, and median and Krum from `x1_fixed`. The CIs come from a seed bootstrap with 20,000 resamples that repeats selection and evaluation.

| factor | cells | in-sample H (optimistic) | Ĥ_LOSO | CI95 | category | d\* | LOSO agreement: d\* |
|---|---|---|---|---|---|---|---|
| 1/3 (confirmatory) | 21 | 3.93 | 3.86 | (3.49; 4.17) | positive | FLTrust | 100% |
| **1** | **21** | 1.01 | **0.95** | (0.83; 1.11) | **positive** | **trimmed mean** | 100% |
| 1/3 (confirmatory) | 18, no FA | 1.45 | 1.36 | (1.03; 1.66) | positive | FLTrust | 100% |
| **1** | **18, no FA** | 1.18 | **1.11** | (0.97; 1.30) | **positive** | **trimmed mean** | 100% |

FA = `fltrust_aligned`. LOSO agreement on the cell winner at factor 1: 99% (21 cells), 98% (18 cells).

- **Mean over the 21 cells:** trimmed mean 42.28, median 39.26, FLTrust 33.91, Krum 23.88.
- **Cell winners:** trimmed mean 15/21, FLTrust 4/21 (krum_collusion, sign_flipping and trim_attack at α = 0.05; trim_attack at α = 0.1), median 2/21.
- **Why the headroom shrinks on 21 cells:** at factor 1/3 most of it came from `fltrust_aligned`, where the best rule (FLTrust) collapses. At factor 1 the best rule is the trimmed mean, which `fltrust_aligned` does not hurt. The remaining headroom comes mostly from the cells where FLTrust or the median beats the trimmed mean.
- **Per-cell means:** `analysis_x1/e2_factor1_cell_means_r100.csv`.

## 2. E3 at factor 1 — how much do adaptive systems capture? (RQ3)

The 12 cells are those of the confirmatory E3. B_−s is cross-fitted and recomputed on them. As in the campaign, GRADF and Random are compared with the hardened rules, and the other systems with the plain rules. Holm is applied within each family.

- **d\* on the 12 cells:** the trimmed mean among the plain rules (37.92) and FLTrust among the hardened ones (41.54).
- **At factor 1/3:** FLTrust under both references.

**Main families (factor 1):**

| family | system | reference | Δ | CI95 | Wilcoxon p (Holm) | category |
|---|---|---|---|---|---|---|
| vs Random | Attack-aware | Random | −0.01 | (−1.46; +1.44) | 1.000 | no evidence (TOST p 0.08) |
| | GRADF | Random | −0.40 | (−1.65; +0.86) | 1.000 | no evidence |
| | **FedStrategist** | Random | **+4.60** | (+3.48; +5.73) | 0.008 | **superiority** |
| | AdaAggRL | Random | −10.96 | (−13.36; −8.55) | 0.008 | inferiority |
| vs B_−s | Random | hardened | −1.84 | (−3.09; −0.59) | 0.039 | inferiority |
| | **Attack-aware** | plain | **+1.77** | (+0.39; +3.15) | 0.039 | **superiority** |
| | **GRADF** | **hardened** | −2.24 | (−3.24; −1.24) | 0.010 | **inferiority** |
| | **FedStrategist** | plain | **+6.38** | (+5.80; +6.97) | 0.010 | **superiority** |
| | AdaAggRL | plain | −9.17 | (−10.99; −7.36) | 0.010 | inferiority |

**Other reference (descriptive):**

| comparison | Δ | CI95 | category |
|---|---|---|---|
| Random vs plain B_−s | +1.78 | (+0.40; +3.16) | superiority (Holm p 0.055 within this family) |
| GRADF vs plain B_−s | +1.38 | (−0.42; +3.18) | no evidence |
| Attack-aware vs hardened B_−s | −1.85 | (−3.85; +0.14) | no evidence |
| Hardened B_−s vs plain B_−s | **+3.62** | (+2.18; +5.06) | **superiority** |

**Mean accuracy over the 12 cells (factor 1):**
- fixed rules, plain: trimmed mean 37.92, FLTrust 35.50, median 35.11, Krum 23.48;
- fixed rules, hardened: FLTrust 41.54, median 34.50, trimmed mean 33.93, Krum 26.00;
- systems: FedStrategist 44.31, Random 39.70, attack-aware 39.69, GRADF 39.31, AdaAggRL 28.75.

**Reading.**
- **The layers carry the layered systems at factor 1.**
  - Hardened FLTrust gains 6.0 p.p. over plain FLTrust, mostly at α ≤ 0.1, where plain FLTrust stays near 24–35%.
  - This fits the earlier ablation finding that Layer 2 acts through the step size.
  - Random, the attack-aware reference and GRADF all pass through the layers and land within 0.4 p.p. of each other.
  - So at factor 1 the selector adds nothing over Random: neither the attack-aware choice nor GRADF's learned choice does better.
  - All three beat or tie the plain d\*, and all three trail hardened FLTrust (Random and GRADF significantly).
- **Whether GRADF beats the best fixed rule depends on the reference.**
  - Against the hardened rules, the confirmatory reference, it is worse: −2.24.
  - Against the plain rules it is not distinguishable: +1.38.
- **FedStrategist:**
  - It beats the plain d\* in 10 of 12 cells, including all 6 cells with cross-fitted headroom.
  - It has the highest mean of any arm, fixed or adaptive, and it is above hardened FLTrust (44.31 vs 41.54).
  - Its largest gains are at α ≤ 0.1, where every plain rule stays below 43%.
  - It does not use the hardening layers.
- **AdaAggRL** stays far below every fixed rule at both factors.

## 3. Side by side: factor 1/3 (confirmatory) and factor 1 (exploratory)

| family | system | reference | factor 1/3: Δ (CI95) | category | factor 1: Δ (CI95) | category |
|---|---|---|---|---|---|---|
| vs Random | Attack-aware | Random | +2.05 (+0.60; +3.50) | superiority | −0.01 (−1.46; +1.44) | no evidence |
| | GRADF | Random | +2.12 (+0.14; +4.10) | superiority | −0.40 (−1.65; +0.86) | no evidence |
| | FedStrategist | Random | +4.12 (+3.22; +5.01) | superiority | +4.60 (+3.48; +5.73) | superiority |
| | AdaAggRL | Random | −10.35 (−12.11; −8.59) | inferiority | −10.96 (−13.36; −8.55) | inferiority |
| vs B_−s | Random | hardened | −4.25 (−6.39; −2.10) | inferiority | −1.84 (−3.09; −0.59) | inferiority |
| | Attack-aware | plain | −2.89 (−4.90; −0.89) | inferiority | +1.77 (+0.39; +3.15) | superiority |
| | GRADF | hardened | −2.12 (−4.38; +0.14) | no evidence | −2.24 (−3.24; −1.24) | inferiority |
| | FedStrategist | plain | −0.83 (−2.04; +0.39) | no evidence | +6.38 (+5.80; +6.97) | superiority |
| | AdaAggRL | plain | −15.30 (−16.91; −13.68) | inferiority | −9.17 (−10.99; −7.36) | inferiority |

| | factor 1/3 | factor 1 |
|---|---|---|
| d\* on 21 cells (E2) | FLTrust | trimmed mean |
| Ĥ_LOSO, 21 cells | 3.86 (3.49; 4.17) | 0.95 (0.83; 1.11) |
| Ĥ_LOSO, 18 cells | 1.36 (1.03; 1.66) | 1.11 (0.97; 1.30) |
| d\* on the E3 grid, plain / hardened | FLTrust / FLTrust | trimmed mean / FLTrust |
| E3 cells with cross-fitted headroom, plain / hardened | 3 / 0 | 6 / 4 |
| Hardened B_−s − plain B_−s | −0.70 (no evidence) | +3.62 (superiority) |

Descriptive only. No test was run across factors.

**Reading fixed before running (ADDENDUM_2 Sec. 4), applied:**
- **d\* changed** (FLTrust → trimmed mean among the plain rules). RQ3 is therefore reported as **dependent on the factor**, and the paper states per factor which system beats which reference.
  - At factor 1/3 no system beats d\*. GRADF and FedStrategist are not distinguishable from it.
  - At factor 1, FedStrategist and the attack-aware reference beat the plain d\*. GRADF and Random are worse than the hardened d\*.
- **A system beats d\* at factor 1 (FedStrategist).** This is exploratory and is reported next to the confirmatory factor-1/3 result, not instead of it.
- **The headroom is not larger at factor 1** (smaller on 21 cells, similar on 18). The finding is that the best fixed rule depends on the global-rate setting, not that the opportunity for adaptation grows.
- **Required in the paper:** this side-by-side table; d\* and Ĥ_LOSO per factor; and a sentence in the abstract restricting the RQ3 conclusion to the factor.

## 4. Layer rates (exploratory)

Rate per round at factor 1, for the arms that pass through the layers:

| arm | Layer-4 rejection | Layer-1 rejection (honest or attacked client) |
|---|---|---|
| Random | 60% | 9% |
| GRADF | 60% | 4% |
| Attack-aware | 52% | 17% |
| Hardened FLTrust | 59% | 1% |
| Hardened trimmed mean | 42% | 22% |
| Hardened median | 44% | 25% |
| Hardened Krum | 69% | 9% |

Layer 4 rejects about twice as often as at factor 1/3, where the rates were 18–51%. This fits the larger, less stable steps at factor 1.

## 5. Limits

- **X1 is exploratory,** and its seeds are the confirmatory ones, so it cannot confirm anything.
- **R = 110 at factor 1:** FLTrust is unstable at this factor (E1 sensitivity). The primary metric averages 5 rounds, but single runs may still oscillate.
- **GRADF's DQN is not deterministic,** so each run is one draw.
- **The hardened reference changes the answer for the layered systems,** as Sec. 2 shows. The table keeps the like-for-like rule of the campaign: hardened for GRADF and Random, plain for the rest.
- **The aggregates average over cells whose accuracy levels differ widely,** from about 10% to 61%. Per-cell values are in the CSVs.

## Outputs (`analysis_x1/`)

- `analysis_x1.json`: every number above, including the checks and the H_s values per seed.
- `analysis_x1.txt`: the console output of the single run.
- `e3_factor1.csv` and `e3_factor1_descriptive.csv`: the E3 tables at factor 1.
- `e3_side_by_side.csv`: E3 at both factors.
- `e2_factor1_cell_means_r100.csv` and `e3_factor1_cell_means.csv`: per-cell means.
- `x1_runs.csv`: one row per run, with block, arm, cell, seed, primary metric, machine, and the Layer-1 and Layer-4 rates.
