# Results — X5: the action set against the reference set (descriptive)

**Date:** 2026-11-15. **Status: DESCRIPTIVE** (`ADDENDUM_3.md`). The analysis ran once on the complete blocks (`scripts/p1/analyze_x5.py --part full`, built on the frozen `scripts/p1/inference.py`). The part that uses only pre-existing runs (`--part existing`) ran once before, right after the freeze. X5 changes no confirmatory result (`RESULTS.md`) and no exploratory result (`X1_RESULTS.md`). It ran on seeds 42–51 to stay paired with them, so it cannot confirm anything.

**Short answer.**
- **At factor 1, FedStrategist's gain over d\* comes from the action set, not from selection.**
  - Against the cross-fitted best of the 7 rules (B7_−s), the difference is −0.22 p.p. (CI95 −0.90 to +0.45; bootstrap with repeated selection −0.75 to +0.35).
  - It is **equivalent** within the 1.0 p.p. margin (TOST p = 0.014).
  - By the reading fixed in ADDENDUM_3 Sec. 4, the +6.38 p.p. of X1 is attributed to the action set. The paper says so explicitly.
- **At factor 1, the best of the 7 fixed rules is clustering (44.5%), outside D_R.** FedAvg (41.2%) and FedProx (40.5%) also beat the best rule of D_R, the trimmed mean (37.9%).
  - In all 12 E3 cells the best of the 7 rules lies outside D_R.
  - B7_−s exceeds B_R,−s by +6.61 p.p., almost exactly FedStrategist's X1 gain.
- **At factor 1/3 (the confirmatory factor), the same issue affects the headroom but not d\*.**
  - d\*₇ = d\*_R = FLTrust.
  - In 7 of the 12 E3 cells, the best of the 7 rules lies outside D_R.
  - Ĥ_LOSO over D_7 is 3.29 p.p. (2.36; 3.71), against 0.73 (0.26; 1.04) over D_R, and the CIs do not overlap. By Sec. 4, the paper states that headroom and d\* were defined over D_R and shows the D_7 values next to them.
  - FedStrategist against B7_−s is −0.83 p.p. (−2.04; +0.39): no evidence, so **inconclusive**. This equals the confirmatory comparison, since d\*₇ = d\*_R.
- **FedStrategist chooses almost uniformly among the 7 rules.**
  - At factor 1/3, its share of rounds per rule and cell stays between 9.1% and 18.4% (uniform: 14.3%).
  - At factor 1, it stays between 5.9% and 24.3%.
  - In 8 of the 10 cells where it beats the plain d\*_R at factor 1, it is worse than the single rule it chooses most there.

## 0. Execution and checks

- **Size:** 600/600 jobs: `x5_fixed` 360 and `x5_replay` 240.
  - They ran from 2026-11-14 15:29 to 2026-11-15 11:34.
  - There were 0 FAIL, 0 INFRA and 0 `SYNC_FAILED`, and no failed job was repeated.
- **Machines (stored results):**
  - `x5_fixed`: PC 143, vm2 108, vm3 109.
  - `x5_replay`: PC 107, vm2 66, vm3 67.
- **Checks passed before any statistic:**
  - both blocks match the job lists of `x5_job.py` exactly;
  - every X5 result carries the X5 code hash (`53e4cebc…`, `X5_CODE.sha256`);
  - every `x5_fixed` run recorded factor 1.
- **Blindness:** no `x5_fixed` accuracy was read before the block was complete. Monitoring read counts, errors, process state and the replay identity flags only.

### 0.1 Changes to the execution, none affecting what was computed

1. **PC processes.** The author raised the PC from 8 to 10 processes during the run. The dispatcher cannot add slots while running, so a second dispatcher was started on the PC with 2 slots (`x5_dispatch_tail.log`). It took the last 50 jobs (the `e3_main` replays) in reverse order (`x5_jobs_tail.txt`).
2. **Push loop.** A loop copied the PC's finished X5 results to the VMs every 10 minutes while the dispatchers ran. A VM job whose result already existed exited at once without running (`x5_job.py` skips existing outputs). This happened 46 times on vm3.
3. **Temporary slot limit.** Twice, the PC ran short of memory (about 4 GB available, swap exhausted). Both times it was running 8 dispatcher jobs, 2 tail jobs and up to 8 same-machine reproductions (Sec. 0.2).
   - The main dispatcher's PC slots were lowered from 8 to 6 until the reproductions finished.
   - No process was killed, and no job failed.
4. **Two jobs ran twice.** The tail dispatcher started two `e3_main` replays (α = 0.5, `label_flipping` and `sign_flipping`, seed 48) at 10:23, before the main dispatcher gave the same jobs to vm2 at 10:49.
   - The skip check runs only at start, so both copies ran. The PC copies (identical to the stored run) overwrote vm2's (not identical) at 11:23.
   - vm2's results were restored to `x5_replay/raw/`, since the main dispatcher had assigned these jobs and vm2's results came back first.
   - Both jobs were then reproduced on the stored run's machine like every other non-identical replay (Sec. 0.2).
   - The two PC copies were not used. They are identical to the stored runs, as their same-machine reproductions are.
5. **Path bug.** The same-machine reproduction script first failed to start the reproductions to be run on the PC (wrong interpreter path). Those 8 were relaunched after the fix. No result was affected.

### 0.2 Replays that did not reproduce their stored run

ADDENDUM_3 Sec. 3 accepts a replay only if its final parameter hash equals the stored run's. **46 of 240 replays did not.** All 46 are at α = 0.5, and all ran on a different machine from the stored run.

| α | replay on the stored run's machine | replay on another machine |
|---|---|---|
| 0.05 | 27/27 identical | 53/53 identical |
| 0.1 | 27/27 identical | 53/53 identical |
| 0.5 | 25/25 identical | **9/55 identical** (46 not) |

- 30 stored runs from the PC were replayed on a VM, and 16 from a VM on the PC.
- **Reproduction on the same machine.** Each of the 46 was run again, as soon as it appeared, on the machine that made its stored run.
  - Each ran in an isolated directory, with the same code (X5 hash `53e4cebc…`) and the same numerical environment.
  - **All 46 reproduced the stored run exactly**, both the final parameter hash and the accuracy curve (`x5_replay_same/raw/`).
- **Conclusion:** FedStrategist is deterministic on each machine. The divergence is a cross-machine numerical difference that grows over 110 rounds only at α = 0.5.
- **Decision**, taken after the first three reproductions and before the full analysis ran:
  - where the same-machine reproduction is identical, its recorded choices replace those of the non-identical replay;
  - this was applied to all 46, so **0 replays are excluded** and the FedStrategist frequencies cover all 240 stored runs.
  - Under the addendum's original rule, the 46 would have been excluded and the α = 0.5 frequencies labelled as partial.
  - The author authorized this decision in advance.
- No accuracy from a replay enters any comparison. The comparisons use the stored runs, which the replays only explain.

Accuracies below are the primary metric, the mean test accuracy over rounds 106–110, in p.p. n = 10 seeds. Every analysis uses root 100 and the 12 E3 cells. D_R = {FLTrust, median, trimmed mean, Krum}; D_7 = D_R ∪ {FedAvg, FedProx, clustering}.

## 1. X5c — the best of the 7 rules against the best of D_R

| | factor 1/3 (confirmatory) | factor 1 |
|---|---|---|
| d\*_R (in sample; LOSO choices) | FLTrust (10/10) | trimmed mean (10/10) |
| d\*₇ (in sample; LOSO choices) | **FLTrust** (10/10) | **clustering** (10/10) |
| cells whose best rule is outside D_R | **7/12** | **12/12** |
| Ĥ_LOSO over D_R | 0.73 (0.26; 1.04) | 1.66 (1.46; 1.95) |
| Ĥ_LOSO over D_7 | **3.29** (2.36; 3.71) | **3.60** (3.15; 3.95) |
| B7_−s − B_R,−s | 0.00 | **+6.61** (+6.03; +7.18), superiority |

The Ĥ CIs come from a seed bootstrap with 20,000 resamples that repeats the selection.

**Mean over the 12 cells, per rule:**

| rule | factor 1/3 | factor 1 |
|---|---|---|
| FLTrust | **40.50** | 35.50 |
| clustering | 38.29 | **44.53** |
| FedAvg | 35.74 | 41.15 |
| FedProx | 35.22 | 40.49 |
| trimmed mean | 31.87 | 37.92 |
| median | 29.79 | 35.11 |
| Krum | 22.43 | 23.48 |

**Cells whose best rule is outside D_R:**
- **Factor 1/3:**
  - FedAvg at α = 0.1 `label_flipping` (42.9 vs FLTrust 36.7).
  - Clustering at α = 0.1 `sign_flipping` (35.8 vs 35.0) and `trim_attack` (45.0 vs 37.7).
  - Clustering in all four α = 0.5 cells (by 2.6 to 6.8 p.p.).
- **Factor 1:**
  - FedAvg in 6 cells: all of α = 0.05 except `trim_attack`, α = 0.1 `krum_collusion` and `label_flipping`, and α = 0.5 `krum_collusion`.
  - Clustering in the other 6.
  - At α = 0.05 the gap is large: FedAvg 44.5 vs FLTrust 24.8 under `krum_collusion`, and 43.8 vs the trimmed mean 27.1 under `label_flipping`.

**Reading (Sec. 4, factor 1/3):** some rule outside D_R beats D_R on the E3 cells, and Ĥ_LOSO over D_7 exceeds Ĥ_LOSO over D_R with non-overlapping CIs. **The same issue affects the confirmatory result.** The paper states that headroom and d\* were defined over D_R and shows the D_7 values next to them. d\* itself does not change at factor 1/3 (FLTrust in both sets). The confirmatory result stands, and D_R stays the preregistered set.

Per-cell means: `analysis_x5/x5c_factor_1_3_cell_means.csv`, `analysis_x5/x5c_factor_1_cell_means.csv`.

## 2. Main comparison — FedStrategist against the cross-fitted best of the 7 rules

| factor | reference | Δ | CI95 | bootstrap CI95 (selection repeated) | Wilcoxon p | TOST p | category | reading (Sec. 4) |
|---|---|---|---|---|---|---|---|---|
| 1/3 | B7_−s (= B_R,−s: FLTrust) | −0.83 | (−2.04; +0.39) | (−1.92; +0.08) | 0.232 | 0.377 | no evidence | **inconclusive** |
| **1** | **B7_−s (clustering)** | **−0.22** | (−0.90; +0.45) | (−0.75; +0.35) | 0.432 | **0.014** | **equivalence** | **gain attributed to the action set** |
| 1 | B_R,−s (trimmed mean; the X1 comparison) | +6.38 | (+5.80; +6.97) | (+5.91; +6.88) | 0.002 | 1.000 | superiority | — |

The bootstrap resamples seeds (20,000 resamples) and repeats the selection of d\*, leaving every copy of the held-out seed out of it. It agrees with `describe` on 0 in every row. There is no Holm correction and no confirmatory test (ADDENDUM_3 Sec. 3).

**Reading at factor 1 (Sec. 4):** equivalence. **FedStrategist's gain in X1 is attributed to the action set, not to selection,** and the paper says so explicitly.
- FedStrategist matches a single fixed rule, clustering, chosen without the evaluated seed.
- Its +6.38 p.p. over the plain d\*_R is almost exactly the +6.61 p.p. by which the best of the 7 rules beats the best of D_R.

## 3. Per cell — FedStrategist against the best of the 7 rules (O7_−s, cross-fitted)

| α | attack | FS (1/3) | O7 (1/3) | FS − O7 | FS − B_R | FS (1) | O7 (1) | FS − O7 | FS − B_R |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | krum_collusion | 31.08 | 37.34 | −6.27 | −6.27 | 37.64 | 44.21 | −6.57 | +18.30 |
| 0.05 | label_flipping | 31.56 | 37.11 | −5.55 | −5.55 | 39.68 | 43.80 | −4.12 | +12.56 |
| 0.05 | sign_flipping | 28.06 | 36.94 | −8.88 | −8.88 | 35.36 | 37.70 | −2.34 | +12.74 |
| 0.05 | trim_attack | 24.63 | 36.29 | −11.66 | −11.66 | 30.16 | 39.34 | −9.19 | +11.84 |
| 0.1 | krum_collusion | 38.31 | 37.96 | +0.35 | +0.35 | 46.66 | 42.17 | **+4.49** | +12.14 |
| 0.1 | label_flipping | 41.62 | 42.89 | −1.27 | +4.89 | 45.37 | 45.96 | −0.60 | +2.59 |
| 0.1 | sign_flipping | 38.16 | 35.80 | +2.35 | +3.19 | 37.44 | 37.88 | −0.44 | −0.49 |
| 0.1 | trim_attack | 30.34 | 45.01 | −14.66 | −7.37 | 32.21 | 46.15 | −13.94 | +3.42 |
| 0.5 | krum_collusion | 53.45 | 53.18 | +0.27 | +7.11 | 57.60 | 58.94 | −1.34 | −0.37 |
| 0.5 | label_flipping | 55.91 | 54.84 | +1.07 | +6.66 | 58.57 | 61.07 | −2.50 | +0.15 |
| 0.5 | sign_flipping | 51.93 | 52.19 | −0.27 | +5.62 | 56.29 | 58.96 | −2.66 | +0.21 |
| 0.5 | trim_attack | 51.07 | 55.89 | −4.83 | +2.01 | 54.71 | 61.36 | −6.65 | +3.52 |

FS = FedStrategist. B_R is the cross-fitted global d\*_R (FLTrust at 1/3, the trimmed mean at 1).

- **Cells where FedStrategist is above O7_−s:** 4/12 at factor 1/3 (by +0.27 to +2.35) and 1/12 at factor 1 (α = 0.1 `krum_collusion`, +4.49).
- **Cells where it beats the plain d\*_R:** 7/12 at factor 1/3 and 10/12 at factor 1.

Table: `analysis_x5/fedstrategist_cells_factor_{1_3,1}.csv`.

## 4. X5b — action frequencies

### 4.1 FedStrategist, from the 240 replays

Share of rounds per rule, as the mean over the 12 cells (per-cell range in brackets):

| rule | factor 1/3 | factor 1 |
|---|---|---|
| FLTrust | 16.1 | 10.4 |
| median | 14.0 | 14.9 |
| trimmed mean | 14.1 | 15.3 |
| Krum | 11.5 | 8.5 |
| FedAvg | 14.8 | 16.3 |
| FedProx | 14.8 | 16.6 |
| clustering | 14.7 | 18.0 |
| range of one rule's share in one cell | 9.1–18.4 | 5.9–24.3 |

FedStrategist's choices are close to uniform (uniform: 14.3%). The largest share of any rule in any cell is 24.3%: clustering at factor 1, α = 0.1 `trim_attack`. Over 110 rounds, the bandit's choices stay spread over all arms.
- At factor 1/3 it leans slightly toward FLTrust and away from Krum.
- At factor 1 it leans toward clustering, FedAvg and FedProx (the rules that are best at factor 1) and away from FLTrust and Krum.

**In the cells where FedStrategist beats the plain d\*_R: FedStrategist minus the rule it chooses most there** (descriptive, uncorrected, n = 10):

| factor | cell | most chosen (share) | Δ | CI95 | category |
|---|---|---|---|---|---|
| 1/3 | 0.1 krum_collusion | FedAvg (15.8%) | +7.36 | (+4.41; +10.31) | superiority |
| 1/3 | 0.1 label_flipping | FLTrust (16.2%) | +4.89 | (+1.19; +8.58) | superiority |
| 1/3 | 0.1 sign_flipping | FLTrust (16.9%) | +3.19 | (−0.24; +6.61) | no evidence |
| 1/3 | 0.5 krum_collusion | trimmed mean (14.5%) | +2.98 | (+0.80; +5.16) | superiority |
| 1/3 | 0.5 label_flipping | FLTrust (14.9%) | +6.66 | (+4.02; +9.29) | superiority |
| 1/3 | 0.5 sign_flipping | clustering (14.5%) | −0.27 | (−2.77; +2.24) | no evidence |
| 1/3 | 0.5 trim_attack | clustering (15.1%) | −4.83 | (−6.79; −2.87) | inferiority |
| 1 | 0.05 krum_collusion | FedAvg (20.5%) | −6.87 | (−9.12; −4.61) | inferiority |
| 1 | 0.05 label_flipping | FedProx (23.0%) | −3.50 | (−5.62; −1.39) | inferiority |
| 1 | 0.05 sign_flipping | FedAvg (19.5%) | −2.70 | (−4.16; −1.24) | inferiority |
| 1 | 0.05 trim_attack | clustering (21.1%) | −9.19 | (−12.36; −6.02) | inferiority |
| 1 | 0.1 krum_collusion | clustering (18.2%) | +5.38 | (+2.26; +8.49) | superiority |
| 1 | 0.1 label_flipping | FedProx (19.4%) | −0.07 | (−1.68; +1.55) | no evidence |
| 1 | 0.1 trim_attack | clustering (24.3%) | −13.94 | (−18.26; −9.63) | inferiority |
| 1 | 0.5 label_flipping | clustering (17.2%) | −2.50 | (−4.00; −0.99) | inferiority |
| 1 | 0.5 sign_flipping | clustering (18.2%) | −2.66 | (−4.65; −0.68) | inferiority |
| 1 | 0.5 trim_attack | clustering (20.5%) | −6.65 | (−8.56; −4.75) | inferiority |

At factor 1, FedStrategist is worse than its most-chosen rule in 8 of its 10 winning cells. At factor 1/3 it is better in 4 of 7. At 1/3, though, its most-chosen rule is chosen in at most 16.9% of rounds, so it is barely a preference.

### 4.2 Layered systems, from the stored Layer-3 logs (`--part existing`)

Share of rounds per rule, as the mean over the 12 cells:

| system | factor | FLTrust | median | trimmed mean | Krum | FedAvg | FedProx | clustering |
|---|---|---|---|---|---|---|---|---|
| Random | 1/3 | 14.3 | 15.2 | 15.8 | 12.6 | 15.0 | 13.8 | 13.4 |
| Random | 1 | 14.2 | 15.3 | 15.9 | 12.7 | 14.9 | 13.8 | 13.3 |
| Attack-aware | 1/3 | 32.5 | 2.6 | 5.0 | 0.0 | 35.9 | 18.1 | 5.8 |
| Attack-aware | 1 | 15.8 | 6.7 | 9.2 | 0.0 | 53.3 | 4.2 | 10.8 |
| GRADF | 1/3 | 87.0 | 2.5 | 1.7 | 2.3 | 2.6 | 1.8 | 2.1 |
| GRADF | 1 | 75.7 | 3.8 | 3.7 | 3.8 | 4.3 | 3.7 | 5.0 |

- GRADF concentrates on FLTrust at both factors, even at factor 1, where FLTrust is not the best plain rule.
- The attack-aware reference shifts toward FedAvg at factor 1.
- Random stays close to 1/7 per rule, as expected.

Frequencies per cell: `analysis_x5/full.json` (FedStrategist) and `analysis_x5/existing.json` (layered systems).

## 5. What changes in the paper

- **RQ3 at factor 1 (X1):**
  - FedStrategist's superiority over the plain d\*_R (+6.38) is reported together with X5. Against the best of the 7 rules it is equivalent (−0.22, TOST p = 0.014), so the gain comes from the action set.
  - The X1 text no longer reads as evidence of useful selection at factor 1.
- **Reference set:**
  - Headroom and d\* are defined over D_R, as preregistered, and the D_7 values are shown next to them: Ĥ_LOSO 3.29 vs 0.73 at factor 1/3, and 3.60 vs 1.66 at factor 1.
  - At factor 1/3, d\* is FLTrust under both sets. At factor 1 it is clustering over D_7 against the trimmed mean over D_R.
- **The paper's own argument applies to its reference.** A comparison against a reference chosen from a smaller action set than the adaptive system's can credit selection with what is really the size of the action space. This is now reported, not only argued.
- **Unchanged:**
  - The confirmatory result (no system beats FLTrust at factor 1/3).
  - The preregistered D_R.
  - Every other X1 result.

## 6. Limits

- X5 is descriptive and uses the confirmatory seeds 42–51, so it confirms nothing. The per-cell tables are uncorrected.
- **The three added rules run plain only.**
  - This is enough for FedStrategist, which does not use the layers.
  - It does not test whether GRADF or Random, which pass through the layers and choose among 7 rules, gain from the larger set against a hardened 7-rule reference.
- **E2 is not extended** to the 7 rules on all 21 cells; the optional 270 jobs were not run.
- **The replays explain the stored runs; they do not add draws.**
  - At α = 0.5, cross-machine numerical differences changed 46 of 55 cross-machine replays.
  - The frequencies come from replays identical to the stored runs, either the original replay or a same-machine reproduction (Sec. 0.2).
  - Substituting the reproductions is a decision taken during the run. ADDENDUM_3 Sec. 3 had planned to exclude non-identical replays.
- **The cell means at factor 1 come from different blocks:** FLTrust and the trimmed mean from `e1_sens_factor`, the median and Krum from `x1_fixed`, the other three from `x5_fixed`. All share the configuration, the seeds and the campaign's frozen code.

## 7. Post-hoc descriptive calculations (requested after the analysis)

Two calculations on existing runs were requested after the X5 analysis. They are **not specified in ADDENDUM_3**, so they are post hoc and descriptive. They needed no new jobs. They ran once (`scripts/p1/analyze_x5_extra.py`) with the same functions as the X5 analysis, and they have no Holm correction.

**Consistency check:** the D_R rows reproduce the confirmatory E2 headroom (3.86 and 1.36) and the X1 deltas against the plain d\* (Random +1.78, attack-aware +1.77, GRADF +1.38, FedStrategist +6.38, AdaAggRL −9.17).

### 7.1 E2 headroom over D_7 at factor 1/3 (21 cells, root 100, `e1_main`)

| cells | set | Ĥ_LOSO | CI95 | d\* (LOSO choices) | cell winners | cells whose best rule is outside D_R |
|---|---|---|---|---|---|---|
| 21 | D_R (confirmatory) | 3.86 | (3.49; 4.17) | FLTrust (10/10) | FLTrust 12, trimmed mean 7, median 2 | — |
| 21 | **D_7** | **6.53** | (5.89; 6.97) | FLTrust (10/10) | FedAvg 7, FLTrust 6, clustering 6, trimmed mean 1, median 1 | **14/21** |
| 18, no FA | D_R (confirmatory) | 1.36 | (1.03; 1.66) | FLTrust (10/10) | FLTrust 12, trimmed mean 4, median 2 | — |
| 18, no FA | **D_7** | **3.55** | (2.85; 4.03) | FLTrust (10/10) | FLTrust 6, clustering 6, FedAvg 4, trimmed mean 1, median 1 | **10/18** |

- FA = `fltrust_aligned`.
- **Mean over the 21 cells:** FLTrust 38.26, FedAvg 36.13, clustering 35.74, FedProx 35.69, trimmed mean 35.62, median 33.13, Krum 22.55.
- **d\* is FLTrust under both sets** at factor 1/3. The headroom over D_7 is 1.7 times the D_R headroom on 21 cells and 2.6 times on 18, and the CIs do not overlap.
- Per-cell means: `analysis_x5/extra_e2_factor_1_3_cell_means.csv`.

### 7.2 Every system against B7_−s (plain), 12 E3 cells

For each system, the table gives Δ with its CI95 and category against B7_−s, then against B_R,−s.

| factor | system | mean | vs B7_−s: Δ | CI95 | bootstrap CI95 | category | vs B_R,−s: Δ | category |
|---|---|---|---|---|---|---|---|---|
| 1/3 | Random | 35.56 | −4.94 | (−6.32; −3.57) | (−6.06; −3.80) | inferiority | −4.94 | inferiority |
| 1/3 | Attack-aware | 37.61 | −2.89 | (−4.90; −0.89) | (−4.67; −1.39) | inferiority | −2.89 | inferiority |
| 1/3 | GRADF | 37.68 | −2.82 | (−5.70; +0.05) | (−5.31; −0.61) | no evidence; CIs disagree on 0 → **inconclusive** | −2.82 | same |
| 1/3 | FedStrategist | 39.68 | −0.83 | (−2.04; +0.39) | (−1.92; +0.08) | no evidence | −0.83 | no evidence |
| 1/3 | AdaAggRL | 25.21 | −15.30 | (−16.91; −13.68) | (−16.63; −13.98) | inferiority | −15.30 | inferiority |
| 1 | Random | 39.70 | **−4.83** | (−6.27; −3.39) | (−6.05; −3.67) | **inferiority** | +1.78 | superiority |
| 1 | Attack-aware | 39.69 | **−4.84** | (−6.28; −3.40) | (−6.05; −3.69) | **inferiority** | +1.77 | superiority |
| 1 | GRADF | 39.31 | **−5.22** | (−7.19; −3.26) | (−7.03; −3.84) | **inferiority** | +1.38 | no evidence |
| 1 | FedStrategist | 44.31 | −0.22 | (−0.90; +0.45) | (−0.75; +0.35) | **equivalence** | +6.38 | superiority |
| 1 | AdaAggRL | 28.75 | −15.78 | (−17.41; −14.15) | (−17.15; −14.49) | inferiority | −9.17 | inferiority |

- d\*₇ is FLTrust at factor 1/3 and clustering at factor 1. At factor 1/3, d\*₇ = d\*_R, so the two columns coincide.
- Wilcoxon p ≤ 0.006 for every inferiority and superiority above, except Random (0.027) and attack-aware (0.037) against B_R,−s at factor 1.
- All rows are against the **plain** rules. In the campaign, GRADF and Random are compared with the hardened d\*, which does not exist over D_7 (Sec. 6).

**Descriptive reading:**
- **At factor 1, the plain 7-rule reference reverses every superiority found against the plain d\*_R.** Random, attack-aware and GRADF go from +1.4 to +1.8 to −4.8 to −5.2. FedStrategist goes from +6.38 to equivalence.
- **No system is above the best of the 7 rules at either factor.** FedStrategist is the closest: equivalent at factor 1, and no evidence at factor 1/3.
- At factor 1/3 the comparisons equal the plain-reference ones, since d\* does not change.

Table: `analysis_x5/extra_systems_vs_dstar.csv`.

**Not done:** E2 over D_7 at factor 1 would need the 270 optional jobs. They were not run, since the argument already holds on the 12 E3 cells.

## 8. Files

- **Analysis:** `scripts/p1/analyze_x5.py`. Post-hoc calculations (Sec. 7): `scripts/p1/analyze_x5_extra.py`, which writes `analysis_x5/extra.json`, `extra.txt`, `extra_systems_vs_dstar.csv` and `extra_e2_factor_1_3_cell_means.csv`.
- **Outputs:** `results/p1_campaign/analysis_x5/`:
  - `existing.json`, `existing.txt`;
  - `full.json`, `full.txt`;
  - `fedstrategist_vs_dstar.csv`;
  - `fedstrategist_cells_factor_{1_3,1}.csv`;
  - `x5c_factor_{1_3,1}_cell_means.csv`.
- **Raw runs:** `x5_fixed/raw/` (360), `x5_replay/raw/` (240), `x5_replay_same/raw/` (46).
- **Dispatch logs:** `x5_dispatch.log`, `x5_dispatch_tail.log`. Job lists: `x5_jobs.txt`, `x5_jobs_tail.txt`.
