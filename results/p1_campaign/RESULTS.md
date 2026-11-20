# Results — P1 confirmatory campaign on CIFAR-10

**Date:** 2026-11-11. **Status: CONFIRMATORY**, analysis run once on the complete blocks (`scripts/p1/analyze_campaign.py`, built on the frozen `scripts/p1/inference.py`). Pre-registration: `PREREGISTRATION.md` (frozen 2026-11-04) and `ADDENDUM_1.md`.

**Short answer.**
- **RQ1:** FLTrust's advantage over the trimmed mean grows with the root at factor 1/3 (+8.4 p.p.) but shrinks at factor 1 (−1.6 p.p.). The categories differ, so by the rule fixed in Sec. 5 **the root effect is reported as dependent on the global rate, and no general claim about the root is made.** The sign of V also flips: FLTrust leads the trimmed mean at 1/3 and trails it at 1.
- **RQ2:** cross-fitted headroom is positive: Ĥ_LOSO = 3.9 p.p. (CI95 3.5 to 4.2) on 21 cells, and 1.4 p.p. (1.0 to 1.7) without `fltrust_aligned`.
- **RQ3:** every system with a learned or oracle selector beats Random except AdaAggRL, and **no system beats the best fixed rule** (FLTrust). GRADF and FedStrategist are not distinguishable from it; Random, the attack-aware reference and AdaAggRL are worse.

## 0. Execution and checks

- **Size:** 4,128/4,128 jobs, 0 FAIL, 0 `SYNC_FAILED`. Every block matches its frozen job list exactly, and every result carries the frozen code hash (`d5ed26d6…`).
- **Repeats (ADDENDUM_1):** 25 jobs lost their VM connection and were requeued once; none was dropped. On rerun, 3 of them found their JSON already written and skipped, as the addendum allows.
- **Machines:** vm3 1,441 runs, this PC 1,372 runs, vm2 1,315 runs.
- **PC restart (2026-11-11 11:44):** the PC restarted from lack of memory and the dispatcher stopped.
  - The 8 jobs running on the PC were lost with no result and ran again from scratch.
  - The 8 jobs running on the VMs finished there and wrote their JSON with the frozen hash. Their results were copied to the other machines, so the restarted dispatcher found them and logged them as DONE without running them again. No result was replaced.
- **Blindness:** no accuracy of seeds 42–51 was read before the blocks were complete.
- **Trimmed-mean invariance (first output of E1):** TM at root 100 and at root 2,500 are identical on the 18 cells outside `fltrust_aligned` (max |Δ| = 0). Under `fltrust_aligned` they differ on all 30 runs, by up to 8.4 p.p., as expected: that attack uses the server's root direction. E1 therefore uses TM at root 2,500 for that attack, from `e1_invariant` (factor 1/3) and from `e1_sens_factor` (factor 1).

Accuracies are the primary metric, the mean test accuracy over rounds 106–110, in p.p.

## 1. E1 — does FLTrust's advantage depend on the root? (RQ1)

D_s = mean over cells of [V(c, s, 2,500) − V(c, s, 100)], with V = FLTrust − TM. n = 10 seeds.

| factor | cells | D | CI95 | Wilcoxon p | category | V at root 100 | V at root 2,500 |
|---|---|---|---|---|---|---|---|
| **1/3 (H1)** | 21 | **+8.40** | (+7.89; +8.90) | 0.002 | **superiority** | +2.64 | +11.03 |
| 1/3 | 18, no FA | +6.83 | (+6.21; +7.46) | 0.002 | superiority | +6.21 | +13.05 |
| 1 (sensitivity) | 21 | **−1.61** | (−2.17; −1.04) | 0.002 | **inferiority** | −8.38 | −9.98 |
| 1 | 18, no FA | −5.12 | (−5.77; −4.46) | 0.002 | inferiority | −5.36 | −10.47 |

FA = `fltrust_aligned`.

**Reading (fixed in advance):**
- **H1 holds at factor 1/3, but the sensitivity at factor 1 lands in the opposite category.** The root effect is therefore reported as **dependent on the global rate**, with no general claim about the root.
- **The level of V changes sign too:**
  - At factor 1/3, FLTrust beats the trimmed mean at both roots. It wins 12/21 cells at root 100 and 19/21 at root 2,500.
  - At factor 1 it loses at both roots. It wins 4/21 cells at root 100 and 2/21 at root 2,500.
- **Pre-stated limit:** the adopted factor was tuned in a way that favours FLTrust under attack (PREREGISTRATION Sec. 10). This is the reading to keep in mind.

**Server steps (factor 1, descriptive factorial).** These runs use FLTrust with `server_steps = native` instead of parity.

| root | FLTrust, parity | FLTrust, native | native − parity (CI95) |
|---|---|---|---|
| 100 | 33.91 | 37.44 | +3.53 (+2.73; +4.34) |
| 2,500 | 32.67 | 41.66 | +9.00 (+8.59; +9.40) |

- **Root effect with native steps:** D = +3.85 p.p. (+3.11; +4.59), so the root helps FLTrust under native steps at factor 1.
- **Against the trimmed mean:** FLTrust still trails it, by −4.84 p.p. at root 100 and −0.99 p.p. at root 2,500.
- **Conclusion:** at factor 1, the sign of the root effect also depends on how the server update is computed.

## 2. E2 — headroom outside the sample (RQ2)

These figures use root 100 and factor 1/3. The comparison set D_R is {FLTrust, median, trimmed mean, Krum}, and the CIs come from a seed bootstrap with 20,000 resamples.

| cells | in-sample H (optimistic) | Ĥ_LOSO | CI95 | category | d\* | LOSO agreement: d\* / cell winner |
|---|---|---|---|---|---|---|
| **21 (H2)** | 3.93 | **3.86** | (3.49; 4.17) | **positive** | FLTrust | 100% / 97% |
| 18, no FA | 1.45 | 1.36 | (1.03; 1.66) | positive | FLTrust | 100% / 96% |

- **H2 holds.** Little of the headroom comes from the winner's curse: LOSO is within 0.1 p.p. of the in-sample value.
- **Where the headroom comes from:** most of it is `fltrust_aligned`.
  - There FLTrust falls to 22–24%, while the trimmed mean and the median reach 27–54%.
  - The rest comes from α = 0.5, where the rules are close and FLTrust is not always first.
  - Per-cell means are in `analysis/e1_main_cell_means_r100.csv`.

## 3. E3 — how much do adaptive systems capture? (RQ3)

This block covers 12 cells: 3 α × {sign flipping, label flipping, trim attack, Krum collusion}, at factor 1/3. B_−s is cross-fitted, and d\* = FLTrust under both references. Holm is applied within each family, n = 10 seeds.

**Confirmatory results:**

| family | system | reference | Δ | CI95 | Wilcoxon p (Holm) | category |
|---|---|---|---|---|---|---|
| vs Random | Attack-aware | Random | +2.05 | (+0.60; +3.50) | 0.020 | superiority |
| | GRADF | Random | +2.12 | (+0.14; +4.10) | 0.037 | superiority |
| | FedStrategist | Random | +4.12 | (+3.22; +5.01) | 0.008 | superiority |
| | AdaAggRL | Random | −10.35 | (−12.11; −8.59) | 0.008 | inferiority |
| vs B_−s | Random | hardened | −4.25 | (−6.39; −2.10) | 0.029 | inferiority |
| | Attack-aware | plain | −2.89 | (−4.90; −0.89) | 0.023 | inferiority |
| | **GRADF** | **hardened** | −2.12 | (−4.38; +0.14) | 0.129 | **no evidence** |
| | **FedStrategist** | plain | −0.83 | (−2.04; +0.39) | 0.232 | **no evidence** (TOST p 0.38, not equivalent) |
| | AdaAggRL | plain | −15.30 | (−16.91; −13.68) | 0.010 | inferiority |

**Like-for-like note.** GRADF and Random are compared against the hardened rules, the confirmatory reference fixed in Sec. 5.
- The attack-aware reference also passes through Layers 1, 2 and 4, as the stored layer logs show. Sec. 5 does not name it, so the table follows the letter and uses the plain rules.
- Against the hardened rules it is also worse: −2.19 (−4.22; −0.17). The category does not depend on that choice.

**Descriptive results:**
- **GRADF against the plain rules:** −2.82 (−5.70; +0.05), no evidence.
- **Random against the plain rules:** −4.94, inferiority.
- **Hardened B_−s against plain B_−s:** −0.70 (−3.18; +1.78), no evidence. The layers add nothing on CIFAR-10 at this factor.
- **Mean accuracy over the 12 cells:**
  - fixed rules: FLTrust 40.50 (hardened 39.80), trimmed mean 31.87, median 29.79, Krum 22.43;
  - systems: FedStrategist 39.68, GRADF 37.68, attack-aware 37.61, Random 35.56, AdaAggRL 25.21.
- **Cross-fitted headroom on the E3 grid** is small. Only 3 of the 12 cells have it against the plain rules, and none against the hardened ones, where FLTrust is best everywhere.
  - **Cells beating B_−s on average:** FedStrategist 7/12 (all 3 headroom cells), attack-aware 5/12, GRADF 4/12, AdaAggRL 2/12, Random 0/12.
  - FedStrategist's wins are mostly at α = 0.5, where FLTrust is not the cell's best rule.
- **Layer-4 rejection rate per round:**

  | arm | rate |
  |---|---|
  | Random | 29% |
  | GRADF | 28% |
  | Attack-aware | 18% |
  | Hardened FLTrust | 21% |
  | Hardened TM | 23% |
  | Hardened median | 29% |
  | Hardened Krum | 51% |

- **Rounds with an honest or attacked client rejected at Layer 1:** 5–12%.

**Sensitivity:**

| system | vs Random | vs B_−s (plain) |
|---|---|---|
| AdaAggRL, pretrained extractor | −10.59, inferiority | −15.54, inferiority |
| FedStrategist with GRADF's detector | +4.50, superiority | −0.45 (−1.77; +0.87), no evidence |

The extractor does not rescue AdaAggRL. Swapping in GRADF's detector changes FedStrategist's result by less than half a point.

**Reading (fixed in advance).** AdaAggRL stays far below every fixed rule on CIFAR-10, as on MNIST. P1 does not attribute this to any internal component. The E3 conclusions hold at factor 1/3 only, since E3 was not run at factor 1.

## 4. Limits

- **R = 110 is short at factor 1/3.** Accuracies here range from about 10% to 56%, and conclusions hold at this horizon.
- **The factor was tuned in FLTrust's favour,** with a margin that failed. E1 reverses at factor 1, and E3 has no factor-1 counterpart.
- **GRADF's DQN is not deterministic,** so its runs are one draw each. No machine effect was tested; the machine of every run is in `analysis/all_runs.csv`.
- **The aggregates average over cells whose accuracy levels differ widely,** from near chance at α = 0.05 to about 55% at α = 0.5. Per-cell values are in the CSVs.
- **`low_mag_backdoor` was not checked as an effective attack on the CNN.** It stays in E1 and E2, as Sec. 7 states.

## Outputs (`analysis/`)

- `analysis.json`: every number above, including D_s and the H_s values per seed.
- `e3_confirmatory.csv`, `e3_descriptive.csv`, `e3_sensitivity.csv`: the E3 tables.
- `e1_main_cell_means_r100.csv` and `e3_cell_means.csv`: per-cell means.
- `all_runs.csv`: one row per run, with block, arm, cell, seed, primary metric, final round, machine, and the Layer-1 and Layer-4 rates.
