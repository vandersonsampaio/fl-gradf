# Result — MNIST rerun recording the trailing mean

**Date:** 2026-11-04. **Status: complete, re-analysis, not blind.** The final-round results on these seeds were known before the metric changed (`ADDENDUM.md`). The trailing mean is the mean of rounds 11–15.

**Pre-registration:** `ADDENDUM.md` (hash and code hashes check out). **Grid:** 50/50 jobs, 0 failures, from 2026-11-03 23:07 to 2026-11-04 07:44, 8 h 37 min, inside the 8 to 12 h estimated.

**Analysis:** `scripts/p1/analyze_mnist_trailing.py`, written after the runs and not frozen, reusing `scripts/p1/inference.py`. Outputs in `analysis/`.

**AdaAggRL:** not part of this rerun, since the jobs of `../mnist_rq3` do not include it. Its trailing mean comes from the separate runs of `ADDENDUM_2.md` (`mnist_adaaggrl/`: 20/20 jobs, 0 failures, ended 2026-11-04 15:55; code hashes check out). Sec. 4 uses them.

## 1. Regression of the final-round column

| arm | pairs | different | largest difference, p.p. |
|---|---|---|---|
| FedAvg, FedProx, clustering, FLTrust, median, trimmed mean, Krum | 2,520 | 0 | 0 |
| Random | 210 | 0 | 0 |
| Attack-aware | 210 | 0 | 0 |
| FedStrategist, variant a (own diagnostic) | 210 | 0 | 0 |
| GRADF | 210 | 60 | 69.62 |
| **FedStrategist, variant b (GRADF's detector)** | 210 | **46** | 17.34 |

- **GRADF's differences are the exception the addendum allows,** from its DQN (`../gradf_repeat`).
- **FedStrategist's differences were not expected. The arm is not deterministic on its own** (`fedstrategist_check/RESULT.md`): two runs of variant b alone, seed 45, differ in 8 of 21 final-round rows, by up to 6.34 p.p. The addendum allows only systems using GRADF's DQN, so this is still a mismatch.
  - The mean difference is +0.05 p.p. over 210 runs, and 1.23 p.p. in absolute value over the 46 that differ. They appear on all seeds and α.
  - They do not follow GRADF's: 24% of cells differ when GRADF's run matches and 17% when it differs.
  - Likely cause: variant b uses GRADF's Keras classifier, and on the logistic model TensorFlow is not seeded. The classifier's initial weights differ between processes without that seed and match with it. The arm itself was not rerun with the seed, so this is supported, not established.
  - **Label correction:** an earlier version of this section and the analysis scripts inverted the variants. In the code, variant a is the arm's own diagnostic and variant b uses GRADF's detector. `analysis/` and Sec. 4 now use the code's meaning, with variant a as the primary arm. `scripts/p1/run_parity.sh`, `audit_mnist.py`, `analyze_parity.py`, `run_mnist_inference.py`, `../PLAN.md` and `../RESULTS.md` still carry the old names (`fedstrategist_check/RESULT.md`).
- **For the campaign:** TensorFlow is seeded on the CNN, and FedStrategist's variant a matched across machines in `../../p1_campaign/checks/`. Variant b, the campaign's sensitivity arm, was not in that cross-machine comparison.

## 2. E1: root dependence

V = FLTrust − trimmed mean, D_s over 10 seeds (p.p., CI95).

| grid | contrast | final round | trailing mean |
|---|---|---|---|
| 18 cells | 3,000 vs 100 | +11.22 (+9.43; +13.02), superiority | **+8.60 (+7.88; +9.33), superiority** |
| | 100 vs 50 | +0.87 (−1.65; +3.39), no evidence | +1.07 (−0.39; +2.54), no evidence |
| 21 cells | 3,000 vs 100 | +10.88 (+9.35; +12.42), superiority | **+8.61 (+7.97; +9.24), superiority** |
| | 100 vs 50 | +1.47 (−0.67; +3.62), no evidence | **+1.64 (+0.36; +2.91), superiority** |

The root effect is smaller and tighter on the trailing mean. The one change of category is root 100 vs 50 on 21 cells, which rests on `fltrust_aligned`: it is not there on 18 cells.

## 3. E2: headroom

Root 100, D_R = {FLTrust, median, trimmed mean, Krum}. d\* is FLTrust under both metrics, with leave-one-seed-out agreement 1.00.

| grid | metric | H in-sample | Ĥ_LOSO (CI95) |
|---|---|---|---|
| 21 cells | final | 3.09 | 3.09 (2.56; 3.41) |
| | trailing | 2.31 | **2.27 (1.97; 2.58)** |
| 18 cells | final | 2.48 | 2.48 (1.87; 2.83) |
| | trailing | 1.67 | **1.63 (1.31; 1.95)** |

Headroom stays positive and shrinks by about a third on 18 cells. d\* gains from smoothing: FLTrust's mean over the 21 cells goes from 75.17 to 77.04, while the other three rules lose a little.

## 4. E3: capture

**Primary FedStrategist arm:** variant a, its own diagnostic, by the author's decision of 2026-11-04, after the labels were found inverted (Sec. 1). This matches the CIFAR-10 campaign. Variant b, GRADF's detector, is the sensitivity arm. The decision came after these results were known; each arm's values did not change, only which family it enters.

Per-seed differences (p.p., CI95), one execution. Categories as in `scripts/p1/inference.py`; the Holm value is in `analysis/`.

| comparison | 18 cells, final | 18 cells, trailing | 12 cells, final | 12 cells, trailing |
|---|---|---|---|---|
| GRADF − Random | +3.29 (+2.08; +4.51), sup. | **+3.46 (+2.20; +4.73), sup.** | +4.60 (+2.88; +6.32), sup. | **+4.73 (+2.99; +6.47), sup.** |
| FedStrategist − Random | −4.24 (−5.73; −2.75), inf. | **−2.27 (−3.02; −1.52), inf.** | −1.99 (−3.99; +0.01), no ev. | **−0.16 (−1.22; +0.90), no evidence** |
| AdaAggRL − Random | +0.65 (−1.31; +2.62), no ev. | **+0.12 (−1.28; +1.52), no evidence** | +1.62 (−0.82; +4.05), no ev. | **+1.27 (−0.43; +2.98), no evidence** |
| Attack-aware − Random | −2.48 (−3.18; −1.78), inf. | **−2.30 (−2.97; −1.62), inf.** | −3.81 (−4.91; −2.71), inf. | **−3.57 (−4.56; −2.59), inf.** |
| GRADF − B₋ₛ | +7.70 (+6.02; +9.38), sup. | **+4.85 (+3.76; +5.95), sup.** | +6.94 (+4.72; +9.15), sup. | **+4.14 (+2.49; +5.80), sup.** |
| FedStrategist − B₋ₛ | +0.16 (−1.71; +2.04), no ev. | **−0.88 (−1.36; −0.40), inf.** | +0.35 (−1.70; +2.41), no ev. | **−0.74 (−1.35; −0.13), inf.** |
| AdaAggRL − B₋ₛ | +5.06 (+2.12; +8.00), sup. | **+1.51 (+0.09; +2.93), sup.** | +3.96 (+0.53; +7.38), sup. | **+0.69 (−1.17; +2.55), no evidence** |
| Attack-aware − B₋ₛ | +1.93 (+0.21; +3.65), sup. | **−0.91 (−1.67; −0.14), inf.** | −1.47 (−3.50; +0.56), no ev. | **−4.16 (−5.10; −3.22), inf.** |
| Random − B₋ₛ | +4.41 (+2.81; +6.01), sup. | **+1.39 (+0.35; +2.43), sup.** | +2.34 (+0.47; +4.21), sup. | **−0.58 (−1.84; +0.67), no ev.** |

**Sensitivity arms, on the trailing mean:**

| arm | comparison | 18 cells | 12 cells |
|---|---|---|---|
| FedStrategist with GRADF's detector (variant b) | vs Random | −2.40 (−3.15; −1.65), inf. | −0.30 (−1.30; +0.70), no ev. |
| | vs B₋ₛ | −1.01 (−1.65; −0.37), inf. | −0.88 (−1.67; −0.09), inf. |
| AdaAggRL, pretrained extractor | vs Random | −1.37 (−2.46; −0.27), inf. | −1.03 (−2.17; +0.11), no ev. |
| | vs B₋ₛ | +0.03 (−1.18; +1.23), no ev. | −1.62 (−3.02; −0.21), inf. |

**AdaAggRL (`ADDENDUM_2.md`):**

- **It does not differ from Random in any of the four columns,** and it is above B₋ₛ only as Random is: on the 18 cells, and less clearly on the trailing mean. On the 12 cells of level 1, trailing, it has no evidence against either reference.
- **The trailing mean removes most of its advantage over B₋ₛ:** from +5.06 to +1.51 p.p. on 18 cells, and that +1.51 does not survive Holm (0.098).
- **The extractor changes the reading.** The pretrained arm is lower than the random one in every column, and on the trailing mean is below Random on 18 cells (Holm 0.014). The pretrained extractor is not the better version here.
- **The difference from the published runs is reported and not read as an effect of anything,** since those runs are unseeded and use the native g₀ under `fltrust_aligned`. Final round, 210 pairs per extractor:

| extractor | pairs that differ | mean difference, p.p. | mean absolute, p.p. | largest, p.p. |
|---|---|---|---|---|
| random | 205 | +0.47 | 1.66 | 26.18 |
| pretrained | 204 | +0.04 | 2.04 | 22.48 |

**Comparisons among the other systems:**

- **Comparisons with B₋ₛ fall by about 3 p.p., and by about 1 p.p. for FedStrategist.** FLTrust, the reference, gains 1.88 p.p. on average from the trailing mean, so part of the advantage measured at the final round was its final-round dip.
- **GRADF stays above Random and above B₋ₛ in all four columns.** This is one execution of a non-deterministic system. `../gradf_repeat` and `../gradf_ablation` state what it does and does not show: the gain comes from the layers.
- **Random against B₋ₛ loses its result on the 12 cells of level 1,** and attack-aware turns negative.
- **Holm on the Wilcoxon test, trailing mean, now with AdaAggRL in the families.** These results hold by CI but do not survive Holm: Random − B₋ₛ (0.059), attack-aware − B₋ₛ (0.098) and AdaAggRL − B₋ₛ (0.098) on 18 cells, and FedStrategist − B₋ₛ on 12 cells (0.082). FedStrategist − B₋ₛ on 18 cells does survive (0.039). The family grew, so these values replace the earlier ones, and Random − B₋ₛ on 18 cells is a new loss.

## 5. Win counts and the size of the metric change

| root | final: FLTrust / median / TM / Krum | trailing: FLTrust / median / TM / Krum | same cell winner |
|---|---|---|---|
| 3,000 | 21 / 0 / 0 / 0 | 21 / 0 / 0 / 0 | 21 of 21 |
| 100 | 8 / 0 / 11 / 2 | 9 / 0 / 11 / 1 | 19 of 21 |
| 50 | 8 / 0 / 11 / 2 | 8 / 0 / 12 / 1 | 20 of 21 |

Trailing minus final, root 100, per run: FLTrust +1.88 on average with a mean absolute change of 3.64 and a maximum of 16.47; trimmed mean −0.92, 0.92 and 2.69. The other rules gain nothing from smoothing because they are still climbing at round 15. **The metric is not neutral between rules:** it lifts the unstable rule and lowers the rising ones.

## Limits

- Not blind (Sec. 7 of the campaign's pre-registration states this).
- 15 rounds on MNIST: rounds 11–15 still include a rising curve.
- One partition per α.
- GRADF and FedStrategist come from one execution each, and both vary between executions. AdaAggRL also comes from one execution per extractor. Its determinism on repetition was not checked.

## Outputs

`analysis/analysis.txt`, `analysis/analysis.json`, `analysis/regression.csv`, `analysis/e3_*.csv` (AdaAggRL's pretrained arm in `*_sensitivity_adaaggrl.csv`), `analysis/wins.csv`, `analysis/grid_root100_rules.csv`, `analysis/grid_systems.csv`.
