# Results — P1 step-parity reruns (items 1–4)

**Date:** 2026-11-02.

**Status: complete, not confirmatory.** All runs finished with no failures and the integrity checks pass. Two findings need the author's decision before the CIFAR-10 pre-registration: GRADF's change of sign on MNIST (Sec. 4) and FLTrust's instability on CIFAR-10 (Sec. 5). The MNIST results are re-analysis, as the P1 plan states. The CIFAR-10 results cover 2 cells and 3 seeds and are descriptive.

**Plan:** `PLAN.md` (frozen 2026-10-30 16:30, `PLAN.sha256` checks out).

**Runs:** 2026-11-01 12:40–20:09, CPU, 1 thread per process, chain log in `chain.log`.

| item | jobs | failures | wall time | planned |
|---|---|---|---|---|
| 1 `mnist_fixed` | 30/30 | 0 | 3 h 38 | ~1–1.5 h |
| 3 `cifar_blockC` | 12/12 | 0 | ~2 h 11 per job | ~1–1.5 h |
| 2 `mnist_rq3` | 20/20 | 0 | 2 h 45 | ~3 h |
| 4 part D | `costs.json` written | 0 | 1 h 05 | ~1 h |

Items 1 and 3 ran together on 20 processes, which is the likely reason for their overrun.

**Analysis:** `scripts/p1/analyze_parity.py` (sha256 `ae4dfe00…`) → `analysis/analysis.txt` (`f5690a01…`), `analysis/*.csv`, `analysis/analysis.json`; tables and figures in `../p1_tables/parity/`. The script was written after the runs ended and was not frozen. It reuses the frozen `scripts/p1/inference.py` and `scripts/p1/analyze_e0b.py`. After its first run I added the stability table and the two extra descriptive lines; every earlier number stayed identical.

**Sources.** Fixed rules, Random, attack-aware, GRADF and FedStrategist use `server_steps = median`. AdaAggRL does not use g₀ and was not rerun, so it comes from the published unseeded runs. Under `fltrust_aligned` those rows still hold the native g₀, so the 21-cell counts that include AdaAggRL mix the two versions there. The 18-cell results do not.

## 1. Integrity checks

- **Completeness:** every expected row and job is present, and every run records `server_steps = median`.
- **Regression (PLAN notes):** non-FLTrust rules outside `fltrust_aligned` equal `results/p1_invariance` exactly: 1,620 pairs, 0 differences.
- **Expected changes:** all 270 non-FLTrust pairs under `fltrust_aligned` change, as the plan predicts, and all 630 FLTrust pairs change.
- **Root invariant under parity (median, trimmed mean, Krum, outside `fltrust_aligned`):** 1,080 pairs, 0 differences.

## 2. MNIST: root dependence (E1)

V = FLTrust − trimmed mean, D_s over 10 seeds, 18 cells without `fltrust_aligned`:

| contrast | native | parity |
|---|---|---|
| root 3,000 vs 100 | +15.90 (+14.93; +16.87) | **+11.22 (+9.43; +13.02), superiority** |
| root 100 vs 50 | +8.23 (+5.98; +10.49) | +0.87 (−1.65; +3.39), no evidence |

FLTrust 2×2, 18 cells (p.p.; CI95):

| contrast | mean | CI95 | α 0.5 | α 0.1 | α 0.05 | cells > 0 |
|---|---|---|---|---|---|---|
| root effect at parity | +11.22 | (+9.43; +13.02) | 3.15 | 14.72 | 15.79 | 18 |
| root effect, native | +15.90 | (+14.93; +16.87) | 8.99 | 20.98 | 17.72 | 18 |
| steps effect at root 100 | +4.69 | (+2.37; +7.01) | 6.41 | 6.61 | 1.05 | 15 |
| steps effect at root 3,000 | +0.01 | (−0.19; +0.21) | 0.57 | 0.35 | −0.88 | 11 |

- Step parity reduces the root effect by about 30%, and **most of it remains**. This agrees with the diagnostic's D_root@median (10.5 p.p. on seeds 0–2, `../p1_g0_probe/RESULTS.md`). That diagnostic's smaller residual root effect (1.8 p.p.) is a different quantity: a main effect over 20 and 470 server steps, not the clients' median.
- The root effect grows as heterogeneity grows.
- The drop from root 100 to 50 is no longer detected.

Win counts by root, 21 cells (Table 1):

| root | parity: FLTrust / median / TM / Krum | native seeded: FLTrust / median / TM / Krum |
|---|---|---|
| 3,000 | 21 / 0 / 0 / 0 | 18 / 1 / 1 / 1 |
| 100 | 8 / 0 / 11 / 2 | 9 / 1 / 9 / 2 |
| 50 | 8 / 0 / 11 / 2 | 6 / 1 / 12 / 2 |

## 3. MNIST: headroom (E2)

Root 100, D_R = {FLTrust, median, TM, Krum}, 10 seeds.

| grid | d\* | H in-sample | Ĥ_LOSO (CI95) | native Ĥ_LOSO (CI95) |
|---|---|---|---|---|
| 21 cells | FLTrust | 3.09 | **3.09 (2.56; 3.41)** | 4.63 (4.17; 4.99) |
| 18 cells | FLTrust | 2.48 | **2.48 (1.87; 2.83)** | 4.91 (4.45; 5.25) |

- Headroom is positive out of sample and smaller than with the native g₀.
- In-sample and LOSO coincide because selection never changes: d\* and all cell winners are the same in every leave-one-out set. With one partition per α, this says the choice is stable across seeds, not across partitions.
- Headroom cells: 13 of 21; 10 of 18.

## 4. MNIST: capture (E3)

18 confirmatory cells, per-seed differences (p.p.). "Earlier" is the same comparison with native fixed rules and the published unseeded systems.

| comparison | earlier | parity (CI95) | Holm Wilcoxon p | category |
|---|---|---|---|---|
| GRADF − Random | −4.64 | **+3.48 (+2.13; +4.84)** | 0.008 | superiority |
| FedStrategist − Random | −5.19 | −4.66 (−5.77; −3.55) | 0.008 | inferiority |
| AdaAggRL − Random | +4.16 | **+0.41 (−1.71; +2.54)** | 0.846 | no evidence |
| Attack-aware − Random | +1.24 | −2.48 (−3.18; −1.78) | 0.008 | inferiority |
| GRADF − B₋ₛ | +0.71 | +7.89 (+5.86; +9.92) | 0.010 | superiority |
| FedStrategist − B₋ₛ | +0.15 | −0.26 (−1.67; +1.16) | 0.846 | no evidence |
| AdaAggRL − B₋ₛ | +9.51 | +4.82 (+1.74; +7.90) | 0.041 | superiority |
| Attack-aware − B₋ₛ | +6.59 | +1.93 (+0.21; +3.65) | 0.098 | superiority by CI, not after Holm |
| Random − B₋ₛ | +5.35 | +4.41 (+2.81; +6.01) | 0.016 | superiority |

Sensitivity: AdaAggRL (pretrained) − Random −1.73 (−4.04; +0.57), no evidence; FedStrategist (GRADF's detector) − Random −4.24 (−5.73; −2.75), inferiority.

**How to read this:**

- **The manuscript's capture story does not carry over as it stands.** GRADF − Random and AdaAggRL − Random both differ from the published pattern.
- **The change is not attributed to the correction.** Per the plan, the earlier systems were unseeded, so this table reports regenerated numbers and not a causal contrast.
- **GRADF needs more scrutiny before it is cited.**
  - This is a single execution of a non-deterministic system. D2 measured a run-to-run SD of 1.77 p.p. for its aggregate Δ (`fl-fedmed/results/d2_variancia_p1`), on the earlier code.
  - It is above the cell-wise best fixed rule in 16 of 21 cells (AdaAggRL 12, FedStrategist 7). That is unusual for a selector over those rules, and **its mechanism was not examined**.
- **AdaAggRL's accuracy did not change.** Its margin fell because the comparators moved: Random's mean over the 21 cells is 80.0%, against 76.2% earlier.

## 5. CIFAR-10: block C with parity

α = 0.05, `sign_flipping` and `trim_attack`, root {100, 2,500}, seeds 0–2, 180 rounds. The server takes 303 steps at both roots. Median ‖g₀‖ is 1.03–1.30, regardless of the root.

Accuracy of FLTrust (%, seed means; S = 5-round trailing mean; change = mean absolute change between consecutive rounds, rounds 51–180):

| attack | root | arm | at R, unsmoothed | S110 | S150 | S180 | S180 − S110 | change |
|---|---|---|---|---|---|---|---|---|
| sign_flipping | 100 | parity | 18.77 | 23.11 | 27.00 | 27.40 | +4.29 | 8.31 |
| | | native | 35.08 | 33.90 | 35.83 | 36.24 | +2.33 | 4.95 |
| | 2,500 | parity | 17.17 | 18.25 | 18.94 | 21.14 | +2.89 | 7.50 |
| | | native | 27.34 | 26.47 | 23.71 | 23.15 | −3.32 | 4.23 |
| trim_attack | 100 | parity | 26.76 | 25.18 | 30.66 | 29.74 | +4.57 | 7.37 |
| | | native | 16.45 | 17.54 | 19.80 | 20.86 | +3.32 | 1.70 |
| | 2,500 | parity | 17.54 | 18.26 | 21.99 | 20.81 | +2.55 | 8.19 |
| | | native | 23.81 | 25.21 | 24.54 | 20.62 | −4.59 | 6.12 |

FedAvg at R, same seeds: 38.35 (`sign_flipping`), 26.71 (`trim_attack`).

- **FLTrust is unstable under parity.** It moves about 8 p.p. between consecutive rounds, so a final-round metric carries a large part of that swing.
- **R = 110 does not cover FLTrust under parity.** It still gains 2.6–4.6 p.p. between S110 and S180. The E0b rule for R reads only FedAvg without attack, so the rule was met as written; this is a limit of the rule, not a breach.
- **The larger root does not help.** It is below root 100 in both cells on S110, S150 and S180, the opposite direction from MNIST. Two cells and three seeds: E1 tests this on the full grid.
- **The bimodality of root 100 under `trim_attack` is gone.** The native seeds ended at 10.9, 9.3 and 29.2% at R; with parity they are 26.3, 25.3 and 28.6%.
- **FLTrust does not beat FedAvg at R** in these cells: far below it under `sign_flipping`, level with it at root 100 under `trim_attack`.

### Values for the pre-registration (`e0b/PLAN.md`, Secs. 3–4, R = 110)

| quantity | E0b (native block C) | parity block C |
|---|---|---|
| τ_CNN | 7.86 | **7.86** (largest honest norm still in block B; block C max 1.66) |
| σ_E0^E1 | 0.77 | **1.69**; 0.5σ = 0.85 |
| σ_E0^E3 | 1.23 | **2.93**; 0.5σ = 1.47 |

- **Sensitivity margin of E3:** 0.5·σ_E0^E3 = 1.47 p.p. is now **wider** than the primary 1.0 p.p. margin, so that TOST is more lenient than the primary one. The primary margin stays binding and is not relaxed.
- **The E0b limitation on σ_E0^E1 no longer applies in the same form.** Both root contrasts now have the same sign, so they no longer cancel. It still rests on 2 cells and 3 seeds, and it uses the unsmoothed final round.

## 6. Costs and compute budget

Part D (`../p1_cifar10/e0b/costs.json`), 12 processes, α = 0.5, `sign_flipping`, s per round: FLTrust 23.1, fixed rule (TM) 24.9, Random 24.6, attack-aware 24.7, GRADF 25.1, FedStrategist 24.5, AdaAggRL 36.5. Pretraining: 52 min per run, 12.9 h over the grid.

This replaces the provisional costs of E0b Sec. 5. Budget 236 h, R = 110:

| level | k | frozen cost model | with the E1 factorial (FLTrust native at both roots, 420 runs) | margin with factorial |
|---|---|---|---|---|
| 0 | 18 | 209.3 h | 235.9 h | **+0.09 h** |
| 1 | 12 | 175.0 h | 201.6 h | +34.4 h |
| 2 | 6 | 140.7 h | 167.3 h | +68.7 h |

- The frozen rule gives **level 0**, with and without the factorial, which was not in the rule when it was frozen. The factorial is costed at s_fixed; at s_FLTrust the margin is +1.97 h.
- **Level 0 with the factorial has no real margin.** Costs come from one cell. Block C took 43.6 s per round on 20 simultaneous processes, against 23–25 s measured on 12 and 15–18 s native on 6. Contention and parity's server steps cannot be separated from these data.
- The 236 h assumed the Mon–Fri 8–18 pause, cancelled on 2026-11-01, so it now takes about 10 calendar days rather than 2 weeks.

## 7. Open decisions (author)

1. **CIFAR-10 metric for FLTrust and the systems that use it.** Keep the unsmoothed final round at R = 110, or register a 5-round trailing mean by addendum (σ_E0 would be recomputed on it). The instability of Sec. 5 is the reason to ask.
2. **Budget.** Level 0 with the factorial at no margin, a small accepted overrun, or level 1.
3. **GRADF on MNIST.** Repeat executions and examine the mechanism behind Sec. 4 before the pre-registration. It uses the CPU; `mnist_rq3` took 2 h 45 on 10 processes, and P1 runs do not overlap W1 unless the author allows it.

Still to do after these: the guard of the confirmatory campaign and the pre-registration (`PLAN.md`, notes).

## Outputs

- `analysis/analysis.txt`, `analysis/analysis.json`: full output.
- `analysis/regression_vs_native.csv`, `invariance_check.csv`: Sec. 1.
- `analysis/e1_2x2_by_alpha.csv`: Sec. 2.
- `analysis/e3_confirmatory_18cells.csv`, `e3_sensitivity_18cells.csv`, `table3_grid_root100.csv`: Sec. 4.
- `analysis/cifar_blockC_runs.csv`, `cifar_fltrust_stability.csv`: Sec. 5.
- `../p1_tables/parity/`: Tables 1–5 and Figs. 1–3 regenerated (`tables.tex`, `figure_data.tex`, PNGs).
