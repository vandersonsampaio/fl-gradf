# Result — Addendum 3: MNIST against the 7-rule set and the hardened reference

**Date:** 2026-11-15. **Status: descriptive re-analysis, not blind** (`ADDENDUM_3.md`; its hashes checked out at analysis time). The analysis ran once (`scripts/p1/analyze_mnist_addendum3.py` → `analysis_addendum3/`). Metric: trailing mean (rounds 11–15), seeds 42–51, root 100, p.p.

**Short answer.**
- **Against the hardened reference, GRADF is inferior on MNIST too:**
  - −1.23 p.p. (−2.18; −0.28) on 18 cells and −1.72 (−3.14; −0.29) on 12, with the bootstrap agreeing.
  - Hardened FLTrust (82.45% over 21 cells) beats every system. All five systems are inferior to it on both grids.
  - By the reading fixed in the addendum, **GRADF's MNIST superiority (+4.85) depends on the reference.** The paper can state that **no system is superior to the hardened reference on either dataset**, and it reports the plain-reference superiority next to that.
- **The 7-rule set does not change d\* on MNIST:** d\*₇ = d\*_R = FLTrust (10/10 LOSO choices) on every grid. So B7_−s = B_R,−s, and GRADF's +4.85 against the plain reference stands against the best of the 7 plain rules too.
- **The headroom over D_7 is about twice the D_R headroom, with non-overlapping CIs, as on CIFAR-10:**

| cells | Ĥ_LOSO over D_R | Ĥ_LOSO over D_7 |
|---|---|---|
| 21 | 2.27 (1.97; 2.58) | 4.06 (3.64; 4.55) |
| 18 | 1.63 (1.31; 1.95) | 3.50 (3.09; 4.01) |
| 12 | 0.95 (0.66; 1.27) | 2.64 (2.26; 3.20) |

## Checks

- **Completeness:**
  - `mnist_fixed` at root 100: 1,470 rows, 7 rules.
  - `mnist_hardened`: 840 rows, 4 rules, 30/30 groups, no failures.
  - Systems: 210 rows each.
- **Reproduction:** before the freeze, the Random system run through the hardened runner's path reproduced the stored `mnist_rq3` runs exactly (max |diff| = 0).
- **Consistency:** against B_R,−s (plain), every system reproduces `RESULTS.md` on 18 cells: Random +1.39, attack-aware −0.91, GRADF +4.85, FedStrategist −0.88, AdaAggRL +1.51.

## A — the 7-rule set

**Mean over 21 cells, plain:** FLTrust 77.04, clustering 75.26, FedAvg 72.33, FedProx 71.98, trimmed mean 69.52, median 64.11, Krum 45.40.

**Cell winners among the 7 rules:**
- 21 cells: FedAvg 7, clustering 7, FLTrust 4, trimmed mean 3. So 14/21 lie outside D_R.
- 18 cells: clustering 6, FedAvg 5, FLTrust 4, trimmed mean 3. So 11/18 lie outside D_R.
- 12 cells: clustering 6, FLTrust 4, FedAvg 2. So 8/12 lie outside D_R.

**Systems against B7_−s (plain), equal to B_R,−s:**

| system | 18 cells | 12 cells |
|---|---|---|
| GRADF | **+4.85** (+3.76; +5.95), superiority | **+4.14** (+2.49; +5.80), superiority |
| AdaAggRL | +1.51 (+0.09; +2.93), superiority | +0.69 (−1.17; +2.55), no evidence |
| Random | +1.39 (+0.35; +2.43), superiority | −0.58 (−1.84; +0.67), no evidence |
| FedStrategist | −0.88 (−1.36; −0.40), inferiority | −0.74 (−1.35; −0.13), inferiority |
| Attack-aware | −0.91 (−1.67; −0.14), inferiority | −4.16 (−5.10; −3.22), inferiority |

The seed bootstrap that repeats the selection agrees with `describe` on 0 in every row.

## B — the hardened reference

**Mean over 21 cells, hardened:** FLTrust 82.45, trimmed mean 74.06, median 73.12, Krum 46.17. d\* hardened is FLTrust (10/10).

The hardened B_−s beats the plain B_−s by +6.08 p.p. (+5.49; +6.68) on 18 cells and by +5.86 (+5.10; +6.63) on 12.

**Systems against B_H,−s:**

| system | 18 cells | 12 cells |
|---|---|---|
| **GRADF** | **−1.23** (−2.18; −0.28), inferiority | **−1.72** (−3.14; −0.29), inferiority |
| AdaAggRL | −4.58 (−5.81; −3.34), inferiority | −5.17 (−6.70; −3.65), inferiority |
| Random | −4.69 (−5.50; −3.88), inferiority | −6.45 (−7.60; −5.29), inferiority |
| FedStrategist | −6.96 (−7.19; −6.74), inferiority | −6.60 (−6.98; −6.23), inferiority |
| Attack-aware | −6.99 (−7.40; −6.58), inferiority | −10.02 (−10.57; −9.47), inferiority |

- Wilcoxon p = 0.002 in every row.
- The bootstrap agrees with `describe` on 0 in every row.

## Reading (fixed in ADDENDUM_3)

- **Primary row: GRADF on 18 cells against the hardened reference.** Its CI95 lies below 0, not above. GRADF's MNIST superiority therefore depends on the reference.
  - Against the hardened d\*, used for GRADF and Random on CIFAR-10, no system is superior on MNIST either.
  - The paper reports the +4.85 against the plain d\* next to the −1.23 against the hardened one.
- **Against B7_−s:** unchanged from B_R,−s on MNIST, since d\* is FLTrust in both sets.
- **The reference-set issue of CIFAR-10 (X5) also holds on MNIST:**
  - the best rule of the 7 lies outside D_R in 14/21 cells;
  - Ĥ_LOSO over D_7 is 1.8 to 2.8 times the D_R value.

## Limits

- **Not blind:** a re-analysis on known seeds, with no Holm correction.
- **The hardened rules use GRADF's MNIST layer settings** (the defaults of the `mnist_rq3` Random system). The three rules outside D_R were not run hardened, as on CIFAR-10.
- **One execution;** GRADF's run-to-run variance is declared elsewhere.
