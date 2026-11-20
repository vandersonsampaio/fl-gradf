# Result — pilot of a smaller FLTrust step on CIFAR-10

**Date:** 2026-11-03. **Status: complete, a pilot on seeds outside the confirmatory block; it confirms nothing.** Plan: `ADDENDUM.md`, frozen before the grid. 12/12 jobs, 0 failures, 10:10–11:33, 4 processes, with the W1 rerun paused. Analysis: `scripts/p1/analyze_cifar_step_pilot.py`, output in `analysis/`.

**In short:** by the frozen rule, the factor of 1/3 stabilizes FLTrust only **partly**. The larger finding is outside that rule: accuracy rises by 16 p.p. on average, and **the root contrast changes sign**. At factor 1 the larger root is worse, and at factor 1/3 it is better. Since E1's hypothesis is about that contrast, the campaign should not start before the factor is settled.

## 1. Frozen reading: stability

Mean absolute change between consecutive rounds, rounds 51–110, seed means, p.p.:

| attack | root | factor 1 | factor 1/3 | ratio |
|---|---|---|---|---|
| `sign_flipping` | 100 | 11.19 | 8.17 | 0.73 |
| | 2,500 | 4.05 | 1.89 | 0.47 |
| `trim_attack` | 100 | 6.89 | 9.88 | 1.43 |
| | 2,500 | 7.60 | 2.27 | 0.30 |

**Reading: partly** (2 of 4 cells at most half). Root 100 stays unstable at this factor, and under `trim_attack` it swings more. FedAvg swings 1.29 and 0.85 in the same cells.

## 2. Accuracy (no test, 3 seeds)

Trailing mean of rounds 106–110, seed means, %:

| attack | root | factor 1 | factor 1/3 | difference | FedAvg |
|---|---|---|---|---|---|
| `sign_flipping` | 100 | 23.11 | 35.53 | +12.42 | 37.40 |
| | 2,500 | 18.25 | 40.10 | +21.85 | |
| `trim_attack` | 100 | 25.18 | 35.03 | +9.85 | 26.03 |
| | 2,500 | 18.26 | 38.58 | +20.32 | |

- **Every one of the 12 runs is higher** at factor 1/3, by +7.79 to +24.29 p.p. (mean +16.11).
- **At factor 1, root 2,500 barely learns,** at about 18% in every run.
- **FLTrust goes from below FedAvg to above it in 3 of 4 cells.** Under `sign_flipping` at root 100 it is still 1.87 p.p. below.
- **The curves are still rising or level at R** at factor 1/3. No gain after R is available for this arm.

## 3. The root contrast changes sign

Root 2,500 − root 100, trailing mean, p.p.:

| attack | factor 1 | factor 1/3 |
|---|---|---|
| `sign_flipping` | −4.86 | **+4.57** |
| `trim_attack` | −6.92 | **+3.55** |

The sign holds in all 6 attack × seed pairs at each factor: −8.15 to −4.66 at factor 1, +1.32 to +10.04 at factor 1/3. E1's confirmatory hypothesis is that FLTrust's advantage is larger with the larger root. Under the same runs it would look refuted at factor 1 and supported at factor 1/3. Only FLTrust's side moves here, as the comparison with the trimmed mean was not run in this pilot.

## 4. What is not established

- **The factor.** 1/3 came from MNIST, and root 100 is still unstable with it. It is not known to be a good value on the CNN.
- **Other rules.** Only FLTrust was scaled. Under parity FLTrust rescales each update to ‖g₀‖, while the other rules average the updates as they are, which could explain why it needs a smaller step. That explanation was not tested, and whether other rules gain from a factor is unknown.
- **Other cells.** 2 cells of 21, α 0.05 only, 3 seeds.
- **Faithfulness.** FLTrust's global learning rate in Cao et al. has not been checked yet, and it bears on how the factor should be presented: a parameter of the method, or a tuning added here.

## 5. Proposals for the author (nothing changes on its own)

1. **Do not start E1 at factor 1 alone.** Its main hypothesis depends on this choice.
2. **Check the global learning rate in the FLTrust paper** before choosing how to present it.
3. **Choose the factor outside the confirmatory block,** on seeds 0–2, with a rule frozen first, including other values such as 1/5 and 1/10 for root 100's instability. Seeds 42–51 must not be used for this.
4. **Then decide the campaign's arms:** which factor is primary, and whether factor 1 stays as sensitivity. Each FLTrust arm over both roots and all 21 cells costs about 26.6 h, which does not fit level 1 unless something else is dropped, such as the native-epoch arm of the E1 factorial.

## Outputs

`analysis/analysis.txt`, `runs.csv`, `curves.csv`; `raw/`.
