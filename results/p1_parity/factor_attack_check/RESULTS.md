# Result — verification under attack of the tuned factors on CIFAR-10

**Date:** 2026-11-03. **Status:** reported check on seeds 0–2, outside the confirmatory block, with no test. **It does not re-choose the factor** (`ADDENDUM.md`). Grid: 18/18, 0 failures, from 20:45 to 23:07. W1 overlapped until the author paused it at 22:20.

Accuracy is the mean of rounds 106–110, seed means. Stability is the mean absolute change between rounds, 51–110.

## Accuracy and stability

| attack | rule | accuracy, factor 1 | factor 1/3 | stability, factor 1 | factor 1/3 |
|---|---|---|---|---|---|
| `sign_flipping` | FedAvg | 37.40 | 29.85 | 1.29 | 0.35 |
| | trimmed mean | 24.64 | 17.98 | 0.63 | 0.15 |
| | FLTrust, root 100 | 23.11 | 35.53 | 11.19 | 8.17 |
| | FLTrust, root 2,500 | 18.25 | 40.10 | 4.05 | 1.89 |
| `trim_attack` | FedAvg | 26.03 | 20.07 | 0.85 | 0.23 |
| | trimmed mean | 17.83 | 14.95 | 0.37 | 0.16 |
| | FLTrust, root 100 | 25.18 | 35.03 | 6.89 | 9.88 |
| | FLTrust, root 2,500 | 18.26 | 38.58 | 7.60 | 2.27 |

## Which rules pay at factor 1/3

Threshold: more than 5.0 p.p. below factor 1, over the 2 attacks.

| rule | factor 1 | factor 1/3 | change | pays |
|---|---|---|---|---|
| FedAvg | 31.72 | 24.96 | −6.76 | **yes** |
| trimmed mean | 21.23 | 16.46 | −4.77 | no, close to the threshold |
| FLTrust, root 100 | 24.14 | 35.28 | +11.14 | no |
| FLTrust, root 2,500 | 18.26 | 39.34 | +21.09 | no |

FedAvg's payment is reported and stated in the paper. It does not change the factor.

## FLTrust against the trimmed mean

| attack | factor | V, root 100 | V, root 2,500 | D |
|---|---|---|---|---|
| `sign_flipping` | 1 | −1.53 | −6.39 | −4.86 |
| | 1/3 | +17.56 | +22.13 | +4.57 |
| `trim_attack` | 1 | +7.35 | +0.43 | −6.92 |
| | 1/3 | +20.09 | +23.64 | +3.55 |

D has the same sign in all 6 pairs of seed and attack at each factor: negative at factor 1, positive at factor 1/3.

**Best rule per attack.** At factor 1, FedAvg on both attacks. At factor 1/3, FLTrust at root 2,500 on both attacks.

## Reading

**The adopted factor favors FLTrust strongly under attack.** Moving to 1/3 raises FLTrust by 11 to 21 p.p. and lowers the other two rules by 5 to 7 p.p. In these 2 cells it changes which rule is best, the sign of the root effect and the size of FLTrust's lead. This extends the warning of `../lr_tuning/RESULTS.md`, where the same dependence appeared with no attack. Any statement that FLTrust dominates at factor 1/3 must be read together with factor 1, which is why E1 runs at both.

**The cause is not established.** FedAvg and the trimmed mean learn more slowly at the smaller step within R = 110, and FLTrust steps less erratically. Longer horizons were not run.

**Limits:** 2 cells of 21, one α, 3 seeds, seeds already used in the tuning and the pilots. At both factors FLTrust at root 100 stays unstable (8.17 and 9.88 at factor 1/3).

## Value for the pre-registration

σ_E0^E3, per-seed mean over the 2 cells of FLTrust at root 100 minus FedAvg, factor 1/3, rounds 106–110: 8.37, 10.82, 11.79; **SD 1.76, so 0.5σ = 0.88 p.p.** This is narrower than the primary margin of 1.0 p.p. It rests on 2 cells, 3 seeds and fixed rules only.

## Outputs

`analysis/analysis.txt`, `analysis/runs.csv`, `raw/`.
