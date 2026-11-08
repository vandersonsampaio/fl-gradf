# Addendum — the trimmed mean and FedAvg at the same factors on CIFAR-10

**Date:** 2026-11-03. **Status: frozen before the grid, after the checks below passed (`checks.txt`).** **Authorized by the author on 2026-11-03,** to run after the FLTrust sweep. It runs while the W1 rerun is paused, so nothing overlaps. Pilot on seeds outside the confirmatory block, descriptive: it confirms nothing.

**Why:** in FLTrust's paper the global learning rate α is common to all methods compared (`../cifar_factor_sweep/ADDENDUM.md`, Sec. 1). The FLTrust sweep cannot choose the campaign's factor alone.

**Question:** which common α suits the rules that E1 and E3 rest on, and does any rule pay for it?

## Runs

α 0.05 × {`sign_flipping`, `trim_attack`} × seeds 0–2 × R = 110, root 100, `server_steps = median`, as in block C:

| rule | factors | runs |
|---|---|---|
| trimmed mean | 1, 1/3, 1/5, 1/10 | 24 |
| FedAvg | 1/3, 1/5, 1/10 | 18 |

**42 runs.** The author's authorization named 36. The other 6 are the trimmed mean at factor 1, because the existing runs (`../../p1_cifar10/e0/separation`) stop at 50 rounds. FedAvg at factor 1 comes from E0b. FLTrust at 1, 1/3, 1/5 and 1/10 comes from block C, the pilot and the sweep. No root contrast is run for these two rules, since they do not read the root under these attacks.

`scripts/p1/cifar_factor_rules.py`, no change to `src/`.

## Checks before freezing

1. FedAvg at factor 1 equals E0b exactly over 2 rounds, under both attacks (seed 0). This also shows that `server_steps` does not move a rule that does not use g₀.
2. The trimmed mean's first aggregate at factor 1/5 has 1/5 of the norm at factor 1.

## Reading rule

Accuracy is the trailing mean of rounds 106–110; stability as in the pilot, rounds 51–110; seed means.

**Score of a factor:** the mean accuracy over the 2 attacks of FLTrust (mean of roots 100 and 2,500), the trimmed mean and FedAvg, each weighing one third.

**Proposed common α:** the factor with the highest score; within 1.0 p.p., the larger factor, which departs less and learns faster.

**A rule pays** if at the proposed α its accuracy is more than 5.0 p.p. below its own accuracy at factor 1. Each such rule is reported, and the choice between a common α and rule-specific rates goes to the author.

Reported alongside, with no test (3 seeds): per factor, FLTrust − trimmed mean per root; the E1 statistic D = V(2,500) − V(100), which under the invariant equals FLTrust's root contrast, since the trimmed mean's run is shared; the best rule per attack; stability per rule.

**This proposes and does not decide.** The factor in the campaign is the author's decision. It must be stated in the pre-registration, and seeds 42–51 are not used to choose it.

**Limits fixed in advance:**
- 2 cells of 21, one α of heterogeneity, 3 seeds.
- Median, Krum, FedProx, clustering and the adaptive systems are not run at these factors.
- The score includes FLTrust, whose instability drove this search, so it is not independent of the problem it answers.
- R = 110 is the campaign's horizon, so smaller factors are read at that horizon and not at convergence.

## Cost

Measured: about 15 s per round at 4 processes, about 27 minutes per run. 6 processes, 7 waves: **about 3 to 3.5 h.**

## Code hashes (sha256)

```
e0911c4b22995ed09b01f6d559f70624b453d526509df6533dfa083c5b75f8ff  scripts/p1/cifar_factor_rules.py
88b3911fe8aa89c44a516c2a9147a325fc8ce955311ca960bf684a878a556f86  scripts/p1/run_cifar_factor_rules.sh
```
