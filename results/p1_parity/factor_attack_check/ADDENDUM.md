# Addendum — verification under attack of the tuned factors on CIFAR-10

**Date:** 2026-11-03. **Status: frozen after the checks below passed (`checks.txt`) and before the grid.** **Authorized by the author on 2026-11-03.** It runs alongside W1, on CPU. Seeds outside the confirmatory block. It reports and confirms nothing.

**Decision of the author (2026-11-03), after `../lr_tuning/RESULTS.md`:**
1. The factors of the rule are adopted, despite the margin failing on both datasets: **factor 1 on MNIST and factor 1/3 on CIFAR-10, common to all rules.**
2. **CIFAR-10 at factor 1 is a pre-registered sensitivity analysis,** with E1 at both factors, because the two factors are nearly tied and FLTrust − trimmed mean changes sign between them. The author accepts the extra machine hours this adds above the budget of 236 h.

**Why this run:** `../lr_tuning/ADDENDUM.md`, Sec. 5, provides for the chosen factor to be run under attack as a reported check. Sec. 5 also holds here: **the factor is not re-chosen from these results.**

## Runs

α 0.05 × {`sign_flipping`, `trim_attack`} × seeds 0–2, R = 110, root 100, `server_steps = median`, as in block C:

| rule | factors | runs |
|---|---|---|
| trimmed mean | 1, 1/3 | 12 |
| FedAvg | 1/3 | 6 |

**18 runs.** The run itself is `scripts/p1/cifar_factor_rules.py::run`, unchanged, called from `scripts/p1/factor_attack_check.py`. No change to `src/`.

**Reused, not rerun:** FedAvg at factor 1 from E0b; FLTrust at factor 1 from block C and at factor 1/3 from the pilot, roots 100 and 2,500. MNIST under attack at factor 1 already exists (`../mnist_fixed`), and it is not repeated.

**Not reused:** the `.log` files of the stopped grid `../cifar_factor_rules/`, which has no result.

## Checks before freezing (`checks.txt`, all PASS)

1. FedAvg at factor 1 equals E0b exactly over 2 rounds, under both attacks.
2. The trimmed mean's first aggregate at factor 1/3 has 1/3 of the norm at factor 1, under both attacks.
3. Seeds 42–51 are refused.

The analysis was exercised on a throwaway fixture for the new runs, not kept.

## What is reported

Accuracy is the mean of rounds 106–110; stability is the mean absolute change between rounds, 51–110; seed means. With no test (3 seeds):

- Accuracy and stability per rule, attack and factor.
- **A rule pays** if, over the 2 attacks, its accuracy at factor 1/3 is more than 5.0 p.p. below its accuracy at factor 1. The threshold is that of `../cifar_factor_rules/ADDENDUM.md`.
- FLTrust − trimmed mean per root and factor, and D = V(2,500) − V(100), with its sign.
- The best rule per attack and factor.

**A rule that pays is reported and stated in the paper. It does not change the factor.**

## Limits fixed in advance

- 2 cells of 21, one α, 3 seeds.
- Median, Krum, FedProx, clustering and the adaptive systems are not run here.
- Seeds 0–2 were used in the tuning without attack, in the pilot and in the sweep.

## Cost

About 60 minutes per wave of 6 alongside W1, as measured at the end of the tuning. 3 waves: **about 3 h.**

## Code hashes (sha256)

```
bd9bcbea033e02f067bd96b3e1390a73807a854d6e5e5a3160dfa03fa6ca1913  scripts/p1/factor_attack_check.py
10de177ea9c6f4f77f0f732f3708de70e10436c8220ce2b93162f5570a88dacd  scripts/p1/run_factor_attack_check.sh
597489d8dd04a5e3a59d979ee229c2eb3b8bb03faff40d7e2f5fd32a7c217eca  scripts/p1/analyze_factor_attack_check.py
e0911c4b22995ed09b01f6d559f70624b453d526509df6533dfa083c5b75f8ff  scripts/p1/cifar_factor_rules.py
```
