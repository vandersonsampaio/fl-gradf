# Addendum — tuning the global learning rate per dataset, by FLTrust's criterion

**Date:** 2026-11-03. **Status: frozen after the checks below passed (`checks.txt`) and before any run of the grid. The grid is not authorized yet: it needs its own go from the author.** Seeds outside the confirmatory block. This chooses a setting and confirms nothing.

**Decision of the author (2026-11-03):** use the criterion of FLTrust's paper and one factor per dataset, to be documented and explained in the paper. The operational choices below were recommended to the author and approved on the same day: tuning with no attack, by training error, one common factor for the three rules, root 100 only.

**This replaces `../cifar_factor_rules/`,** which the author stopped at 15:08 with 0 of 42 runs finished. Its reading was never applied and it has no result. It would have chosen by test accuracy under attack, which is not the paper's criterion.

## 1. What the source says

Cao et al., NDSS 2021, arXiv 2012.13995v3:

- **Criterion (Sec. VI-A).** "We set this combined learning rate on each dataset to achieve small training error rates and fast convergence."
- **Common to the methods.** FedAvg, Krum, Trim-mean, Median and FLTrust "all use the parameters τ, R_l, R_g, α, β, and b".
- **Horizon.** "We set the number of global iterations R_g such that the FL methods converge."
- **Table I.** α·β is 3×10⁻⁴ on MNIST (0.1 and 0.5), with R_g 2,000 and b 32, and 2×10⁻⁴ on CIFAR-10, with R_g 1,500 and b 64. R_l = 1 throughout.

The authors' code (`https://people.duke.edu/~zg70/code/fltrust.zip`, sha256 `07b68bfe246968e6247dd73e34121439a9c810c093fb6b6f17d22d8ba0f3028f`, not stored in this repository):

- It implements FashionMNIST only. Any other dataset raises `NotImplementedError`, so **it does not show how MNIST or CIFAR-10 were run.**
- It has no tuning routine. The default `lr = 0.006` matches Table I for FashionMNIST.
- It applies a single rate, `w ← w − lr·g`.
- It computes g₀ on the whole root. The `minibatch` it draws is never used, unlike the batch in the paper's algorithm.

**What the source leaves open, and what this addendum therefore decides:** thresholds for "small" and "fast"; whether the tuning runs with or without attack; which rule it is read on.

**The paper's values do not transfer.** Its rate assumes R_l = 1. The clients here train by epochs with `local_lr` 0.01, and α is a separate multiplier, fixed at 1 until now (`../PLAN.md`, `../cifar_factor_sweep/ADDENDUM.md`). The paper's horizons are also not used: R is 15 on MNIST and 110 on CIFAR-10, the values of the existing runs and of the campaign's budget.

## 2. Runs

Per dataset: {FedAvg, trimmed mean, FLTrust} × factors {1, 1/3, 1/5, 1/10} × seeds 0–2, **no attack,** α 0.05, root 100, `server_steps = median`. **36 runs per dataset, 72 in total,** MNIST first.

| dataset | model | R | local epochs |
|---|---|---|---|
| MNIST | logistic regression | 15 | 5 |
| CIFAR-10 | CNN | 110 | 2 |

**Training error** is 1 minus the global model's accuracy, after every round, on a fixed sample of 5,000 examples from the clients' pooled training data, drawn from the seed. The test accuracy is logged as before and **takes no part in the choice.**

`scripts/p1/lr_tuning.py`, `scripts/p1/run_lr_tuning.sh`, `scripts/p1/analyze_lr_tuning.py`. No change to `src/`.

## 3. Checks before freezing (`checks.txt`, all PASS)

1. CIFAR-10, FedAvg, no attack, factor 1, with the training log on, equals E0b exactly over 2 rounds.
2. The training log leaves the run unchanged: test accuracy is identical with it on and off, FLTrust, both datasets.
3. The logged value equals the training accuracy of the learner's own global model.
4. The first aggregate at factor 1/5 has 1/5 of the norm at factor 1, each rule, both datasets.
5. Seeds 42–51 are refused.

The choice logic in `analyze_lr_tuning.py` was also exercised on a throwaway fixture, not kept, covering both branches of the rule below.

## 4. Choice rule, for each dataset separately

For each rule and factor, E is the training error averaged over the last 5 rounds, then over seeds. The gap of a factor for a rule is E minus that rule's lowest E across factors.

1. A factor is **admissible** if its gap is at most **1.0 p.p. in all three rules.**
2. **The chosen factor is the largest admissible one.** A larger factor converges faster and departs less from what has been run.
3. **If none is admissible,** the chosen factor is the one with the smallest largest gap, with ties going to the larger factor, and this is reported as a failure of the margin.

Convergence speed enters through step 2. Reported alongside, with no test and no part in the choice: test accuracy on the last 5 rounds; the first round within 1.0 p.p. of the final error; the mean absolute change of the training error per round, second half of the run; errors per seed.

## 5. After the choice

- **Verification.** The chosen factor is then run under attack, in a separate addendum, as a check that is reported. The author decides whether to adopt the factor, and any departure from this rule is stated in the paper.
- **The factor is not re-chosen from attacked results.** If tuning without attack returns a factor at which FLTrust is unstable under attack, that instability is a result to report.
- **MNIST.** A factor other than 1 requires rerunning `../mnist_fixed` and `../mnist_rq3` before the MNIST numbers can be used, and changes those numbers throughout the paper.
- **Pre-registration.** The factor of each dataset is stated there. Seeds 42–51 are not used to choose it.

## 6. What was already seen

This tuning is not blind on CIFAR-10. FLTrust's test accuracy and stability under attack at all four factors are known (`../cifar_factor_sweep/RESULTS.md`), as is FLTrust at factors 1 and 1/3 on MNIST under attack (`../gradf_layer2/step/RESULTS.md`). No result without attack at reduced factors, and no result for the trimmed mean or FedAvg at reduced factors, has been seen. The rule above does not use the seen quantities, but it was written after them.

## 7. Limits fixed in advance

- One α, the most heterogeneous, 3 seeds. The paper uses one rate across its two MNIST heterogeneity levels, which is some support for one rate per dataset, not evidence for this setting.
- With 3 seeds and a sample of 5,000, a 1.0 p.p. margin may be close to the noise.
- Fixed horizons penalize small factors, unlike the paper's choice of R_g until convergence.
- Root 100 only: FLTrust at root 2,500 is not tuned, although E1 uses it.
- Median, Krum, FedProx, clustering and the adaptive systems are not tuned. They would take the same factor.

## 8. Cost

CPU, 6 processes. MNIST: 0.7 s per round alone, about 2 to 5 minutes for all 36 runs. CIFAR-10: 10.1 s per round alone, and 24.9 s measured in the sweep at 6 processes, so about 46 minutes per wave and **about 4.6 h** over 6 waves. W1 is paused at this date. If the author resumes it during the grid, the overlap is reported.

## Code hashes (sha256)

```
e2227ebcef74a42fa0947c175d992141bae5578551025171d09f01d6e5e28a50  scripts/p1/lr_tuning.py
1c470aa6b4c583e34c7d183291b068db37cda882828b4b0b3c34d25159204c40  scripts/p1/run_lr_tuning.sh
c9187fc492b2d81a0f2b51faf733bf43ba2d1bd083b87f527740471f4a7e952c  scripts/p1/analyze_lr_tuning.py
```
