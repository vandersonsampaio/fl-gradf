# Addendum — the global learning rate in FLTrust, and a factor sweep on CIFAR-10

**Date:** 2026-11-03. **Status: frozen before the grid, after the checks below passed (`checks.txt`).** **Authorized by the author on 2026-11-03.** It runs while the W1 rerun is paused, so nothing overlaps. Pilot on seeds outside the confirmatory block, descriptive: it confirms nothing.

**Why:** a factor of 1/3 on FLTrust's aggregate changes the sign of the root contrast on block C and leaves root 100 unstable (`../cifar_step_pilot/RESULTS.md`).

## 1. What the paper says

Read in Cao et al., NDSS 2021 (arXiv 2012.13995v3):

- **The method has a global learning rate.** Eq. 5 and Algorithm 2: w ← w + α·g, "where α is the global learning rate".
- **Normalization to ‖g₀‖ is part of the method** (Eq. 3), and small updates are enlarged too.
- **Server and clients use the same R_l, β and b** in Algorithm 2.
- **α is not given alone.** Sec. VI-A sets R_l = 1 and treats α·β "as a single learning rate", set "on each dataset to achieve small training error rates and fast convergence". Table I: 3×10⁻⁴ for MNIST-0.1 and MNIST-0.5, 6×10⁻³ for Fashion-MNIST, 2×10⁻⁴ for CIFAR-10, 3×10⁻³ for HAR, 3×10⁻⁴ for CH-MNIST.
- **The rate is common to all methods compared.** Sec. VI-A: FLTrust, FedAvg, Krum, Trim-mean and Median "all use the parameters τ, R_l, R_g, α, β, and b".

**Consequences for P1:**

1. A factor on the aggregate is not an addition to FLTrust. It is α, which this code fixes at 1.
2. The paper's values cannot be transferred. Its combined rate assumes R_l = 1, while the clients here train by epochs with `local_lr` 0.01, as `../PLAN.md` already states.
3. **A factor applied to FLTrust alone departs from the paper's design,** in which all rules share the rate. E1's contrast is FLTrust − trimmed mean, so a factor chosen for FLTrust alone would change one side only.

## 2. Sweep

**Runs:** the block of `../cifar_step_pilot/ADDENDUM.md`: FLTrust, `server_steps = median`, α 0.05 × {`sign_flipping`, `trim_attack`} × root {100, 2,500} × seeds 0–2, R = 110, at **factors 1/5 and 1/10. 24 runs.** Factors 1 and 1/3 are reused from block C and the pilot. No change to `src/`.

**Checks before freezing:** the first round's step along g₀ is 1/5 and 1/10 of factor 1's (`trim_attack`, seed 0, root 100).

## 3. Reading rule

Same quantities as the pilot: stability is the mean absolute change between consecutive rounds, rounds 51–110; accuracy is the trailing mean of rounds 106–110; per attack × root, seed means.

For each of 1/3, 1/5 and 1/10:
- **Stabilizes** if its stability is at most half of factor 1's in all 4 cells; **does not stabilize** if in none; otherwise **partly**.

**Best for FLTrust:** among the factors that stabilize, the highest accuracy averaged over the 4 cells; within 1.0 p.p., the larger factor, which departs less and learns faster. If none stabilizes, this is reported and no best is named.

Reported alongside, with no test: accuracy per cell and per seed; the root contrast per factor, with its sign; FedAvg at factor 1.

**This does not choose the campaign's factor.** By Sec. 1, that needs the trimmed mean and FedAvg at the same factors. Those runs are not authorized here and need their own addendum.

**Limits fixed in advance:** 2 cells of 21, one α, 3 seeds. Smaller factors learn more slowly, and R = 110 is the campaign's horizon, so the reading is about that horizon and not about convergence.

## 4. Cost

Measured in the pilot: about 15 s per round at 4 processes. 6 processes, 4 waves of 110 rounds: **about 2 h.**

## Code hashes (sha256)

```
26ba11636932a7aa2789d03e13c6bf745c283ac2bc096a08072f63f0634a346d  scripts/p1/cifar_factor_sweep.py
c5106d462e281b75544592fce239ac9e08939656170a6c5f13296e87b728701f  scripts/p1/run_cifar_factor_sweep.sh
48413cec84e2edc418030f4efe3f098225f224c12fecbfa470911236ed502fd0  scripts/p1/cifar_step_pilot.py
```
