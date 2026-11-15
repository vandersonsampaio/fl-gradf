# Gate P-2 — is E4b needed?

**Date:** 2026-11-15. **Decision (the author's): E4b is not run.** It is declared as future work in the paper's limitations section. The four questions of the gate are answered by the confirmatory campaign (`RESULTS.md`), X1 (`X1_RESULTS.md`), X5 (`X5_RESULTS.md`) and E4a (`results/p1_e4a/RESULT.md`, on branch `p1/segunda-submissao`). The gate's own rule is: if the answers are strong enough, do not enlarge the paper only to add experiments.

## The four questions

**1. Does the root effect seen on MNIST also appear on CIFAR-10?** Yes, and it depends on the global rate.
- At factor 1/3 (confirmatory), FLTrust's advantage over the trimmed mean grows with the root: D = +8.40 p.p. (+7.89; +8.90), superiority.
- At factor 1 (sensitivity), it shrinks: D = −1.61 p.p. (−2.17; −1.04), inferiority.
- By the rule fixed in the preregistration, the root effect is reported as dependent on the global rate, with no general claim about the root.

**2. Is there measurable headroom outside the sample on CIFAR-10?** Yes, at both factors.

| factor | Ĥ_LOSO over D_R, 21 cells | Ĥ_LOSO over D_R, 18 cells without `fltrust_aligned` | source |
|---|---|---|---|
| 1/3 | 3.86 (3.49; 4.17) | 1.36 (1.03; 1.66) | confirmatory |
| 1 | 0.95 (0.83; 1.11) | 1.11 (0.97; 1.30) | X1 |

Over the 7-rule action set the headroom is larger: 6.53 (5.89; 6.97) and 3.55 (2.85; 4.03) at factor 1/3 (X5, post-hoc).

**3. Do the systems differ informatively in how much of it they capture?** Yes.
- **Factor 1/3 (confirmatory):**
  - every system with a learned or oracle selector beats Random, except AdaAggRL;
  - no system beats the best fixed rule (FLTrust);
  - GRADF and FedStrategist are indistinguishable from it;
  - Random, the attack-aware reference and AdaAggRL are worse.
- **Factor 1:** apparent capture depends on the reference.
  - Against the plain d\*_R, FedStrategist gains +6.38 p.p., and Random, attack-aware and GRADF gain +1.4 to +1.8 (X1).
  - Against the best of the 7 rules, FedStrategist is equivalent (−0.22 p.p., TOST p = 0.014). The others are worse, by −4.8 to −5.2 (X5).
- These differences show the fragility the paper's framework is built to expose: the global rate, a plain or hardened reference, and the reference set against the action space.

**4. Do MNIST, CIFAR-10 and E4a support the thesis of the two-stage evaluation framework?** Yes.
- **E4a** detects the constructed temporal headroom on MNIST: H_T = +1.65 p.p. (+0.82; +2.48) at α = 0.1 and +0.75 (+0.73; +0.77) at α = 0.5. This validates the metric; it makes no confirmatory claim.
- **CIFAR-10** confirms positive cross-fitted headroom.
- **CIFAR-10 also shows that no adaptive system captures it** against an appropriate reference.

## Consequences

- E4b (temporal headroom with a development/confirmation seed split) moves to future work and is stated in the limitations section.
- No new jobs are planned for P1. What remains is writing:
  - abstract, contributions and title;
  - RQ3 restricted to the factor (ADDENDUM_2);
  - the D_7 values shown next to D_R (ADDENDUM_3 Sec. 4).
