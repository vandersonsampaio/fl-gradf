# P1 experiments: frozen experimental plan

**Date:** 2026-10-29.
**Scope:** experiments only, for the revision of P1. P1 evaluates when adaptive Byzantine-robust aggregation has measurable headroom over fixed rules and whether evaluated systems capture it. It covers root-data dependence, cross-condition and temporal headroom, and capture, on MNIST (existing data) and CIFAR-10 (new).
**Status:** design frozen. Any later change enters only as a declared deviation, with its reason, in an addendum (`ADDENDUM*.md`) or in the E0 pre-registration.
**Freezing context, stated so nothing is hidden:**
- Gate P-1 was already under way when this plan was frozen.
  - The MNIST audit and the MNIST application of the E1–E3 analyses (`results/p1_audit/`) were done on data that existed and had been published.
  - A seeded MNIST rerun of the fixed rules (`results/p1_invariance/`) was launched on 2026-10-29 at 09:48.
- Everything about MNIST in this plan is therefore re-analysis, not confirmation.
- No CIFAR-10 run of E0–E4 exists.

---

## 1. Boundary of the experiments

- AdaAggRL is evaluated as a black-box system that may or may not capture headroom.
- It is **not** opened, decomposed or ablated here. Isolating detector × signal × policy × action space belongs to the companion audit study (P2), so no such experiment is run under this plan.
- A change in AdaAggRL's behavior between datasets is read only as evidence that system-level behavior is regime dependent. No internal component is credited with the change: reconstruction, TD3, MMD and the extractor are all left unattributed.
- Out of scope: other Byzantine fractions, a second continuous-weighting mechanism, and privacy analysis of gradient inversion. Computational cost is measured only as AdaAggRL's per-round cost on CIFAR-10, in E0.

---

## 2. Notation (frozen for tables and code)

Let D\_R = {FLTrust, median, trimmed mean, Krum} be the reference set and C the set of cells of the grid in use.

**Population quantities** (expectations over the seed distribution):

| Symbol | Definition | Reading |
|---|---|---|
| acc(d, c) | expected accuracy of rule d in cell c | |
| O(c) | max\_{d ∈ D\_R} acc(d, c) | cell-wise fixed optimum |
| d\* | argmax\_{d ∈ D\_R} E\_c[acc(d, c)] | global best fixed rule |
| B(c) | acc(d\*, c) | the global rule evaluated in cell c |
| H(c) | O(c) − B(c) | cell headroom |
| H | E\_c[H(c)] | global headroom |
| R\_s(c) | [A\_sys(c) − B(c)] / [O(c) − B(c)], if O(c) > B(c) | fraction captured by a system |

**Cross-fitted (leave-one-seed-out) estimators**, for evaluation seed s:

| Symbol | Definition |
|---|---|
| r\*\_{−s}(c) | rule of D\_R with the highest mean accuracy in cell c over seeds ≠ s |
| d\*\_{−s} | rule of D\_R with the highest global mean accuracy over seeds ≠ s |
| O\_{−s}(c, s) | acc(r\*\_{−s}(c), c, s) |
| B\_{−s}(c, s) | acc(d\*\_{−s}, c, s) |
| Ĥ\_s | E\_c[O\_{−s}(c, s) − B\_{−s}(c, s)] |
| Ĥ\_LOSO | mean of Ĥ\_s over seeds |

In-sample estimators use the same symbols without −s and are labelled **in-sample estimates**. They are not labelled upper bounds: O(c) is biased upward by selection, but so is B(c) (see Sec. 9).

Three quantities are always kept apart, and none is called an "oracle" without qualification: O(c), the attack-aware selector, and the phase-wise greedy reference of E4.

---

## 3. Statistical protocol

- **Pre-registration.** Confirmatory hypotheses are declared before running, in a pre-registration frozen by hash in the repository (the output of E0, Sec. 5).
- **Unit of analysis: the seed.**
  - In aggregate hypotheses, each seed contributes one observation: its mean over the cells of the grid in use.
  - Cells × seeds are never treated as independent observations.
- **Tests.**
  - Seed-paired Wilcoxon signed-rank test, primary.
  - Paired t-test, complementary.
  - 95% CI of the mean per-seed difference from the t distribution.
  - Cohen's d\_z.
- **Multiplicity:** Holm correction within each family of hypotheses.
- **Four outcome categories:**

| Category | Criterion |
|---|---|
| superiority | the whole 95% CI is above zero |
| inferiority | the whole 95% CI is below zero |
| equivalence | TOST: 90% CI inside the margin |
| no evidence | any other case |

- **Equivalence margin (frozen):**
  - Primary margin δ = 1.0 p.p. It represents practical relevance and does not grow with the experiment's noise.
  - Sensitivity: TOST repeated with δ = 0.5·σ\_E0, where σ\_E0 is the across-seed SD estimated in E0 on seeds outside the confirmatory block.
  - If the 1.0 p.p. TOST does not close on CIFAR-10 because of noise, the result is **no evidence**, not equivalence. The primary margin is never relaxed.
- **Confirmatory vs. descriptive.** Confirmatory and exploratory/descriptive analyses are kept apart. Per-cell win and capture counts are descriptive.
- **Population quantity vs. estimator.**
  - Headroom and capture are population quantities (Sec. 2).
  - When an estimate both picks a winning rule (O(c), d\*, phase winner) and measures that rule's advantage, the picking and the measuring use different seeds: LOSO in E2–E3, a development/confirmation split in E4b.
  - This is a choice of estimator, not part of the definition.
- **Inference with LOSO.**
  - Point estimate: cross-fitted by leave-one-seed-out.
  - Uncertainty: seed-level bootstrap that redoes the whole LOSO selection and evaluation in every resample. When a resample contains copies of the held-out seed, those copies are also left out of the selection set.
  - The bootstrap gives CIs, not a precise p-value (n = 10 seeds).
  - No claim is made that LOSO is unbiased or that the Ĥ\_s are independent: selection sets share 8 of 9 seeds.
- **Reproducibility.**
  - Every new run seeds NumPy's global RNG per run from (seed, cell, rule/system), never from the root size.
  - Server-side fits on root data use their own RNG stream (`--seed_rng` in `exp9_dominance_grid.py`; the same mechanism is required in every new runner).
  - Under the same configuration, a run must reproduce bit for bit on CPU.
- **Partition.** The client partition is fixed per α on disk and does not change with the seed. The seed changes the root subsample, the attack noise and the selectors' RNGs. All inference is therefore conditional on one partition per α, and is reported as such.
- **Shared code.** The same analysis code (`scripts/p1/inference.py`) is used for MNIST and CIFAR-10.

---

## 4. Gate P-1: MNIST audit before any CIFAR-10 run

P-1 establishes a single pipeline: raw per-seed data → analysis script → tables, figures and numbers. It is complete when all of the following hold:

1. Every reported MNIST number is reproduced from the raw per-seed data by a script.
2. Tables and figure data are regenerable from versioned raw files. The raw inputs currently git-ignored under `results/tables/` are added explicitly.
3. The E1–E3 analyses of Sec. 5, including LOSO and the bootstrap, are implemented and run on MNIST.
4. The trimmed-mean root invariant is verified, as follows.

**Invariant.** The fixed trimmed mean uses no server data. Hence, for every cell c and seed s:

> Acc\_TM(c, s, r = 100) = Acc\_TM(c, s, r = 3000) (MNIST) and Acc\_TM(c, s, r = 100) = Acc\_TM(c, s, r = 2500) (CIFAR-10)

The check has three parts:
- **Code check:** the fixed trimmed mean does not pass through the accuracy-based detector or the hardening pipeline's validation, both of which use the root.
- **Exact check:** with seeded runs (Sec. 3), the equality must hold bit for bit on MNIST (P-1) and on CIFAR-10 (first output of E1). It is checked for median and Krum too.
- **Known exception:** under the `fltrust_aligned` attack, the attacker submits a copy of the server's root update, so every rule legitimately depends on the root there. This cell type is reported separately and excluded from the confirmatory grids.

**If the invariant fails** anywhere else, it is treated as a possible experimental leak. E1 does not start until the cause is found and fixed; changing the comparator is not enough.

---

## 5. Experiments

### 5.0 Common configuration (CIFAR-10)

| Item | Decision | Note |
|---|---|---|
| Dataset | CIFAR-10 | |
| Model | small CNN (2–3 conv layers) | not ResNet-20; keeps cost down and the flattened-vector interface of the 7 rules |
| Clients | N = 10, 20% Byzantine | as on MNIST |
| Heterogeneity | α ∈ {0.5, 0.1, 0.05} | **to be confirmed in E0** |
| Canonical root | 100 examples | FLTrust's reference configuration |
| Enlarged root | 2,500 examples (5% of the training pool) | matches the 5% used on MNIST |
| Rounds | **fixed in E0** | |
| Local epochs | **fixed in E0** | the per-round cost scales with it |
| Seeds | **fixed in the E0 pre-registration** | proposal: 42–51 for E1–E3 (paired across experiments); a new, never-used block for E4b; no collision with blocks used elsewhere in the repository |
| Attacks | the 7 MNIST attacks | `fltrust_aligned` outside the confirmatory grid |
| Systems | 7 fixed rules, Random-selector, attack-aware, GRADF, FedStrategist, AdaAggRL (random-weight extractor) | |
| Device | declared per experiment in the E0 pre-registration | the runs of one paired comparison never mix devices; any GPU run uses deterministic operations or is excluded from exact-invariance checks |

**Two grids:**
- **Full grid:** 3 α × 7 attacks, fixed rules only (E1, E2).
- **Common confirmatory grid:** the full grid without `fltrust_aligned`, or a reduced subset if P-0 requires it (E3). It is the **same** for every adaptive system and for the Random-selector.

**Implementation prerequisites before E0 runs:**
- the CIFAR-10 runner covers the 7 rules × 7 attacks × 3 α with seeded runs;
- AdaAggRL's gradient inversion is implemented for the CNN (the current implementation only supports the linear model);
- GRADF's Layer-2 noise is recalibrated to the CNN's parameter count, since its noise grows with √(number of parameters);
- each item is recorded as a reproduction deviation (specification × reproduction × reason × expected impact × classification: faithful / necessary / sensitivity).

### E0. Feasibility pilot (gate P-0)

Seeds outside the confirmatory block.
1. FedAvg without attack at the three α, 2–3 seeds: accuracy-per-round curves.
2. Fixed rules under 1–2 attacks at α = 0.05: do the rules separate beyond the across-seed noise?
3. Costs. On CPU: s/round for 1 and for k parallel processes. If the GPU is used for the campaign: GPU, with and without deterministic operations. AdaAggRL's per-round inversion cost on the chosen CNN, on each device used. Total cost of the grids.

**Gate P-0:**
- Rounds are fixed where no-attack FedAvg accuracy stabilizes.
- If at α = 0.05 accuracy sits near the floor and the rules do not separate, α = 0.05 is replaced by a milder level (e.g., 0.3), recorded as a deviation from MNIST.
- If AdaAggRL's cost is infeasible, the **common confirmatory grid is reduced for every system** (e.g., 2 α levels and 3–4 representative attacks), keeping the same cells for Random-selector, attack-aware, GRADF, FedStrategist and AdaAggRL. No asymmetric grid for AdaAggRL alone.
- If the confirmatory grid is reduced, d\*, B(c) and O(c) used in E3 are recomputed on the reduced grid. The full-grid E2 values remain reported as the RQ2 result, not as E3's reference.

**Output:** a pre-registration frozen by hash, fixing rounds, local epochs, α, confirmatory grid, seeds, devices, the E1 comparator and the hypotheses.

### E1. Root dependence (RQ1)

- Only the fixed rules.
- Full grid: 3 α × 7 attacks × 2 roots (100 and 2,500) × 10 seeds. Optional: root 50, as on MNIST.
- The same seed gives the same partition and the same run seed at both root sizes, so the comparison is paired.
- Outputs: wins per rule and per root (descriptive), and the frozen statistic below.

**FLTrust advantage (frozen):**

> V(c, s, r) = Acc\_FLTrust(c, s, r) − Acc\_TM(c, s)

- **Why the trimmed mean is the comparator:** it is a strong robust baseline and uses no server data. It is therefore root-invariant (verified in P-1 and at the start of E1), a natural negative control for the manipulated resource. That it was d\* on MNIST is supporting evidence, not the reason.
- The cell's in-sample best relational rule is **not** used, because it reintroduces maximization bias.

**Confirmatory hypothesis H1:** FLTrust's mean advantage is larger with the enlarged root than with the canonical one.
- Per-seed statistic: D\_s = mean over c of [V(c, s, 2500) − V(c, s, 100)], on the grid without `fltrust_aligned`.
- The 21-cell version is reported as sensitivity.
- Paired test over the 10 seeds (Sec. 3).

The same analysis is applied to MNIST (root 3,000 vs. 100) in P-1.

### E2. Cross-condition headroom (RQ2)

- Uses E1's canonical-root runs; no new runs.
- Computes O(c), B(c), d\*, H(c) and H. The criterion for d\* and its sensitivity when the runner-up is close are declared.
- **Estimator: LOSO (Sec. 2):**
  1. For each seed s, select r\*\_{−s}(c) and d\*\_{−s} on the other 9 seeds.
  2. Compute Ĥ\_s on seed s.
  3. Ĥ\_LOSO is the mean of the Ĥ\_s.
- **Uncertainty:** 95% seed-bootstrap CI, redoing steps 1–3 in each resample (Sec. 3).
- **Confirmatory hypothesis H2:** out-of-sample headroom is positive, i.e., the 95% CI of Ĥ\_LOSO lies entirely above zero.
- Reported side by side: the in-sample estimate and Ĥ\_LOSO with its CI (the confirmatory estimate). Also reported: how often d\*\_{−s} and r\*\_{−s}(c) agree with the in-sample choices.
- **Pre-declared reading:** Ĥ\_LOSO close to zero is a possible and informative outcome. If the in-sample estimate is positive and Ĥ\_LOSO is near zero, part of the apparent opportunity comes from selecting the winner on the same data used to measure its advantage.
- The same procedure is applied to MNIST (P-1).

### E3. Headroom capture (RQ3)

- **Systems:** Random-selector, attack-aware, GRADF, FedStrategist, AdaAggRL.
- **Grid:** the common confirmatory grid (Sec. 5.0).
- **References:** d\*, B(c) and O(c) are computed on the confirmatory grid itself (full or reduced). The confirmatory analysis uses the cross-fitted B\_{−s}(c, s) and O\_{−s}(c, s). R\_s(c) with in-sample references may be reported, labelled descriptive.
- **Confirmatory families** (Holm within each; unit = seed):
  - each system × Random-selector, aggregated over cells;
  - each system × cross-fitted global best fixed rule: A\_sys(c, s) − B\_{−s}(c, s), aggregated over cells.
- Capture counts are descriptive.
- **Sensitivity:** AdaAggRL with the pretrained extractor; FedStrategist with GRADF's detector.
- **Dependency:** E3 runs may proceed in parallel with E1. E3's **analysis** starts only after B(c), O(c) and d\* of the confirmatory grid are computed from the fixed-rule runs.
- The same procedure is applied to MNIST (P-1), on the 18-cell grid without `fltrust_aligned`.

### E4. Temporal headroom

**Goal:** show a regime in which the best rule changes **within** a run.

- **Dataset:** MNIST, which is cheap; repeated on CIFAR-10 if cost allows. One or two α levels.
- **Phase skeleton,** to be fixed in the E4 plan. Example:
  1. rounds 1–T1: no attack;
  2. T1–T2: sign flipping (preserves magnitude);
  3. T2–T3: Gaussian noise (changes magnitude);
  4. T3–T4: trim attack (informed).

**Phase-wise greedy reference (not an oracle).**
- When the rule switches mid-run, the model carries the state left by the previous phase. The sequence of phase winners is therefore a greedy reference, not a ceiling.
- Definition: run the winning rule of each phase only during that phase, switching at the change point.
- Temporal headroom = phase-wise greedy reference − best fixed rule over the whole run.

**E4a (always run): metric check.**
- Phases may be chosen deliberately from the rule ranking, so that the best rule differs in each phase.
- The headroom is declared as constructed to validate the metric. No independent confirmatory claim.

**E4b (conditional on P-2): confirmatory.** Development/confirmation split:
1. On development seeds (MNIST 42–51, existing), design the scenario: phase sequence and each phase's rule in the greedy reference.
2. Freeze the scenario and the hypothesis in a pre-registration.
3. Test it on confirmation seeds (a new, never-used block) and/or on CIFAR-10.

- **Systems:** fixed rules, phase-wise greedy reference, Random-selector, GRADF, FedStrategist, AdaAggRL.
- **Hypotheses:**
  - E4a: concept validation only.
  - E4b: confirmatory, temporal headroom > 0 on the confirmation seeds.
  - Exploratory: which systems capture part of it.

### Order and dependencies

```
P-1: MNIST audit + E1-E3 analyses on MNIST + root invariant
        |
        v
E0: CIFAR-10 pilot --> P-0 --> pre-registration (hash)
                                    |
                +-------------------+--------------------+
                v                   v                    v
      E1 runs (fixed rules,   E3 runs (adaptive,     E4a (MNIST)
      full grid)              confirmatory grid)
                |                   |
                v                   |
      E1 analysis --> E2 analysis   |
                |                   |
                +----> B(c), O(c), d* of the confirmatory grid (LOSO)
                                    |
                                    v
                              E3 analysis
                                    |
                                    v
                                   P-2
                         +----------+----------+
                         v                     v
                  E4b (new seeds)        E4b not run
```

---

## 6. Gate P-2: is E4b needed?

After E1–E3 and E4a, four questions:
1. Does the root effect seen on MNIST also appear on CIFAR-10 (H1)?
2. Is there measurable out-of-sample headroom on CIFAR-10 (H2)?
3. Do the systems differ informatively in capturing it (E3)?
4. Do MNIST, CIFAR-10 and E4a together establish the two-stage evaluation (headroom first, capture second)?

If the answers are conclusive, E4b is not run. Otherwise it is.

---

## 7. Success criteria of the experimental campaign

The campaign succeeds if it can establish the following, whatever the direction of each result:
1. whether the root configuration materially changes the comparative structure of the fixed defenses (H1, MNIST and CIFAR-10);
2. whether headroom can be estimated out of sample (H2). A small or null Ĥ\_LOSO does not invalidate this criterion;
3. whether the phenomena go beyond MNIST with logistic regression;
4. whether complete systems differ in capturing the available opportunity (E3);
5. in all cases, without crediting the learned component with an effect that these experiments do not isolate (Sec. 1).

---

## 8. Changes relative to the design document of 2026-10-28

The design was written on 2026-10-28 (working document, not versioned). This plan freezes it restricted to the experiments. The following points differ from that document or were added after the P-1 audit (`results/p1_audit/AUDIT.md`):

| # | Change | Reason |
|---|---|---|
| 1 | Wilcoxon as primary test, t-test as complementary, CI from the t distribution | the design listed the tests without ranking them |
| 2 | Bootstrap LOSO also removes copies of the held-out seed from the selection set | avoids leakage from evaluation into selection within a resample |
| 3 | In-sample headroom labelled "in-sample estimate", not "optimistic upper bound" | on MNIST (21 cells) the cross-fitted estimate exceeded the in-sample one, because B(c) is also selection-biased |
| 4 | Root invariant made exact via seeded runs, with `fltrust_aligned` as a declared exception; checked for median and Krum too | the audit found that NumPy's global RNG was never seeded and that the attacker in `fltrust_aligned` copies the server's root update |
| 5 | E1 confirmatory statistic computed without `fltrust_aligned`; 21-cell version as sensitivity | same reason as 4 |
| 6 | Partition stated as fixed per α (inference conditional on it) | the audit found the partition does not change with the seed |
| 7 | MNIST E3 on the 18-cell grid, with d\* recomputed on it | the plan's E3 rule applied to MNIST |
| 8 | Implementation prerequisites (seeded runner, CNN inversion for AdaAggRL, GRADF noise recalibration) and per-experiment device declaration | not covered by the design; needed for CIFAR-10 |
| 9 | E0 cost measurement on each device used (CPU and, if used, GPU) | the design only measured CPU |
| 10 | Local epochs fixed in E0 | the per-round cost scales with it |
