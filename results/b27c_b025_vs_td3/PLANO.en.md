# Plan — B2.7c (reformulated): does the fixed threshold b = 0.25 beat TD3 at H = 150?

*English translation of `PLANO.md`. The Portuguese original is the frozen document (hash in `PLANO.sha256`); if the two differ, the original prevails.*

**Status:** **CONFIRMATORY**. Final plan, written before the code and before any run (hash in `PLANO.sha256`). The code will be written, smoke-tested and have its hash recorded in an addendum **before** launch, without changing anything in this plan.
**Date:** 2026-10-11
**Roadmap:** `references/roadmap_tese_gradf_v4_1.md` (§4.1 B2.7c; seeds 82–91 reserved). The original B2.7c (confirming that TD3 learns) was cancelled at B2.7a (condition 1: 0/3). That freed the seeds, and they are used here for a different question.
**Background:**
- **B2.6** (`results/b26_decomposicao/`, seeds 52–61): the threshold b is the component with the largest effect. b = 0.25 is the best variant (+1.6 p.p. over the center), and b = 0.75 is the worst (−10.5 p.p.).
- **B2.7b** (`results/b27b_td3_constante/`, seeds 42–51): td3_ref − fixed_b025 = −6.9 / −4.4 / −4.4 p.p. in the 3 cells, with the whole CI95 below 0. **Limitation stated there:** "b = 0.25 is chosen in hindsight".

## 0. Why this experiment

B2.7b stated that "TD3 loses to a simple constant", but that constant was chosen after seeing the data: it was the best in hindsight among 4 arms, on the same seeds. This experiment **settles** the question. It fixes b = 0.25 **beforehand**, based on B2.6 (disjoint seeds), and tests it against td3_ref on fresh seeds (82–91) with a pre-registered one-sided test.

## 1. Systems, cells, horizon and seeds

| system | definition |
|---|---|
| **td3_ref** | the ablation learner (`scripts/frente1_ablacao_adaaggrl.build_learner_cls`) in `td3` mode, identical to B2.7/B2.7a |
| **fixed_b025** | the same skeleton with a constant action a = [0.5; 0.5; 0.5; 0.5], b = 0.25 and no TD3 (the same path as B2.7b's `fixed_b025`) |

- **Cells:** the same 3 as in B2.7a/b, chosen in B2.7 because they favor TD3, which makes the test conservative for the hypothesis:
  - `label_flipping` α 0.05;
  - `label_flipping` α 0.1;
  - `low_mag_backdoor` α 0.05.
- **H = 150:** the accuracy of the global model at the end of round 150. H = 15 and 50 are also recorded, as descriptive only.
- **Seeds 82–91** (n = 10 per cell), paired by seed between the two systems.
- **Regime:** identical to B2.7: logistic MNIST, 10 clients, Byzantine [0, 1], root 100 (`root_seed = seed`), with `keras.utils.set_random_seed(seed)` before each system.
- **Grid:** 2 × 3 × 10 = **60 runs**.

## 2. Hypothesis and tests (fixed now)

The unit is the seed. D = acc(fixed_b025) − acc(td3_ref) at H = 150, paired by seed.

**H1 (primary, per cell):** E[D] > 0.
- **Test:** **one-sided** paired t-test (H0: E[D] ≤ 0), n = 10, df = 9.
- **Multiplicity:** Holm–Bonferroni over the 3 cells, family α = 0.05 (one-sided).
- **Verdict:**
  - **"b = 0.25 beats TD3" (H1 confirmed):** Holm rejects H0 in **≥ 2 of 3 cells**, and no cell has a mean D < 0.
  - **"confirmed in all cells":** Holm rejects in all 3.
  - **"not confirmed":** any other outcome. Not confirming is **not** evidence of equivalence and will not be reported as such.

**H1b (secondary, pooled):** E[D̄] > 0, where D̄ is the per-seed mean of D over the 3 cells (n = 10). One-sided paired t-test at α = 0.05, with no correction (it is a single secondary test).

**Also reported (descriptive, no verdict):**
- mean D, two-sided t CI95 and one-sided lower CI90, per cell;
- one-sided Wilcoxon signed-rank p, for robustness to non-normality;
- Cohen's d_z;
- D at H = 15 and H = 50;
- comparison with B2.7b's D (seeds 42–51), to check whether the effect shrank because b was selected in hindsight.

**Power** (per-cell sd from B2.7b, 3.3–5.3 p.p.):
- if B2.7b's effect replicates: 0.80–0.94 per cell at the strictest Holm α (0.0167), and 0.93–0.99 at α = 0.05;
- if the effect halves: 0.26–0.40 and 0.47–0.63.

The pooled test H1b exists for the second scenario.

## 3. Checks (before launch, recorded in the addendum)

A 15-round smoke test through the new code, on seed 42, `label_flipping` α 0.05:
- td3_ref must reproduce B2.7's `td3_ref` at H = 15 (|Δ| = 0);
- fixed_b025 must reproduce B2.7b's `fixed_b025` at H = 15 (|Δ| = 0).

If either check fails, the code is fixed **before** the freeze. The plan does not change.

## 4. Rules

- **Mandatory order:**
  1. this plan, with its hash;
  2. the code, the smoke test and the code hash in the addendum;
  3. the grid.
- No accuracy from seeds 82–91 is inspected before the full grid (60/60) is complete.
- A failed run is re-run with the same seed. If it cannot be re-run, the cell is analysed with the available n, and this is stated.
- **Cost:** ~30 min per run, 12 CPU processes (`nice 19`, 2 threads, no GPU). That is 5 waves, ~2.5–3 h of wall-clock time. The grid respects the Mon–Fri 7:00–18:00 window (`scripts/janela_execucao.sh`).
- No commit. Files stay untracked until the author decides.
