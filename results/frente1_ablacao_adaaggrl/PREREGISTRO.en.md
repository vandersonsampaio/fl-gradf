> English translation of `PREREGISTRO.md`. The Portuguese original is the record (written before the grid, timestamped, no hash file); if the two ever disagree, the original prevails.

# Pre-registration: AdaAggRL ablation (Front 1, after V0)

**Date:** 2026-09-24, written **before** running the grid (step 3).
**Not committed at the time, by the author's instruction.** This file and the results are kept in `results/frente1_ablacao_adaaggrl/`. The file timestamp and the order recorded in the session serve as the record.
**Origin:** the V0 diagnostic (`scripts/frente1_v0_diagnostico.py`: TD3 does not learn in 15 rounds, and S_R dominates) and two proposals by the author: (1) AdaAggRL without RL; (2) S_R vs. trivial detectors.
**Code:** `scripts/frente1_ablacao_adaaggrl.py`. Changes nothing in `src/`.

## 0. What had already been seen before this record (transparency)

- **V0:** 1 seed, 3 cells, with cues, ranges and TD3 actions.
- **Step 1, detector AUC along the reference trajectory:** seed 42, **21 cells** (`passo1_auc_detectores.csv`). `cos_median`, declared before seeing the data as the trivial detector of the confirmatory variant, had the **worst** mean AUC (0.465). `cos_server` had the best among the trivial ones (0.704).
  - Consequence: `cosmed_only` stays **confirmatory**, as declared. `cosserver_only` enters as **exploratory**, because it was chosen after seeing data from seed 42, which is also part of the grid.
- **Methodology check, not a result:**
  - the script's `td3` trajectory reproduces the official seed-42 AdaAggRL with a difference of 0.0 in the 21 cells;
  - the headroom function reproduces the documented **9/21** for the reference (and gives 3/21 for Random).

## 1. Protocol, identical to the reference's

MNIST, root=100, seeds 42–51 (10), α ∈ {0.5; 0.1; 0.05}, 7 attacks (`fltrust_aligned`, `gaussian_noise`, `krum_collusion`, `label_flipping`, `low_mag_backdoor`, `sign_flipping`, `trim_attack`), `n_rounds=15`, `n_clients=10`, Byzantine = clients {0, 1}. The metric is the global accuracy of the last round (mean of the per-client test accuracies).

**Reference (`td3_ref`):** AdaAggRL with the `random` extractor, reused from `results/tables/exp10_selector_comparison_variantb_full_seed{42..51}_raw.csv`, not recomputed. It is the reference recommended by V0.
**Random:** from the same file.
**Best fixed rule:** `exp9_dominance_grid_10seeds_ALL_root100_raw.csv`.

## 2. Variants

All keep the AdaAggRL mechanism: min–max normalization of ŵ = S·a, threshold δ = max(w̃)·b, counter h, penalty λ^h with λ=2, and weighted aggregation of the full parameters. The `random` extractor is always built, to preserve the RNG and the pairing by seed.

| variant | per-client score | action | gradient inversion | status |
|---|---|---|---|---|
| `fixed` | (S_R, S_cl, S_cg, S_lg), as in the reference | **a = [0.5; 0.5; 0.5; 0.5], b = 0.5**, no TD3 | yes | confirmatory (H1) |
| `sr_only` | S_R | a = [1, 0, 0, 0], b = 0.5 | yes, without extractor or MMD | confirmatory (H2) |
| `cosmed_only` | cosine to the coordinate-wise median | a = [1, 0, 0, 0], b = 0.5 | **no** | confirmatory (H3) |
| `cosserver_only` | cosine to the server update on the root | a = [1, 0, 0, 0], b = 0.5 | no | **exploratory** |

## 3. Hypotheses and tests

**Unit of analysis:** seed. In the aggregate tests, the per-seed mean over the 21 cells is computed first (n=10), and only then the test. Per attack, the per-seed mean is over the 3 α.

- **Equivalence:** paired TOST (t) with a **±0.01** accuracy margin, α=0.05. There is equivalence when the 90% CI of the mean difference lies within ±0.01.
- **Difference:** paired, two-sided Wilcoxon signed-rank, α=0.05. Δ and paired Cohen's d are also reported.
- **Per attack:** Holm-Bonferroni over the 7 attacks, separately for the TOST p-values and the Wilcoxon p-values.
- **Headroom:** count of cells in which the difference to the best fixed rule exceeds the sum of the standard deviations, by the same criterion as exp9 and Step Zero. It is **descriptive**, not a test.

**H1. TD3 does not contribute in this regime:** `fixed` − `td3_ref`.
- Aggregate equivalent (p_TOST < 0.05) → **H1 confirmed**. V0's finding stops being observational.
- Aggregate Wilcoxon with p < 0.05 and `fixed` worse → TD3 contributes. H1 refuted.
- Aggregate Wilcoxon with p < 0.05 and `fixed` better → TD3 hurts. H1 refuted in the strong sense.
- None of the above → **inconclusive**. The ±0.01 margin is not resolvable with n=10.

**H2. The MMD cues do not contribute beyond S_R:** `sr_only` − `td3_ref`, with the same rules.

**H3. Gradient inversion does not contribute (a trivial detector is enough):** `cosmed_only` − `td3_ref` and `cosmed_only` − `sr_only`, with the same rules.
- *Prediction recorded after step 1:* H3 should **fail** with `cos_median`, which had a mean AUC of 0.465 and AUC ≈ 0 against informed attacks.
- If it fails, this does **not** refute the general idea that "trivial detectors are enough". It only refutes this trivial one. The exploratory `cosserver_only` variant informs about the general idea, but does not confirm it.

**Readings for Front 2:** a privacy differentiator is only sustainable with a variant without inversion that is **equivalent or better** than `td3_ref` in the aggregate and also beats Random. An exploratory result (`cosserver_only`) requires replication on new seeds before becoming a claim.

**Additional descriptive comparisons:** each variant vs. Random, aggregate and per attack, with Holm.

## 4. Do not

- Do not change the margin, the values of a and b, or the list of confirmatory variants after seeing grid results.
- Do not report `cosserver_only` as confirmatory.
