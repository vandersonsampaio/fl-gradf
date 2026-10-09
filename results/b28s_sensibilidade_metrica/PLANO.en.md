> English translation of `PLANO.md`. The Portuguese original is the frozen record (its hash is in `PLANO.sha256`); if the two ever disagree, the original prevails.

# Plan — B2.8s: does the granularity verdict (B2.8) depend on the metric?

**Status:** FINAL before any run (hash in `PLANO.sha256`). **Post hoc** sensitivity, motivated by the finding that the sample-weighted oracle ceiling underestimates the ceiling under the uniform metric (`results/c0b_espaco_h150/VERIFICACOES.md` §3).
**Date:** 2026-10-04
**Background:** B2.8 (`results/b28_oraculo_por_regra/`). Verdict: "granularity explains" (8/14 target cells; robust at α ≤ 0.1 with model attacks).

## 1. Question

Does the conclusion "not even the best rule per round reaches the per-client filter" hold when the metric stops being the **uniform mean** of the accuracies on the clients' test sets and becomes the **mean weighted by each client's test-set size** (≈ accuracy on the aggregate test set)?

**Why it matters:** the skeleton's binary mask amounts to uniform weights among the honest clients, and the uniform metric rewards that. In the per-round oracle's arsenal, FedAvg and FedProx weight by sample size; the other 5 rules do not (inventory in C0b's `VERIFICACOES.md`).

## 2. Design

Re-run all of B2.8's pieces **recording the per-client accuracies**, so that both metrics come from the same runs:
- **per-round oracle `oracle_u`:** chooses the best of the 7 rules every round by the **uniform** metric, as in B2.8;
- **per-round oracle `oracle_w`:** same, but choosing by the **weighted** metric, so that the oracle is optimal in the metric it is evaluated on;
- **skeleton:** `fixed`, `sr_only` and `td3_ref` (the ablation learner, as in B2.8);
- **static rules:** FLTrust, Krum, Median and Trimmed-Mean, with the exp9 learners (`InformedAttackedFederatedLearner` for the informed attacks), as in B2.8.

**Regime:** identical to B2.8: MNIST, 10 clients, Byzantine [0, 1], 15 rounds, root 100, **seeds 42–51**, 21 cells. Total: 9 systems × 21 cells × 10 seeds = **1,890 runs**, one job per cell × seed, ~1.5–3 CPU hours with 10 processes.

**RNG:** `keras.utils.set_random_seed(seed)` before each system, as in B2.7/C0b. The original B2.8 did not reseed, so the uniform-metric numbers **do not need to reproduce** B2.8 bit for bit. The comparison between metrics is **within this grid**.

**Metrics** (of the final global model, round 15):
- **uniform:** mean of the 10 clients' accuracies on their `X_test`;
- **weighted:** Σᵢ nᵢ·accᵢ / Σᵢ nᵢ, with nᵢ = size of client i's `X_test`.

## 3. Quantities and criterion (fixed now)

For each metric m, with the corresponding oracle (`oracle_u` for the uniform one, `oracle_w` for the weighted one), per cell (paired mean by seed, CI95):
- **Q** = oracle − best skeleton;
- **W** = best skeleton − best static rule;
- **G** = oracle − best static rule.

The best of each group is chosen per cell, in its own metric. The target cells, the verdict and the 19 valid cells follow **B2.8's PLANO §4**.

**Sensitivity criterion:**
- **"Granularity robust"** if B2.8's verdict is the **same** under both metrics **and**, among the target cells common to both, the sign of Q with CI95 (below / above / inconclusive) agrees in **at least 75%**.
- **"Depends on the metric"** if the verdict changes. In that case, P2 must say that part of the per-client filter's advantage comes from weighting the honest clients uniformly under the uniform metric.
- Intermediate case (same verdict, agreement < 75%): **"robust in the verdict, sensitive per cell"**, reporting the cells that change.

**Reported without a criterion:**
- whether the uniform-metric verdict in this grid reproduces the original B2.8 verdict;
- the frequency of the rules chosen by `oracle_u` and `oracle_w` (do FedAvg/FedProx gain ground under the weighted metric?).

## 4. Rules

- Code with its hash in an addendum before the launch; pre-written analysis, run once with the complete grid.
- No accuracy is inspected before the grid ends.
- Logs not versioned.
