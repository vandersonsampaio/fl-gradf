> English translation of `PLANO.md`. The Portuguese original is the frozen record (its hash is in `PLANO.sha256`); if the two ever disagree, the original prevails.

# Plan — B2.4r: reduced sweep of the threshold a₅ in the official AdaAggRL

**Status:** FINAL before any run (hash in `PLANO.sha256`). **Exploratory.** The code does not exist yet: it will be written, tested and have its hash recorded in an addendum **before** the launch, without changing anything in this plan.
**Date:** 2026-10-02

## 1. Questions

1. **Causal effect of the threshold in the published code:** does changing only a₅ (the threshold δ = max(k)·a₅), with the rest of the action at the center, change the accuracy under EB? In the in-house framework the threshold has the largest effect (B2.6: b = 0.75 costs 10.5 p.p.). In the official code, the current evidence is only 2 accidental collapses in B2.3.
2. **Is there a constant better than the center?** It is the same question as B2.7b, now in the official code.

## 2. Design

- **Code:** official (`external/AdaAggRL`, commit `27b9c18`), venv `external/.venv_adaaggrl`, without any edit. The runner reuses `run_b21.py`/`run_oficial.py` and only changes the fixed action.
- **Regime:** identical to Step 2 and B2.1 (MNIST, q = 0.5, 100 clients, 10% per round, 20 attackers, 500 rounds). Attack **EB** only: LMP is dropped because it is insensitive to a₅.
- **Conditions:** fixed action [0.475; 0.475; 0.475; 0.475; a₅] with **a₅ ∈ {0; 0.1; 0.25; 0.95}**. The official Box is [0; 0.95].
  - The **center** (a₅ = 0.475) comes from Step 2's `fixed` EB runs (`results/frente1_passo2_oficial/raw/`, same seeds). It is not re-run.
  - With a₅ = 0, the lowest-scoring client is still excluded, because k = 0 after the min-max. This is a property of the official mechanism, not a disabled filter.
- **Seeds:** 100–104 (the same as Step 2, for pairing with the center).
- **Total:** 4 × 5 = **20 runs**, 5 in parallel on the GPU, **~28–30 h**.
- **Logging:** the same as Step 2 and B2.1 (accuracy, loss, action, mass on attackers, number of excluded clients and of excluded attackers, resets). Partial checkpoints every 25 rounds.

## 3. Check before the grid

Run the new runner with a₅ = 0.475 on seed 100 for **25 rounds** and compare with Step 2. It must reproduce exactly (|Δacc| < 1e-9 per round) the trajectory of `fixed` EB, seed 100. If it does not, the center is re-run on the 5 seeds (+5 runs), via an addendum, before the grid.

## 4. Metrics

- **Primary:** median accuracy over rounds 401–500 (the same as B2.1 and B2.5).
- **Secondary, reset-robust:** mean accuracy over the 500 rounds (area under the curve).
- **Mechanism (descriptive):** number of resets, mean weight mass on real attackers, mean number of excluded clients and of excluded attackers per round.

## 5. Criteria (exploratory; n = 5; no multiplicity correction, 4 comparisons)

For each a₅: Δ = metric(a₅) − metric(center), paired by seed, with a t CI95.
- **Q1, the threshold has a causal effect:** in at least one a₅, |Δ| > 2 p.p. with a CI95 that excludes 0, on the primary **or** on the secondary. Also report the shape of the response (monotonic or not) and the mechanism (exclusions, mass, resets).
- **Q2, there is a constant better than the center:** in at least one a₅, Δ > 0 with CI95 > 0 on the primary **and** Δ > 0 on the secondary.
- Wilcoxon reported as descriptive: with n = 5, the smallest possible two-sided p is 0.0625.
- Reset cascades count as results, because they are a consequence of the action.

## 6. Reading

- **Q1 yes:** the threshold is causal in the published code too, which closes the Part I evidence. It is the component to explain (R6) and to protect (R5) in P3.
- **Q2 yes:** the published AdaAggRL operates below the best constant and TD3 does not find it (B2.1/B2.2). This reinforces "learning amounts, at most, to tuning the threshold".
- **Q2 no:** the center is already near the optimum under EB, and TD3 would have nothing to gain by tuning a₅.

## 7. Rules

- Pre-written analysis (script with its hash in the addendum), run once with the complete grid.
- No accuracy is inspected before the grid ends. Monitoring reports only health and progress.
- Runs that fail due to an environment error are repeated with the same seed and recorded.
- Logs and checkpoints are not versioned, as in the previous experiments.
