> English translation of `PLANO.md`. The Portuguese original is the frozen record (its hash is in `PLANO.sha256`); if the two ever disagree, the original prevails.

# Plan — B2.8: oracle ceiling of per-round selection (granularity axis)

**Status:** FINAL before the data (hash in `PLANO.sha256`). Exploratory.
**Date:** 2026-09-30
**Cost note:** recalibrated: it **cannot be computed from the logs**, because the per-round counterfactuals are not recorded; these are new, cheap CPU runs.
**Record:** local, with a hash; committing is up to the author.

## 1. Question

P1 attributed the difference between discrete rule selection (GRADF, FedStrategist) and per-client weighting (AdaAggRL) to "discrete vs. continuous". P2 (B2.1–B2.3) shows that learning does not explain it: the fixed skeleton does the work. **Does granularity explain it?** That is, would a policy that chose the **best possible rule every round** reach the skeleton's per-client filtering?

## 2. Design

- **Greedy per-round oracle:**
  - every round, the updates (with the attack) are computed once;
  - each of the 7 rules of the exp10 arsenal (`fedavg`, `fedprox`, `median`, `trimmed_mean`, `fltrust`, `clustering`, `krum`) aggregates those updates;
  - the candidate with the highest accuracy becomes the new global model.

  The metric is the same as C.0's: mean accuracy over the 10 clients' `X_test`.
- **It is an upper bound**, not a deployable method: it chooses by looking at the test set. If **not even it** reaches the skeleton, the conclusion is strong.
- **Regime identical to C.0, the ablation and exp9:** MNIST, 10 clients, Byzantine [0, 1], 15 rounds, root 100, **seeds 42–51**, 3 alphas × 7 attacks = 21 cells, 210 runs.
- **References paired by seed** (data that already exist):
  - best of the skeleton among `fixed`, `sr_only` and `td3_ref` (ablation);
  - best **static** fixed rule (exp9, root 100);
  - FedAvg without attack (C.0).
- Cost: ~0.68 s per round (measured), ≈ 36 process-minutes; CPU, `nice 19`, 2 threads, in parallel with B2.3b on the GPU.
- Script: `scripts/b28_oraculo_por_regra.py` (`run` and `analisar`).

## 3. Quantities per cell (paired mean by seed, CI95)

- **Q** = per-round oracle − best skeleton;
- **W** = best skeleton − best static rule;
- **G** = per-round oracle − best static rule (how much per-round choice gains over a fixed choice).

## 4. Criterion (fixed before the data)

- **Valid cells:** 19 (excludes `fltrust_aligned` α = 0.05 and 0.1, an attack artifact; C.0 `NOTAS.md` §2).
- **Target cells:** valid cells in which the skeleton beats the best static rule (**W with CI95 > 0**). This is where per-client weighting "wins" and the question makes sense.
- **Verdict:**
  - **"Granularity explains"** if, in at least half of the target cells, the per-round oracle is **below** the skeleton (Q with CI95 < 0);
  - **"Granularity does not explain"** if, in at least half of the target cells, it is **above** (Q with CI95 > 0);
  - otherwise, **inconclusive**;
  - if there are no target cells, report it as such.
- Reported without a criterion: G per cell, frequency of the rules chosen by the oracle, and the 2 excluded cells.

## 5. Declared limitations

- **Exploratory:** already-used seeds; ablation and exp9 references already known.
- **Greedy oracle:** optimal per round, not necessarily over the trajectory. Even so, it is a generous ceiling for any deployable selector.
- **15 rounds:** the gap mixes robustness and convergence (C.0 §6.3); B2.7 addresses the horizon.
