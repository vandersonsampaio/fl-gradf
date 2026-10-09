> English translation of `PLANO.md`. The Portuguese original is the frozen record (its hash is in `PLANO.sha256`); if the two ever disagree, the original prevails.

# Plan — C0b-iii: does the skeleton × static rules comparison (and the C0b map) depend on the metric?

**Status:** FINAL before any run (hash in `PLANO.sha256`). **Post hoc** sensitivity of C0b, restricted to **α ≤ 0.1**, where the client sizes are unequal (from ~10² to ~10⁴).
**Date:** 2026-10-04
**Background:** C0b (`results/c0b_espaco_h150/`: RESULTADO (iii), VERIFICACOES.md §3, sensitivity with the uniform oracle). Weighting inventory: none of the 8 C0b systems weights by sample size; the FedAvg of the ceilings does.

## 1. Question

Do C0b's conclusions, (iii) "the best skeleton variant beats the best static rule in most cells" and (i) the remaining-headroom map, hold with **two coherent pairs of metric and ceiling**?
- **Pair U:** **uniform** metric (mean of the accuracies on the 10 clients' test sets) with a **uniformly weighted** oracle-8 ceiling;
- **Pair P:** metric **weighted by test-set size** with a **sample-size-weighted** oracle-8 ceiling (the framework's FedAvg).

## 2. Design

- **Cells:** the 12 valid ones with α ≤ 0.1 (α 0.05 and 0.1 × `trim_attack`, `krum_collusion`, `low_mag_backdoor`, `sign_flipping`, `gaussian_noise` and `label_flipping`).
- **Systems and regime identical to C0b:** `sr_only`, `sr_bin`, `sr_b025`, `cosserver_only`, FLTrust, Trimmed-Mean, Clustering and Median; nested H 15/50/150; seeds 72–81; `_reseed(seed)` before each system.
- **New:** records the **per-client accuracies** of the global model at H = 15, 50 and 150, so that both metrics come from the same run.
- **Ceilings**, per α × seed: FedAvg-10 without attack and oracle-8 (honest clients only), each with uniform and sample-size weighting, with the per-client accuracies.
- **Total:** 12 × 10 × 8 = 960 cell runs + 20 ceiling jobs (4 runs each). ~18 CPU hours with 12 processes.
- **Reproduction check:** the uniform metric of these runs must **exactly** reproduce C0b's `grade_raw.csv` (|Δ| < 1e-9). The ceilings must reproduce C0b's (sample-weighted) and the sensitivity analysis's (uniform). If they do not, this plan's data only hold internally, and this is declared.

## 3. Criterion (fixed now)

For each pair (U, P) and each H, per cell (paired mean by seed, CI95):
- **(iii):** D = best variant − best static rule (each chosen per cell, in the pair's metric). Cell class: "variant" (CI95 > 0), "rule" (CI95 < 0) or "tie".
- **(i):** gap = the pair's ceiling − best existing. "Headroom" if gap > 2 p.p. with CI95 > 0 (the C0 criterion).

**Reading at H = 150 (primary):**
- **"(iii) robust"** if the class agrees between U and P in **≥ 75% of the 12 cells** **and** the majority direction (number of "variant" cells vs. number of "rule" cells) is the same in both pairs.
- **"(iii) depends on the metric"** if the majority direction changes. In that case, P2 says that part of the per-client filter's advantage comes from weighting the honest clients uniformly under the uniform metric.
- Intermediate case: "robust in direction, sensitive per cell", with the cells that change.
- **(i):** report the number of cells with headroom in each pair and which ones. The expectation (not a criterion) is that `label_flipping` α 0.05 remains the only or the main one.

## 4. Rules

- Code with its hash in an addendum before the launch; pre-written analysis, run once with the complete grid.
- No accuracy is inspected before the end.
- **Queue:** runs on the CPU after D2.
- Logs not versioned.
