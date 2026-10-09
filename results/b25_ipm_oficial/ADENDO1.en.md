> English translation of `ADENDO1.md`. The Portuguese original is the frozen record (its hash is in `ADENDO1.sha256`); if the two ever disagree, the original prevails.

# Addendum 1 — B2.5: ε chosen in the sanity check

**Date:** 2026-10-01, before any grid run (hash in `ADENDO1.sha256`).
**Plan:** `PLANO.md` §3 (commit `1e92014`). Mechanical application: `scripts/passo2_oficial/analisar_b25.py sanity` → `sanity.txt`.

## Sanity result (seed 100, 100 rounds; mean 81–100)

| condition | accuracy 81–100 | criterion |
|---|---|---|
| FedAvg without attack | 93.38% | reference |
| FedAvg, ε = 2 | 87.48% (−5.90 p.p.) | degrades ✔ |
| FedAvg, ε = 10 | 29.91% (−63.48 p.p.; 8 resets) | degrades ✔ |
| fixed, ε = 2 | mean mass on attackers 0.1408 (92 rounds with real attackers) | not excluded ✔ |
| fixed, ε = 10 | mean mass on attackers 0.0103 | not excluded ✔ (borderline) |

td3 did not enter the sanity check.

## Choice

**ε = 2.** Both candidates meet both criteria, and the tie-break rule picks the smaller one.

Observations, descriptive and with no effect on the choice:
- With ε = 2, IPM degrades FedAvg only slightly above the threshold (5.9 p.p.), and fixed gives substantial weight to the attackers (~0.14 vs. ~0.2 for uniform). It is the regime in which the weighting **can** make a difference.
- ε = 10 would be almost fully excluded by fixed (0.0103, at the edge of the criterion).

## Grid

20 runs: real IPM ε = 2 × {td3, fixed} × seeds 105–114, 500 rounds, 5 in parallel on the GPU:
`bash scripts/passo2_oficial/run_grid_b25.sh grade 2`.
Analysis: `analisar_b25.py grade --eps 2`, once, with the complete grid.
