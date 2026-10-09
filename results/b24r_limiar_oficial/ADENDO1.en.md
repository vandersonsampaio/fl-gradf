> English translation of `ADENDO1.md`. The Portuguese original is the frozen record (its hash is in `ADENDO1.sha256`); if the two ever disagree, the original prevails.

# Addendum 1 — B2.4r: the center check failed; center re-run in the grid

**Date:** 2026-10-02 20:36:26 -0300, before any grid run. Foreseen in §3 of `PLANO.md`.

## Check (PLANO §3)

- The new runner (`scripts/passo2_oficial/run_b24r.py`, which only replaces `run_oficial.A_FIXED` and calls `run_oficial.run`, the same runner as Step 2) was run with a₅ = 0.475, seed 100, 25 rounds.
- It **did not reproduce** Step 2's `fixed` EB: max. |Δacc| = 8.4 p.p. over the 25 rounds.
- The initial state is identical (accuracy and loss of the initial reset are equal), with the same torch version (2.3.0+cu121) and the same device (cuda:0). The divergence starts **at round 1**.
- **Diagnosis:** the official code on GPU is **not bit-reproducible across executions** (non-deterministic cuDNN operations). The seed fixes the initialization, but not the trajectory.

## Consequence (PLANO §3 rule)

- The **center (a₅ = 0.475) is re-run** on seeds 100–104, within the same grid. The grid now has **25 runs** (5 values of a₅ × 5 seeds), ~35 GPU hours with 5 processes.
- The analysis uses the re-run center. The Step 2 center appears only as a descriptive reference (`centro_passo2`).
- Criteria, metrics and reading **do not change**.
- **Note for P2:** the GPU non-determinism applies to all experiments on the official code (Step 2, B2.1–B2.5). There, pairing by seed pairs the configuration, not the trajectory. The paired tests remain valid, because each run is a sample from the same distribution.

## Code at freeze time
dc95bbbd2312ed4cb73ec2855b22ac83980f19f9dde8f5baf4d367c051675c99  scripts/passo2_oficial/run_b24r.py
715e40e4a405c8e9a9ef07e6267acf84e82a7249f28b23c722e20bd9fde28f2d  scripts/passo2_oficial/run_grid_b24r.sh
794d9adc9d1b808bedf27ef7b2f08ce896daab4318c3224998ee771256396ba2  scripts/passo2_oficial/analisar_b24r.py
