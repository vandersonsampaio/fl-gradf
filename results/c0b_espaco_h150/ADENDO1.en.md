> English translation of `ADENDO1.md`. The Portuguese original is the frozen record (its hash is in `ADENDO1.sha256`); if the two ever disagree, the original prevails.

# Addendum 1 — C0b: code, per-client logging and checks

**Date:** 2026-10-03 09:25:37 -0300, before launching the grid. The plan (`PLANO.md`, commit `d676ec3`) and its criteria **do not change**.

## Implementation

- **`scripts/c0b_espaco_h150.py`**, with one job per cell × seed and `keras.utils.set_random_seed(seed)` before each system:
  - variants: B2.6's `DecompLearner`;
  - static rules: B2.7's `PlainRuleLearner`;
  - ceilings: B2.7's `run_ceiling`, already with the FedAvg-8 oracle fix.
- **Launcher:** `scripts/run_grid_c0b.sh`, 12 processes, 190 cell jobs + 30 ceiling jobs.
- **Grid checked:** 19 cells (`fltrust_aligned` only at α 0.5) × 8 systems: `sr_only`, `sr_bin`, **`sr_b025`**, `cosserver_only`, `fltrust`, `trimmed_mean`, `clustering` and `median`.

## Addition: per-client logging (author's request)

- In the 4 skeleton variants, per round × client, written to `raw/scores_<cell>_seed<s>_R150.csv.gz`:
  - `is_byz`;
  - **S_R**, in the variants that already do the inversion; NaN in `cosserver_only`, to avoid paying for the inversion;
  - **cos_server**, **always** computed, with save/restore of `np.random` as in B2.6;
  - the **mask** (`incluido` = weight > 0) and the normalized weight.
- **What it is for:** the A0 agreement analysis between signals (Spearman S_R × cos_server and AUC per signal, now with 10 seeds and 19 cells) and the signal-selection ceiling, without re-running.
- The pre-written analysis gained a descriptive agreement section (`sr_only` trajectory, rounds ≥ 2). The plan's criteria do not change.

## Checks (15 rounds, cell `label_flipping` α 0.05, seed 72)

- **Determinism (PLANO §4):** the same job in two processes gave identical accuracies for the 8 systems (max. |Δ| = 0).
- **Neutral logging:** the same job without logging gave identical accuracies (max. |Δ| = 0). Computing cos_server in all variants does not change the trajectory.

## Declaration

The dry run of the analysis (`analisar`), done on the smoke run above, **displayed accuracies** for that cell and seed (seed 72, `label_flipping` α 0.05, H = 15) before the grid. The plan and the criterion were already frozen and committed (`d676ec3`), and no design or analysis choice was made based on those numbers. These smoke runs are not part of the grid: it re-runs all cells and seeds.

## Revised cost

Each job has 3 variants with inversion (~1,600–1,900 s each with the CPU under contention), `cosserver_only` and 4 rules (cheap). That is ~1.8 h per job and 190 jobs with 12 processes, i.e. **~28–35 h**.

## Code at freeze time
97b0cdf13b85112927b969d0b770c4fdc25d1cd0b9bf9cbdf2d9f46f40d854fc  scripts/c0b_espaco_h150.py
0d2454805068b74779ffb5bf98af4f60911a0aff5556cba5a43825b8adfe1549  scripts/run_grid_c0b.sh
