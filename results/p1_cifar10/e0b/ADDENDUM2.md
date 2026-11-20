# Addendum 2 — E0b part D: cost probe code

**Date:** 2026-10-30 11:16 -0300, before the probe runs. The plan (`PLAN.md`, commit `8c9b28c`) and addendum 1 (commit `5d411e8`) do not change.

## What the probe does

`scripts/p1/run_cost_probe_d.sh [N=12]` starts N identical processes of `scripts/p1/cost_probe_d.py` at once (CPU, 1 thread each, deterministic ops), the campaign's load, then aggregates them into `costs.json`, the file `analyze_e0b.py` reads (PLAN Sec. 5).

Each process, on CIFAR-10, α = 0.5, `sign_flipping`, seed 0, root 100, 2 local epochs:

1. times the pretraining (`train_gradf_models`, 8 rounds, as in `exp10_selector_comparison`);
2. for each system (trimmed mean as the fixed-rule cost, Random-selector, attack-aware, GRADF, FedStrategist variant b, AdaAggRL with the random extractor), runs one warm-up round and times 3 rounds.

`costs.json` holds the mean over the N processes of the seconds per round of each system, the mean pretraining time per run, and `pretraining_h_full_grid` = 180 runs (18 cells × 10 seeds) × that time / 12 processes. Accuracies are not saved: the probe yields costs only.

## Points that make the plan more precise (no change to the rule)

- **Fixed rules and AdaAggRL are measured again**, under the same load and code as the other systems, instead of the plan's provisional values (cost probe of 2026-10-29 × 0.8 for the memory fix). The plan says the measured values replace the provisional ones; this applies it to all six systems.
- **One pretraining per (cell, seed)**, shared by GRADF, the attack-aware reference and FedStrategist, as `exp10_selector_comparison` does. The plan's "pretraining of GRADF and of the attack-aware reference" is therefore counted once, not twice.
- **Limitation:** costs are measured at one cell (α = 0.5, a blind attack). Informed attacks and other α may cost slightly more or less per round; this is not corrected.

## When it runs

It needs N × ~2 GB at the same time (the launcher refuses to start below 2N + 4 GB available), so it cannot run while W1's paused processes hold ~10 GB. It runs after W1 finishes (expected 2026-11-01) and after E0b, outside the Mon–Fri 8–18 window or with the author's authorization. Expected wall time ≈ 45–60 min.

Smoke test: the full sequence (pretraining, the six systems, the aggregate) ran on small synthetic 4×4×1 data in a scratch directory (not in the repository), with 2 pretraining rounds and 1 timed round. It found and fixed a duplicate `input_shape` argument for AdaAggRL before this hash.

## Code hashes (sha256)

```
f015325a1e96f788533e9bfa3c1219b215c782eb0b826c0a0ed5f0d348bf25af  scripts/p1/cost_probe_d.py
cfa7db7776acaaa6c1ed987f92bc7e3e91fbadfe22c0de0b45274488dab60304  scripts/p1/run_cost_probe_d.sh
ac6fcf0cda8ca52bbba6e86dd3f0fcccfb11578712dc259d0b4bb060972000f1  scripts/p1/analyze_e0b.py   (unchanged since 5d411e8)
```
