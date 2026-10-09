> English translation of `ADENDO1.md`. The Portuguese original is the frozen record (its hash is in `ADENDO1.sha256`); if the two ever disagree, the original prevails.

# Addendum 1 — B2.7a: code at freeze time

**Date:** 2026-10-02 20:36:26 -0300, before any B2.7a run. The B2.7b code will have its own addendum, also before its launch. The B2.7b criterion, fixed in `PLANO.md`, does not change.

## Implementation

- `scripts/b27b_td3_constante.py b27a`: runs `td3_ref` (the ablation learner, td3 mode) with `_reseed(seed)` first, as in B2.7.
- Logging is done by wrappers of `agent.select_action` and `learner._run_round` that only copy values and do not consume RNG: mean state and executed action every round, and `W_actor`/`b_actor` after 0, 50, 100 and 150 completed rounds. `train` numbers rounds starting from 1, and this was corrected in the snapshot before the freeze.
- **Smoke** (15 rounds, `label_flipping` α 0.05, seed 42): accuracy identical to B2.7's `td3_ref` at H = 15 (|Δ| = 0).
- The launcher `scripts/run_grid_b27a.sh` runs 30 jobs, with 10 CPU processes.

## Code at freeze time
d4c600290e5999b14919fd921750c0c6fd81b9782adb57ffab6a3f83ee2a6458  scripts/b27b_td3_constante.py
57923a053b402e7418c58710d086c4abe5be81a732d309b6c5b4dafc6968fdda  scripts/run_grid_b27a.sh
