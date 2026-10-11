# Addendum 1 — reformulated B2.7c: code at freeze

*English translation of `ADENDO1.md`. The Portuguese original is the frozen document (hash in `ADENDO1.sha256`); if the two differ, the original prevails.*

**Date:** 2026-10-11 11:18:56 -0300, before any run on seeds 82–91. The plan (`PLANO.md`, hash in `PLANO.sha256`) does not change.

## Implementation

- **`scripts/b27c_b025_vs_td3.py run`:** calls `_reseed(seed)`, then builds the ablation learner in td3 mode.
  - For `fixed_b025`, `select_action` returns the constant [0.5]×4 with b = 0.25 (via `scripts/b27b_explicativo.constant_for`, unchanged), and `train_step` is a no-op.
  - This is the same path as B2.7b.
- **`analisar`:**
  - one-sided paired t per cell;
  - Holm over the 3 cells (α = 0.05);
  - the verdict from PLANO §2, the pooled H1b test and the descriptive statistics.

  It was tested on **synthetic** data in the scratchpad, outside the repository. The p-values and the Holm decision were checked by hand.
- **Launcher `scripts/run_grid_b27c.sh`:** 60 jobs on 12 CPU processes; resumable.

## Check (PLANO §3): 15-round smoke test, seed 42, `label_flipping` α 0.05

- td3_ref: new 0.719400 vs reference 0.719400, |Δ| = 0.00e+00 OK
- fixed_b025: new 0.725400 vs reference 0.725400, |Δ| = 0.00e+00 OK

## Code at freeze
4ca49c72f1972891f5c361847481971240b9334a70f389488077a4c286de2d8e  scripts/b27c_b025_vs_td3.py
f93c1c51e166be007cb88a9c7041dcb2e0fb798da2228539539ee50ad401cedf  scripts/run_grid_b27c.sh
7b4c3fd2d8692e9156c938031f2548a71ceb482e5e9fc6f7d28067734d893d34  scripts/b27b_explicativo.py
c5310907f80fee535196b22cac3d5e2d71929d9c9cf9987b773a87cd5ae8df3c  scripts/b27_horizonte.py
4525cd80e2d0b9ec70be759f4323e7285cbf4feac6107e89e93ec20a20de2e41  scripts/frente1_ablacao_adaaggrl.py
