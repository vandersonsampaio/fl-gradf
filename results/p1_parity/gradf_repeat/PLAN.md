# Plan — repetition of GRADF on MNIST under step parity

**Date:** 2026-11-02, before these runs. Changes only through dated addenda.

**Why:** in `../RESULTS.md` (Sec. 4), GRADF − Random is +3.48 p.p. in a single execution of a non-deterministic system, and GRADF is above the cell-wise best fixed rule in 16 of 21 cells. Before the pre-registration, this checks whether the result repeats and records what the pipeline decides each round.

**Authorized by the author on 2026-11-02** to run alongside the W1 rerun, an exception to "P1 runs do not overlap W1", with half of the planned parallelism: **5 processes** instead of the 10 of `mnist_rq3`.

## Runs

`bash scripts/p1/run_gradf_repeat.sh 5` → `scripts/p1/gradf_repeat.py`. CPU, 1 thread per process.

- Same configuration as item 2 variant b: exp10 seeded, root 100, `server_steps = median`, 21 cells, 15 rounds.
- Systems: Random and GRADF only. Each system reseeds itself from (seed, α, attack, system), so the others can be left out.
- **2 new executions of seeds 42–51: 20 jobs.** With the first run, that makes 3 executions. Execution 1 runs entirely before execution 2.
- Expected: about 200 s per cell, about 70 min per job, about 5 h in all; possibly more under load.

**Instrumentation.** Per round, reading only: voted rule, Layer-4 acceptance, clients rejected by Layer 1, Layer-3 errors, old and new accuracy on the root, global accuracy. It uses no RNG.

**Check before this plan.** One cell (seed 42, α 0.05, `sign_flipping`): Random and GRADF both equal the first run to the CSV's precision (0.5816, 0.7982). That GRADF matches here says nothing about the other cells; it voted FLTrust in all 15 rounds.

## Reading rule

**Regression:** Random must equal `../mnist_rq3` in all 210 cell × seed pairs. Mismatches are reported, not explained away.

**Repetition.** For each execution e, Δ_e,s = mean over the 18 confirmatory cells of GRADF − Random for seed s. Executions 1 and 2 use the new Random, the first run its own. The result is **robust to run-to-run variance** only if both hold:

1. In all 3 executions the CI95 of the mean over seeds is above zero.
2. As in D2 (`fl-fedmed/results/d2_variancia_p1`), the SD across executions of the aggregate Δ is below ½ |mean aggregate Δ|.

Otherwise it is **not robust**, and +3.48 p.p. is not used as a finding. GRADF − B₋ₛ is reported per execution, with the same fixed rules. **Comparison with published runs is not attributed to the correction** (`../PLAN.md`, notes).

**Mechanism (descriptive, no test).** Per cell and system: share of rounds per voted rule, Layer-4 rejection rate, Layer-1 rejections by honest or Byzantine client, and GRADF against fixed FLTrust in the cells where GRADF votes FLTrust nearly always. GRADF and Random share Layers 1, 2 and 4, so a difference between them comes from the votes, while a difference from a fixed rule may come from those layers. This plan does not separate the layers. An ablation, if wanted, needs its own plan.

## Rules

- This run informs the P1 pre-registration; it is not confirmatory.
- RAM: a job starts only with at least 6 GB available. W1 is not touched.
- Outputs in `raw/`. No commit without the author's request.

## Code hashes (sha256)

```
54e2d6c16eb64c1db44b5689b334909be22559a1fab93d9ef58018358b703598  scripts/p1/gradf_repeat.py
17436a07e14c2fa7199591e9ddcd006f0cc6c5876ea7c90c8cd66fbcde7da995  scripts/p1/run_gradf_repeat.sh
```
