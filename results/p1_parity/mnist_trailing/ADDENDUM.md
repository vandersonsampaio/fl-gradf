# Addendum — MNIST rerun recording the trailing mean

**Date:** 2026-11-03. **Status: frozen after the checks below passed (`checks.txt`) and before the grid.** **Decided by the author on 2026-11-03,** to run after `../factor_attack_check` ends, 6 processes, alongside W1.

**Why:** the MNIST tables use the final round only. FLTrust at factor 1 oscillates by 9.39 p.p. per round in training error with no attack (`../lr_tuning/RESULTS.md`), so a final-round comparison depends on the round at which training stops. The author decided that MNIST uses the mean of the last 5 rounds (rounds 11–15), as on CIFAR-10.

**Not blind.** The final-round results on seeds 42–51 are known (`../RESULTS.md`). The metric is changed for the instability above, on results of seeds 0–2 and of block C, and not to reach a given outcome on these seeds. This is stated in the paper. The final-round values stay recorded and are reported alongside.

## Runs

The jobs of `../mnist_fixed` and `../mnist_rq3` as in `../PLAN.md`, items 1 and 2, with the same commands, seeds 42–51, `server_steps = median`, factor 1 (adopted, `../factor_attack_check/ADDENDUM.md`): 30 + 20 jobs. Only the output folder changes, and the raw tables gain the column `accuracy_trailing5`.

## Code change

`round_accuracies` in `src/experiments/exp9_dominance_grid.py`, used by `exp9` and `exp10`, returns the final accuracy and the trailing mean. The column `accuracy` keeps the final round, so no existing reader changes.

## Checks before freezing (`checks.txt`, all PASS)

The final accuracy equals the earlier run bit for bit, and the trailing mean is recorded:
1. `exp9`, seed 42, root 100, α 0.05, `trim_attack` and `sign_flipping`, FLTrust and trimmed mean.
2. `exp10`, seed 42, α 0.05, `trim_attack`, Random.

## Reading

**Free regression check over the whole grid.** The column `accuracy` must equal `../mnist_fixed` and `../mnist_rq3` exactly, except for systems using GRADF's DQN, which is not deterministic across executions (`../gradf_repeat`). Mismatches are reported, not explained away.

The tables, LOSO, d\*, headroom and E3 capture are then recomputed on the trailing mean, with the same procedures as `../RESULTS.md`. E3 is also computed on the 12 cells of level 1, the author's decision for the common confirmatory grid.

## Cost

Earlier wall times: `mnist_fixed` 3 h 38 min on 8 processes alongside other work; `mnist_rq3` 2 h 45 min on 10 processes. At 6 processes alongside W1: **about 8 to 12 h,** more than the 4 to 5 h stated earlier to the author.

## Code hashes (sha256)

```
d411c4773d682e9d56980f9e20d5f9dafe1e415b316db9772d3d9beef2968f04  src/experiments/exp9_dominance_grid.py
ef13ce4028db135d4093403ae1217ab2ce9a7c604644440e0354d45512206ecb  src/experiments/exp10_selector_comparison.py
123023d4de18f278590033f008a4455265882083c569048f377e38a3a2e1d4f3  scripts/p1/run_mnist_trailing.sh
```
