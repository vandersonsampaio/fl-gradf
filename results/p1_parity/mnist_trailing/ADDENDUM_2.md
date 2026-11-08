# Addendum 2 — AdaAggRL on MNIST with the trailing mean

**Date:** 2026-11-04, before these runs. **Decided by the author on 2026-11-04.**

**Why:** `RESULTS.md` has no trailing mean for AdaAggRL, since it was not in `../mnist_rq3`. Without it, E3 on MNIST either mixes metrics or leaves a system out.

**Not blind.** AdaAggRL's final-round results on these seeds are known from the published runs.

## Runs

`scripts/p1/run_mnist_adaaggrl.sh`: `exp10`, both extractors (`random` and `pretrained`), seeds 42–51, 21 cells, root 100, 15 rounds, `--seed_rng`, `--server_steps median`, the settings of `../mnist_rq3`. 20 jobs, 2 processes, on the PC alongside W1 and the campaign.

## Differences from the published AdaAggRL runs

- **Seeded.** The published runs are unseeded, so the final-round column is not expected to match them. The difference is reported and is not read as an effect of anything.
- **`server_steps = median`.** Under `fltrust_aligned` the published rows hold the native g₀.

## Reading

E3 is recomputed with the rows of `RESULTS.md` and these, as AdaAggRL − Random and AdaAggRL − B₋ₛ, in the same Holm families, on 18 and on 12 cells, on the trailing mean and on the final round. The extractor `pretrained` is the sensitivity arm. The new families replace those in `RESULTS.md`, whose Holm values change since they had one system fewer.

## Code hashes (sha256)

```
e7f1974d9c9528cc9e77cfe75d0148091db2b78d49b52b252fbf0608124bb060  scripts/p1/run_mnist_adaaggrl.sh
ef13ce4028db135d4093403ae1217ab2ce9a7c604644440e0354d45512206ecb  src/experiments/exp10_selector_comparison.py
7d726b5c9eee6c1db7cc311f502d626610bc2d4ead0be24d6843cdd72074c19d  results/models/adaaggrl_pretrained_extractor_mnist.weights.h5
```
