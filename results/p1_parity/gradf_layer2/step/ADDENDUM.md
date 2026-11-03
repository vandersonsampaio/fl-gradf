# Addendum — a noise-free arm that shrinks the server's step

**Date:** 2026-11-03. **Status: frozen before the grid, after the checks below passed (`checks.txt`).** **Authorized by the author on 2026-11-03.** It runs while the W1 rerun is paused, so nothing overlaps. Re-analysis on seeds already seen, MNIST only, not confirmatory. It extends `../PLAN.md`, Sec. 7.

**Why:** the Layer 2 split (`../RESULTS.md`) read as noise: N010 − F000 = +8.72 p.p., with the step along g₀ falling from 0.338 to 0.113. That fits a smaller effective step, but it does not show that the step causes the gain.

**Question:** does a smaller server step, without any noise, reproduce the gain of the noise?

## Arm

**S010:** fixed FLTrust, Layers 1 and 4 off, no clipping, no noise, and Layer 3's aggregate multiplied by **c = 1/3**. All else as in `../PLAN.md`, Sec. 2: 21 cells, 15 rounds, seeds 42–51. **210 runs.** F000 and N010 are reused from `../raw`.

**The factor** is the ratio of the mean steps along g₀ over all rounds and cells, 0.113 / 0.338 = 0.335, rounded to 1/3. It was taken from N010 and F000 on these same seeds, so this arm is tuned on the data it is compared with. The factor is constant, although under noise the step falls every round (0.37 to 0.05). A schedule is not in this addendum.

## Checks before freezing (seed 42, α 0.05, `sign_flipping`)

1. With factor 1 the arm equals F000 exactly.
2. Layer 2 returns the update unclipped and draws no random numbers.
3. The first round's step along g₀ is exactly 1/3 of F000's, and every round votes FLTrust and is accepted.

## Reading rule

On the 18 confirmatory cells, per seed, using `inference.describe`. The 21-cell values are reported alongside. Contrasts: S010 − F000 and N010 − S010.

- **Explained by the step:** S010 − F000 is superior and N010 − S010 is equivalent (CI90 inside ±1.0 p.p.).
- **Step alone goes further:** S010 − F000 is superior and N010 − S010 is inferior.
- **Partly explained:** S010 − F000 is superior and N010 − S010 is superior.
- **Not supported by this arm:** S010 − F000 is not superior.
- **Anything else:** inconclusive, including S010 − F000 superior with N010 − S010 in no evidence.

**Limit fixed in advance:** a miss does not rule out the step, since this factor is constant and the noise's effect on the step is not. A schedule would need its own addendum.

Descriptive: contrasts per α and per attack; the logged step, ‖Δ‖ and trust per round.

## Cost

Measured: about 20 s per run under load, 21 runs per job. 8 processes, two waves: **about 15 minutes.**

## Code hashes (sha256)

```
a816d9833c140a3db2fc69cf93f60abc46b8ddecdce263a8fd30a703cfd711f8  scripts/p1/gradf_layer2_step.py
2068a908accd8e79c9cf328e428ceeb546f7838469a36a646e4c430a3b79a55a  scripts/p1/run_gradf_layer2_step.sh
62f57d1f60888b718f14315ef65d6044432cf78797674f0713ac377bfa2db117  scripts/p1/gradf_layer2_split.py
```
