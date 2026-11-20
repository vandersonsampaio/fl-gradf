# Addendum — pilot of a smaller FLTrust step on CIFAR-10

**Date:** 2026-11-03. **Status: frozen before the grid, after the checks below passed (`checks.txt`).** **Authorized by the author on 2026-11-03.** It runs while the W1 rerun is paused, so nothing overlaps. Pilot on seeds outside the confirmatory block, descriptive: it confirms nothing.

**Why:** on MNIST, multiplying FLTrust's aggregate by 1/3 without noise reproduces the gain of Layer 2's noise, and FLTrust at factor 1 swings about 10 p.p. per round (`../gradf_layer2/step/RESULTS.md`). On CIFAR-10, FLTrust swings about 8 p.p. per round under parity and is below FedAvg (`../RESULTS.md`, Sec. 5). If the two have one cause, the campaign's E1 and E3 would compare everything with a poorly scaled reference.

**Question:** does a smaller step stabilize FLTrust on the CNN?

## Runs

Block C of `../PLAN.md`, item 3: FLTrust, `server_steps = median`, α 0.05 × {`sign_flipping`, `trim_attack`} × root {100, 2,500} × seeds 0–2, with the aggregate multiplied by **c = 1/3**, the MNIST factor, not tuned here. **12 runs, R = 110.** The reference is block C itself at rounds 1–110, same seeds and streams. FedAvg comes from E0b's raw runs. `scripts/p1/cifar_step_pilot.py`, no change to `src/`.

**Rounds:** 110, the campaign's R. Block C ran 180, so gains after R are not available for this arm.

## Checks before freezing

1. With factor 1, two rounds equal block C exactly, under both attacks (seed 0, root 100).
2. With factor 1/3, the first round's step along g₀ is 1/3 of factor 1's.

## Reading rule

Per attack × root (4 cells), seed means. Stability is the mean absolute change in accuracy between consecutive rounds, rounds 51–110, as in `analyze_parity.py`. Accuracy is the trailing mean of rounds 106–110 (`../ADDENDUM1.md`).

- **Stabilizes:** in all 4 cells, stability at factor 1/3 is at most half of block C's.
- **Does not stabilize:** in none of the 4 cells.
- **Partly:** anything else.

Reported alongside, with no test (3 seeds): accuracy at factor 1/3 minus block C per cell and per seed; the root contrast (2,500 − 100); FedAvg under the same attack and seeds; final-round accuracy at R.

**Limits fixed in advance:** 2 cells of the 21, one α, 3 seeds, one factor. A smaller step also slows learning, and 110 rounds may be too few for it. Accuracy below block C at R would not show that the factor is worse at convergence.

## What it decides

A proposal for the author, which does not change on its own any frozen plan:
- **Stabilizes, with accuracy not lower:** whether E1 and E3 keep FLTrust at factor 1, add a scaled arm, or replace it. The cost is computed after that decision, and any factor must not be tuned on seeds 42–51.
- **Stabilizes, with accuracy lower:** the same decision, with the slower learning stated.
- **Otherwise:** the campaign keeps factor 1, and the instability is reported as a limit.

## Cost

Measured in the checks: about 10 s per round alone. With 4 processes, 3 waves of 110 rounds: **about 1 to 1.5 h.** 4 processes because the machine has about 13 GB available with W1's paused processes held in memory, at about 2 GB per process.

## Code hashes (sha256)

```
48413cec84e2edc418030f4efe3f098225f224c12fecbfa470911236ed502fd0  scripts/p1/cifar_step_pilot.py
ddde55bfd570ac514420f4fdda38bccb9adcec57834faa104820ac7883196f51  scripts/p1/run_cifar_step_pilot.sh
```
