# Addendum 3 — MNIST: the 7-rule reference set and the hardened reference

**Date:** 2026-11-15, before the hardened runs and before any output of this analysis. **Requested by the author on 2026-11-15.**

**Why:** on CIFAR-10, GRADF and Random are compared with the hardened d\*, and X5 showed that the 4-rule reference set D_R understates the best fixed rule. On MNIST, GRADF − B₋ₛ = +4.85 p.p. (+3.76; +5.95) on 18 cells (trailing mean) is measured only against the plain d\* over D_R, which prevents claiming absence of superiority on both datasets.

**Status: descriptive re-analysis, not blind.** Seeds 42–51 and every existing MNIST result are known.

## Runs

`scripts/p1/mnist_hardened.py`: FLTrust, median, trimmed mean and Krum through GRADF's Layers 1, 2 and 4, built exactly like exp10's Random system in `mnist_rq3`.
- Settings: logistic model, R = 15, 5 local epochs, `local_lr` 0.01, `server_steps = median`, root 100.
- The random selector is replaced by a fixed one.
- Each run is seeded with crc32("{seed}|{alpha}|{attack}|{rule}_hardened"), as in the CIFAR-10 hardened block.
- 4 rules × 21 cells × 10 seeds = 840 runs, on this PC.

**Check, passed before this freeze:** the Random system run through this same path reproduces the stored `mnist_rq3` runs exactly (seed 42, α = 0.1, 7 cells, max |diff| = 0).

## Analysis (fixed now, run once: `scripts/p1/analyze_mnist_addendum3.py`)

Metric: the trailing mean, as in `RESULTS.md`. No Holm correction.

- **A, D_7:**
  - Ĥ_LOSO over D_7 next to D_R on 21, 18 and 12 cells, with d\* and the cell winners. The 7 rules at root 100 already exist in `mnist_fixed`.
  - Every system against B7_−s (plain), with `inference.describe` and the seed bootstrap that repeats the selection.
  - Every system against B_R,−s (plain), which must reproduce `RESULTS.md`.
- **B, hardened:**
  - Every system against B_H,−s, the cross-fitted best hardened rule of D_R, on 18 and 12 cells.
  - The hardened reference against the plain one.

## Reading fixed now

The primary row is GRADF on 18 cells, against the hardened reference (B) and against B7_−s (A).
- **GRADF's CI95 lies entirely above 0 against the hardened reference:** its MNIST superiority survives the reference used on CIFAR-10. The paper cannot claim absence of superiority on both datasets and states the MNIST superiority as such.
- **Otherwise** (no evidence, equivalence or inferiority): GRADF's MNIST superiority depends on the reference. The paper reports it next to the hardened contrast and may state that no system is superior against the hardened reference on either dataset.
- **B7_−s** is reported alongside with the same reading. A disagreement between `describe` and the bootstrap on 0 is reported as inconclusive.

## Code hashes

See `ADDENDUM_3.sha256` (this file, `scripts/p1/mnist_hardened.py`, `scripts/p1/analyze_mnist_addendum3.py`).
