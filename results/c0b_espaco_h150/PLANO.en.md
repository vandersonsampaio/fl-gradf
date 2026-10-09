> English translation of `PLANO.md`. The Portuguese original is the frozen record (its hash is in `PLANO.sha256`); if the two ever disagree, the original prevails.

# Plan — C0b: remaining headroom at H = 150 against the best existing method

**Status:** FINAL before any run (hash in `PLANO.sha256`). **Descriptive**, with a criterion fixed now. The code does not exist yet: it will be written, tested and have its hash recorded in an addendum **before** the launch, without changing anything in this plan.
**Date:** 2026-10-02
**Scope:** defines the R1 targets for P3.

## 1. Motivation

C0 had two defects:
1. it compared the oracle with `sr_only`, the **worst** skeleton variant (B2.6);
2. it used **15 rounds**, during which the ceiling does not converge: the FedAvg-8 oracle gains 8–10 p.p. up to H = 150 at α ≤ 0.1 (B2.7).

B2.7 indicates that, at H = 150, a static rule closes `label_flipping` α ≤ 0.1.

## 2. Grid

- **Cells:** C0's **19 valid ones** (3 α × 7 attacks, minus `fltrust_aligned` at α 0.05 and 0.1, an attack artifact).
- **Nested horizons** 15 / 50 / **150**, as in B2.7. The primary is H = 150.
- **Seeds:** **72–81** (new, reserved for C0b).
- **Regime:** identical to C0/B2.6/B2.7: logistic MNIST, 10 clients, Byzantine [0, 1], root 100.
- **Systems (8):**
  - **skeleton variants** (the B2.6 learner, `scripts/b26_decomposicao.py`): `sr_only` (reference), `sr_bin`, `sr_b025`, `cosserver_only`;
  - **static rules** (applied as in B2.7, `PlainRuleLearner` from `scripts/b27_horizonte.py`: same updates, `server_update` on the root every round): FLTrust, Trimmed-Mean, Clustering, Median.
- **Ceilings** per α × seed: FedAvg-10 without attack and the FedAvg-8 oracle (only the 8 honest clients, evaluated on the 10 test sets), as in B2.7.
- **RNG:** `keras.utils.set_random_seed(seed)` before each system and each ceiling (B2.7's fix, PLANO §6.3).
- **Total:** 19 × 10 × 8 = 1,520 runs + 60 ceilings = **~1,580 runs**, ~25–35 CPU hours with 10 processes (`nice 19`, 2 threads, no GPU).

## 3. Criterion and outputs (fixed now)

**Metric:** accuracy of the global model at the end of round H, mean over the 10 clients' test sets (the same as C0). **Unit:** seed, n = 10.

**(i) Remaining-headroom map at H = 150 (main):**
- **Best existing method per cell** = the system with the highest mean accuracy among the 8, chosen in hindsight. It is optimistic for the existing methods, i.e. conservative for "headroom".
- **gap_global** = FedAvg-8 oracle − best existing, paired by seed.
- **Cell with headroom:** gap_global > **2 p.p.** with the whole CI95 above 0 (the C0 criterion).
- **Also reported:**
  - the same gap with FedAvg-10 as the ceiling;
  - **the gap against the reference skeleton** (`sr_only`);
  - the gaps at H = 15 and 50 (trend).

**(ii) Signal-selection ceiling at H = 150** (repeats A0(b) with the C0b data), mean over the 19 cells per seed:
- per-seed max between `sr_only` and `cosserver_only` (biased upwards);
- per-cell signal choice, in-sample;
- per-cell choice leaving the seed out (LOSO, **reference**);

all compared with the best single signal per seed and with each system.

**(iii) Skeleton × static rules table:** per cell and H, the best skeleton variant against the best static rule (Δ, CI95).

## 4. Check

Repeat one job (cell `label_flipping` α 0.05, seed 72) in two processes. The accuracies of the 8 systems must be identical. This test also covers determinism, since C0b does not use the DQN.

## 5. Reading (expectation, not a criterion)

- **Against the best global method:** almost no cell with headroom, which confirms Gate C0. In that case, P3's differentiator lies in R3/R4/R5.
- **Against the reference skeleton and AdaAggRL:** large headroom on label and backdoor attacks.
- **The signal-selection ceiling at H = 150** sizes P3 direction 1.

## 6. Rules

- Pre-written analysis (script with its hash in the addendum), run once with the complete grid.
- No accuracy is inspected before the grid ends. Monitoring reports only health and progress.
- If B2.7c is launched (conditional, B2.7a/b), C0b is paused and resumed afterwards. The launcher is resumable.
- Logs not versioned.
