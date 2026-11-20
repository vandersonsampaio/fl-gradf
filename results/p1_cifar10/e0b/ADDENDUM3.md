# Addendum 3 — FLTrust server-step diagnostic

**Date:** 2026-10-30 15:19 -0300, before the diagnostic runs. The plan (`PLAN.md`) and the E0b results (`RESULTS.md`, commit `87dd9e7`) do not change. Outputs go to `results/p1_g0_probe/`.

## Why

In this code the FLTrust server update g₀ is computed by the same `fit()` as the clients, i.e. by `local_epochs` passes over the root (`src/fl/attacked_learner.py`, `_server_model(...).fit(root)`). With batch 32 the server takes ceil(|root| / 32) × epochs SGD steps per round:

| | root 100 | enlarged root | clients (median over the partition) |
|---|---|---|---|
| CIFAR-10, 2 epochs | 8 | 158 (2,500) | 106–664 at α = 0.05 |
| MNIST, 5 epochs | 20 | 470 (3,000) | 820 / 918 / 752 at α = 0.5 / 0.1 / 0.05 |

FLTrust rescales every client update to ‖g₀‖, so the aggregate's norm is at most ‖g₀‖. The root size therefore also sets the effective global step, and E1's root contrast (here and on MNIST) may mix the root's content with the server's step budget.

**Specification.** FLTrust (Cao et al., NDSS 2021, arXiv:2012.13995 v3, read on 2026-10-30 before any probe result): clients and server run the same `ModelUpdate(w, D, b, β, R_l)` (Alg. 1–2), R_l mini-batch SGD iterations with the same b and β; default R_l = 1, b = 32 (64 on CIFAR-10), |D₀| = 100; Eq. (3) normalizes each client update to ‖g₀‖. The paper does not use epochs. This code's clients train by epochs, so "the same number of iterations as the clients" needs a mapping; the candidate, fixed here before any result, is **server steps = median over clients of ceil(n_i / 32) × epochs, computed from the partition only**.

## What runs

`scripts/p1/run_g0_probe.sh 10` → `scripts/p1/g0_probe.py`, one FLTrust run per job, CPU, 1 thread each, seeds outside the confirmatory block 42–51. Per round it records ‖g₀‖, the server step count, the aggregate's L2 norm, every client update's L2 norm and the global accuracy.

- **MNIST, 2×2 plus the candidate mapping (504 jobs):** 3 α × 7 attacks × root {100, 3,000} × server steps {native, 20, 470, median} × seeds 0–2; the MNIST grid's settings (logistic model, 10 clients, 2 Byzantine, 15 rounds, 5 local epochs, lr 0.01), seeded as `exp9_dominance_grid --seed_rng`. "native" is the unchanged code path; a fixed N draws N mini-batches of 32 from the root with the server RNG and makes one pass (FLTrust's `ModelUpdate`); clients are unchanged.
- **CIFAR-10, ‖g₀‖ only (4 jobs):** α = 0.05 × {`sign_flipping`, `trim_attack`} × root {100, 2,500}, native, seed 0, 20 rounds, seeded as `e0_pilot.py`, so these repeat the first 20 rounds of the matching E0b block-C jobs.

**Checks before this hash:** a smoke test (MNIST, 2 rounds, the three server-step modes). Native mode reproduces the seeded P-1 MNIST runs bit for bit: seed 42, α 0.5 `trim_attack` and α 0.05 `fltrust_aligned`, roots 100 and 3,000, final accuracies equal to `results/p1_invariance/raw/root{100,3000}_seed42` (4/4); that check ran from a scratch copy with the seed guard removed, outside the repository. Expected wall time: ~10 min (MNIST ≈ 9 s per job, CIFAR-10 ≈ 5–6 min per job).

## How it is read (descriptive; no test)

1. CIFAR-10: the 20 accuracy values must equal E0b's first 20 rounds (regression); report ‖g₀‖ and the aggregate norm per root.
2. MNIST, per cell and over seeds 0–2: native root contrast D_native = FLTrust(3,000, native) − FLTrust(100, native); root contrast at fixed steps D_N = FLTrust(3,000, N) − FLTrust(100, N) for N ∈ {20, 470, median}; step contrast at fixed root. Reported per cell, by α, and as the mean over the 21 cells (and the 18 without `fltrust_aligned`).
3. If D_N is much smaller than D_native at matched steps, the MNIST root effect is largely a step-budget effect.

**The decision on correcting the server update is taken from the specification above, not from these numbers.** The probe sizes the consequences (which results must be rerun: the MNIST root sensitivity, LOSO d* and headroom, E0b block C, σ_E0^E1) and records the epoch-based contrast that would be kept as a sensitivity analysis.

## Code hashes (sha256)

```
4b1a894f081fa2498bf5652d03cb39a7ce698e3e154bef5fa26a93615c8794c3  scripts/p1/g0_probe.py
914b5f5b5cc77bae6b8a81d41df3747db754ada732141a562899be689cc55145  scripts/p1/run_g0_probe.sh
```
