# FLTrust server-step diagnostic: results

Plan: `results/p1_cifar10/e0b/ADDENDUM3.md` (what runs; registered 2026-10-30 15:19) and `ADDENDUM4.md` (how it is read; registered 15:22, before any output was read). Runs: 2026-10-30 15:19–15:46, 10 processes, CPU, 1 thread each, 508/508 jobs, 0 failures (`scripts/p1/run_g0_probe.sh`, `scripts/p1/g0_probe.py`, hashes as in addendum 3). Analysis: `scripts/p1/analyze_g0_probe.py` → `analysis.txt`, `per_cell.csv`. Descriptive only: seeds 0–2 (outside the confirmatory block), no test.

**Deviation in the analysis code:** its first run stopped with an error before printing or writing anything (a column named `agg` clashed with the pandas `Series.agg` method in attribute access). The column was renamed to `agg_l2` and nothing else changed. The sha256 is now `ad58a4c8be9e94c342929b3545a1b403486d1dbedfc323d95eab9e1f166d1491` (addendum 4 recorded `3ae01443…`).

## Specification

FLTrust (Cao et al., NDSS 2021, Alg. 1–2) runs the same `ModelUpdate` with R_l mini-batch iterations on the clients and the server (default R_l = 1). This code computes g₀ with the clients' `fit()`, i.e. by local epochs over the root, so the server's step count, and with it ‖g₀‖, to which every client update is rescaled, grows with the root size. The candidate correction, server steps = median over clients of ceil(n_i / 32) × epochs, **preserves the specification's step parity between server and clients**. It is not faithful to the paper's values (addendum 4, Sec. 1).

## CIFAR-10: ‖g₀‖ (α = 0.05, seed 0, native code path, 20 rounds)

The 20 accuracy values of all 4 jobs equal the first 20 rounds of the matching E0b block-C jobs (regression check, 4/4). Medians over rounds:

| attack | root | server steps | ‖g₀‖ | aggregate norm | client update norm |
|---|---|---|---|---|---|
| sign_flipping | 100 | 8 | 0.085 | 0.069 | 0.654 |
| sign_flipping | 2,500 | 158 | 0.471 | 0.212 | 0.561 |
| trim_attack | 100 | 8 | 0.058 | 0.036 | 0.724 |
| trim_attack | 2,500 | 158 | 0.472 | 0.222 | 0.667 |

With the native code, root 2,500 gives a global step 3–6× larger than root 100, and both are well below the clients' update norm. The E0b root contrasts on CIFAR-10 (`../p1_cifar10/e0b/RESULTS.md`, Sec. 4 and 6) therefore mix the root's content with the step size. Twenty rounds do not show whether the larger step explains the drop of root 2,500 under `sign_flipping`.

## MNIST: the 2×2 and the candidate mapping

Settings of the MNIST grid (logistic model, 10 clients, 2 Byzantine, 15 rounds, 5 local epochs, lr 0.01), 21 cells (3 α × 7 attacks) × root {100, 3,000} × server steps {native, 20, 470, median} × seeds 0–2. The native step counts are 20 (root 100) and 470 (root 3,000); the median mapping gives 820 / 918 / 752 steps at α = 0.5 / 0.1 / 0.05.

### Norms and accuracy (median over cells, seeds and rounds; accuracy is the mean final accuracy over cells and seeds)

| root | server steps | ‖g₀‖ | aggregate norm | client update norm | accuracy (%) |
|---|---|---|---|---|---|
| 100 | native (20) | 0.199 | 0.117 | 1.835 | 70.4 |
| 100 | 20 | 0.197 | 0.113 | 1.833 | 70.3 |
| 100 | 470 | 1.073 | 0.649 | 1.009 | 81.2 |
| 100 | median | 1.532 | 0.992 | 1.041 | 76.0 |
| 3,000 | native (470) | 0.661 | 0.240 | 1.020 | 86.6 |
| 3,000 | 20 | 0.179 | 0.103 | 1.875 | 70.7 |
| 3,000 | 470 | 0.661 | 0.248 | 1.021 | 86.6 |
| 3,000 | median | 0.846 | 0.366 | 0.971 | 86.8 |

Fixed-step mode at the native step count gives nearly the same results as the native path (mini-batches sampled with replacement instead of epoch permutations).

### Contrasts (p.p., seed mean; D_root@N = FLTrust(3,000, N) − FLTrust(100, N))

| quantity (addendum 4) | 21 cells | 18 cells without `fltrust_aligned` | cells > 0 (of 21) |
|---|---|---|---|
| native root contrast D_native | 16.2 | 16.5 | 21 |
| D_root@20 | 0.5 | −0.7 | 10 |
| D_root@470 | 5.5 | 4.3 | 21 |
| D_root@median (candidate correction) | 10.9 | 10.5 | 21 |
| **step effect** (main effect, fixed root) | **13.4** | **14.9** | |
| **residual root effect** (main effect, fixed steps) | **3.0** | **1.8** | |
| **interaction** (D_root@470 − D_root@20) | **5.0** | **5.0** | |

By α (21 cells):

| α | D_native | D_root@20 | D_root@470 | D_root@median | step effect | residual root effect | interaction |
|---|---|---|---|---|---|---|---|
| 0.5 | 9.0 | −0.5 | 2.4 | 3.6 | 8.2 | 0.9 | 3.0 |
| 0.1 | 21.2 | −1.0 | 5.6 | 14.7 | 18.9 | 2.3 | 6.5 |
| 0.05 | 18.6 | 2.9 | 8.5 | 14.4 | 13.2 | 5.7 | 5.6 |

Per-cell values: `per_cell.csv`.

## Reading (as fixed in addendum 4)

1. **The step effect dominates.** It is 13.4 p.p. (14.9 without `fltrust_aligned`), against a residual root effect of 3.0 (1.8). At 20 server steps, root 3,000 does no better than root 100 (70.7 vs 70.3%; 10/21 cells positive). The root dependence reported in the FGCS version (root 3,000 vs 100 with the native code) is mostly a step-budget effect. With the native code, FLTrust at root 100 behaves like a heavily damped FedAvg: the aggregate's norm is 0.12, against 1.8 for the client updates.
2. **The residual root effect is not negligible, and it grows with the step budget** (interaction 5.0 p.p.). Under the candidate step parity the root contrast is 10.9 p.p., positive in 21/21 cells, about two thirds of the native contrast. It is concentrated at α ≤ 0.1; at α = 0.5 it is 3.6 p.p. and the residual root effect is 0.9.
3. **Much of that residual is g₀ overfitting at the small root,** the confound addendum 4 named. Going from 470 steps to the median (820–918) lowers root 100 from 81.2 to 76.0% while root 3,000 stays at 86.6–86.8%, and ‖g₀‖ at root 100 grows from 1.07 to 1.53 (≈ 150 to ≈ 240–290 passes over 100 examples per round). Under step parity, a larger root helps mainly because a small root cannot sustain many steps without overfitting. This is not a "pure content" effect.

By the reading fixed in addendum 4, both effects are relevant: **RQ1 stands, with the caveat that FLTrust's root dependence is mostly mediated by the server's step budget, which epoch-based implementations couple to the root size.** The residual root effect under step parity mixes data diversity with g₀ overfitting at the small root.

**Limitations:** MNIST logistic model only, 15 rounds, 3 seeds. On CIFAR-10 only ‖g₀‖ was measured (1 seed, 20 rounds), not the accuracy contrast at fixed steps.

## Consequences (decision pending)

The decision on correcting the server update follows the specification (addendum 3), not these numbers. If the step-parity correction is adopted, with the epoch-based version kept as a sensitivity analysis, these must be rerun or recomputed:

- the MNIST root sensitivity (`references/sensibilidade_root_fltrust.md`) with the median mapping;
- the MNIST LOSO d\* and headroom (P-1);
- E0b block C on CIFAR-10 with server steps = median client steps (303 / 342 / 307 at α = 0.05 / 0.1 / 0.5, 2 epochs), and σ_E0^E1 from it.

The native-code results remain valid as the description of the epoch-based implementation.
