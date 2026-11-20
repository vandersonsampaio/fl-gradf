# Plan — E0b: rounds, GRADF Layer-1 threshold and σ_E0 (CIFAR-10, before the P-0 decision)

**Status:** frozen by hash (`PLAN.sha256`) before the first E0b run. Any change after the hash enters only as a dated addendum.
**Context:** `results/p1_plan/PLAN.md` (Sec. 5, E0 and gate P-0); E0 results in `../e0/RESULTS.md`. E0 left three quantities that P-0 needs but whose rules were not fixed in advance: the number of rounds R (FedAvg had not stabilized by round 50), GRADF's Layer-1 threshold on the CNN (D5 in `../DEVIATIONS.md`), and σ_E0. This plan fixes the three rules, and the campaign's compute budget, before any E0b data exist.

**What was already seen:** E0 (seeds 0–2, 50 rounds) and the cost probe (`../e0_bench/`), which includes honest update norms on the CNN for rounds 1–3 at α = 0.5 only (median 0.73–0.81, max 1.08). No run beyond round 50 and no norm beyond round 3 has been observed.

## 1. Runs

CIFAR-10, the small CNN (34,522 parameters), 10 clients, Byzantine [0, 1], local lr 0.01, **2 local epochs**, **180 rounds**, **seeds 0–2** (outside the confirmatory block 42–51). Seeded as in E0 (`e0_pilot.py`: crc32 run seed of `e0|seed|α|attack|rule|epochs`, which does not depend on the root size, so both roots are paired; deterministic ops; NumPy mini-batch order; separate server RNG). CPU, 1 thread per process.

| Block | Rule | Attack | α | Root | Jobs |
|---|---|---|---|---|---|
| A. Curves | FedAvg | none | 0.5, 0.1, 0.05 | 100 | 9 |
| B. Norms under attack | FedAvg | sign_flipping, trim_attack | 0.05 | 100 | 6 |
| C. Root contrast | FLTrust | sign_flipping, trim_attack | 0.05 | 100 and 2,500 | 12 |
| | | | | | **27** |

Every job saves the per-round global accuracy and the L2 norm of every client update at every round (`update_l2`, rounds × clients; the Byzantine clients are listed in `byzantine_ids`). Root 2,500 is the whole CIFAR-10 `server_val`.

Launcher: `bash scripts/p1/run_e0.sh e0b 8` (8 processes, ~2 GB each), outputs in `raw/`. Estimated wall time ≈ 4 h (180 rounds × ~19 s per round at 8 processes, 4 waves). Runs only while W1 is paused or stopped; inside the Mon–Fri 8–18 window only with the author's explicit authorization.

**Regression check (before any analysis):** 21 E0b jobs repeat an E0 configuration (blocks A and B, and C with root 100). Their first 50 rounds must equal E0's bit for bit (global accuracy at every round). Any mismatch stops the analysis and is investigated.

## 2. Rule for R

Let S_α(r) be the FedAvg no-attack accuracy (block A) averaged over the 3 seeds and smoothed over rounds r−4 … r (5-round trailing mean). Candidate rounds r ∈ {20, 30, …, 150}.

- **R** = the smallest candidate r such that, for **all three α**, both
  S_α(r + 20) − S_α(r) < 1.0 p.p. **and** S_α(r + 30) − S_α(r + 10) < 1.0 p.p.
  (the second condition guards against a premature stop by noise: the seed mean has an SD of ~0.7 p.p. in E0).
- **R_max = 150**, the largest candidate the 180-round horizon can evaluate. If no candidate satisfies the rule, R = 150 and the campaign is reported as "not stabilized at R = 150" (deviation from P-0).
- R is then subject to the budget (Sec. 5). The budget is met by shrinking the common confirmatory grid first, never R, except at the floor (Sec. 5). Reason: the ranking of the rules changes between rounds 30 and 50 in E0, so truncating R changes what is measured (d\*, O(c)); fewer cells narrow the scope but leave each cell valid.

## 3. Rule for the Layer-1 threshold

GRADF's Layer 1 rejects an update whose L2 norm exceeds an absolute threshold τ (5.0, calibrated on MNIST; the skewness bound is unchanged). On MNIST (logistic, seed 0, 30 rounds, no attack, α 0.5/0.1/0.05) the largest honest norm is 3.4/4.0/3.5, so τ = 5.0 sits 1.25–1.47× above it and no honest update is rejected.

- **τ_CNN = 1.5 × the largest honest update L2 norm** observed over all 27 E0b jobs, all 180 rounds, all three α. Honest = every client in block A, clients 2–9 in blocks B and C.
- The ratio 1.5 is fixed here, before the E0b norms exist. It is the upper end of the MNIST range, chosen to err towards not rejecting honest updates.
- Blocks B and C include honest norms under attack, with the global model degraded by FedAvg under `trim_attack`. GRADF degrades the model less than that, so this is a conservative (worst-case) calibration.
- In the confirmatory block, GRADF's honest-rejection rate per cell is reported as a diagnostic. τ is **not** re-tuned on it.

## 4. Definition of σ_E0

σ_E0 is the across-seed SD of the **per-seed paired difference aggregated over cells**, i.e. the same statistic the confirmatory test is applied to, read at round R (final-round accuracy, unsmoothed, as in the confirmatory analyses):

- **E1 family (root dependence):** D_s = mean over the 2 cells (α = 0.05; `sign_flipping`, `trim_attack`) of [Acc_FLTrust(c, s, 2,500) − Acc_FLTrust(c, s, 100)]. This equals E1's V(c, s, 2,500) − V(c, s, 100), because the trimmed mean does not use the root. σ_E0^E1 = SD over seeds 0–2 of D_s.
- **E3 families (approximation):** no adaptive system exists in any pilot. σ_E0^E3 = SD over seeds of the per-seed mean over the same 2 cells of [Acc_FLTrust(c, s, 100) − Acc_FedAvg(c, s)], a paired difference between fixed rules of the same kind, at no extra cost.
- **Limitations (stated, both make the sensitivity margin more conservative):** the aggregate covers 2 cells instead of 18–21 (averaging over more cells lowers the SD), and the difference for E3 is between fixed rules, not adaptive systems. With 3 seeds (2 df) σ_E0 is itself imprecise.
- σ_E0 is used only where the P1 plan uses it (the sensitivity margin 0.5·σ_E0); the primary margin of 1.0 p.p. is unchanged.

## 5. Compute budget of the campaign

**Budget: 236 h** of wall-clock on this machine (12 processes × 1 thread, CPU), counted outside the Mon–Fri 8–18 window (~118 h per week, so about 2 calendar weeks), for E1 + E3 + E3 sensitivities. E4 is outside this budget.

**Cost model** (h, wall-clock, 12 × 1), with R rounds and k cells in the common confirmatory grid:

C(R, k) = E1(R) + [E3(R) + Sens(R)] · k / 18

- E1(R) = 1,680 runs (7 rules × 21 cells × 10 seeds at root 100, plus FLTrust × 21 × 10 at root 2,500) × R × s_fixed / 12.
- E3(R) = Σ over the 5 systems of 180 runs × R × s_sys / 12, plus the pretraining of GRADF and of the attack-aware reference.
- Sens(R) = AdaAggRL with the pretrained extractor and FedStrategist with GRADF's detector, on the same grid.
- s_fixed, s_AdaAggRL: measured at 12 × 1 (`../e0_bench/`, ×0.8 for the memory fix): ≈ 21 s and ≈ 35 s per round. Provisional values for the systems never timed on the CNN (Random, attack-aware, GRADF, FedStrategist: ≈ 23 s per round; pretraining ≈ 20 h in total) give C(R, 18) ≈ 1.65·R + 20 h.
- **Cost probe (part D, before the decision):** Random, attack-aware and GRADF (both including pretraining) and FedStrategist are timed on the CNN at 12 × 1 (α 0.5, seed 0, 3 timed rounds after a warm-up). The measured values replace the provisional ones in C(R, k). This probe yields costs only, no outcome data; its code and hash are recorded in an addendum before it runs.

**Decision rule** (applied once R is known, with the measured costs):

| Level | Common confirmatory grid | k |
|---|---|---|
| 0 | 3 α × 6 attacks (all but `fltrust_aligned`) | 18 |
| 1 | 3 α × {`sign_flipping`, `label_flipping`, `trim_attack`, `krum_collusion`} | 12 |
| 2 (floor) | α ∈ {0.5, 0.05} × {`sign_flipping`, `label_flipping`, `trim_attack`} | 6 |

1. Use the lowest level whose C(R, k) ≤ 236 h. E1/E2 always run the full 21-cell grid.
2. If even level 2 exceeds the budget, R is lowered to the largest multiple of 10 with C(R, 6) ≤ 236 h, reported as a deviation ("R truncated by budget").
3. If the grid is reduced, d\*, B(c) and O(c) for E3 are recomputed on the reduced grid (P1 plan, P-0).

The reduced levels keep blind (`sign_flipping`, `label_flipping`) and informed (`trim_attack`, `krum_collusion`) attacks, and drop the two attacks that carry the least information on MNIST (`gaussian_noise`, weak; `low_mag_backdoor`, whose MNIST version was found degenerate in W2). They were chosen without CIFAR-10 data beyond E0.

With the provisional costs, level 0 fits up to R ≈ 130, level 1 up to R ≈ 160, so with R ≤ 150 the floor is not expected to be needed.

## 6. Descriptive outputs (not decisions)

- Curves to round 180 per α; the round at which the rule in Sec. 2 is met.
- Ranking stability after R: only the pair FedAvg × FLTrust (root 100) under the two attacks at α = 0.05, sign and size of the difference at R and at 180. This is a limited descriptive check, not a measure of the 7-rule ranking.
- FLTrust root contrast per cell and per seed at R (feeds σ_E0^E1 only; it is not an E1 result).
- Honest norms per round, α and block; the share of honest updates above τ_CNN in each block (zero by construction for the maximum, reported for completeness).

## 7. Order

1. Hash this plan (`PLAN.sha256`); commit on the author's request.
2. Smoke test of the modified `e0_pilot.py` (2 rounds, seed 3, its output moved out of `raw/` at once), then run blocks A–C (27 jobs).
3. Regression check (Sec. 1), then apply Secs. 2–4 with `scripts/p1/analyze_e0b.py` (written before the runs end, hash in an addendum).
4. Part D cost probe; then the budget rule (Sec. 5).
5. Record R, τ_CNN, σ_E0, the grid level and the costs in `RESULTS.md`; they feed the P-0 decision and the confirmatory pre-registration.
