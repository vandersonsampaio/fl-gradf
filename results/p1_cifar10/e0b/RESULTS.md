# E0b results (CIFAR-10 P-0 parameters)

Plan: `PLAN.md` (frozen, commit `8c9b28c`); analysis code and launch: `ADDENDUM1.md` (commit `5d411e8`); part D cost probe code: `ADDENDUM2.md` (commit `e6f087d`). Runs: 2026-10-30 10:49–14:49, 6 processes (deviation declared in addendum 1), 27/27 jobs, 0 failures, 180 rounds, 2 local epochs, seeds 0–2, CPU, 1 thread per process, deterministic ops. Analysis: `scripts/p1/analyze_e0b.py` (sha256 `ac6fcf0c…`, unchanged since addendum 1) → `analysis.txt` (sha256 `7df30cbb…`), `decisions.json`.

**The budget decision (Sec. 5) is provisional**: part D has not run yet, so it uses the plan's provisional costs. Secs. 1–4 do not depend on the costs and are final.

## Sec. 1 Regression check

All 21 jobs that repeat an E0 configuration match E0 exactly in each of the first 50 rounds (0 mismatches). Running 6 processes instead of E0's 8 changed no result.

## Sec. 2 Number of rounds: R = 110 (rule met)

No-attack FedAvg, seed mean of the 5-round trailing mean (%). The rule needs S(r+20) − S(r) < 1 p.p. and S(r+30) − S(r+10) < 1 p.p. for all three α; the first candidate where it holds is r = 110.

| α | r50 | r110 | gain r110→r130 | gain r120→r140 | r150 | r180 |
|---|---|---|---|---|---|---|
| 0.5 | 57.0 | 61.2 | +0.41 | −0.72 | 61.7 | 61.4 |
| 0.1 | 48.1 | 51.7 | +0.28 | −0.32 | 52.0 | 51.3 |
| 0.05 | 42.2 | 47.5 | +0.99 | +0.66 | 48.6 | 48.4 |

α = 0.05 is the binding cell, and it is close to the limit (+0.99 p.p.). Between r110 and r180 no α gains more than 0.9 p.p. Across-seed SD at r110 (5-round trailing mean): 0.4 / 0.9 / 1.5 p.p. for α 0.5 / 0.1 / 0.05.

## Sec. 3 GRADF Layer-1 threshold: τ_CNN = 7.86

Honest update L2 norms over all 180 rounds (block A: all clients; blocks B and C: clients not in `byzantine_ids`):

| block | n | median | p99 | max | share > τ |
|---|---|---|---|---|---|
| A FedAvg, no attack | 16,200 | 0.51 | 0.81 | 1.63 | 0 |
| B FedAvg, attack | 8,640 | 0.58 | 3.19 | 5.24 | 0 |
| C FLTrust, attack | 17,280 | 0.52 | 1.61 | 2.49 | 0 |

Largest honest norm by α: 1.16 (0.5), 1.36 (0.1), 5.24 (0.05). τ_CNN = 1.5 × 5.24 = 7.86 (MNIST: 5.0). The maximum comes from block B (FedAvg under attack at α = 0.05), the conservative worst case the plan names: honest clients whose global model the attack has degraded produce larger updates. Under no attack the threshold sits 4.8× above the largest honest norm, so Layer 1 rejects no honest update but is loose against attackers whose norm is not inflated. As the plan says, the honest-rejection rate is a diagnostic in the confirmatory block, with no re-tuning.

## Sec. 4 σ_E0 at R = 110 (final-round accuracy, unsmoothed)

| statistic | per-seed D_s (p.p.) | σ_E0 | 0.5 σ_E0 |
|---|---|---|---|
| E1: mean over 2 cells of FLTrust(2500) − FLTrust(100) | +0.70, −0.61, −0.65 | 0.77 | 0.38 |
| E3 (approximation): mean over 2 cells of FLTrust(100) − FedAvg | −7.19, −7.73, −5.38 | 1.23 | 0.62 |

Both sensitivity margins (0.5 σ_E0) are narrower than the primary 1.0 p.p. margin, so the primary margin is the binding one.

**Limitation of σ_E0^E1 (applies the rule as written; no change):** the small σ comes from averaging two cells whose root contrasts are large and of opposite sign, not from low per-cell noise. Per cell at R (seed mean, unsmoothed): −7.7 p.p. under `sign_flipping` (per seed −13.9, −14.2, +4.9) and +7.4 p.p. under `trim_attack` (+15.3, +13.0, −6.2). In the confirmatory E1, D_s averages over the grid's cells; whether the cell-level contrasts cancel there as they do over these two cells is unknown, so σ_E0^E1 may understate the across-seed SD of D_s. The pre-registration should state this, and report the per-cell contrasts alongside the TOST.

## Sec. 5 Compute budget (provisional)

With the plan's provisional costs, C(110, k): level 0 (k = 18) 194.6 h, level 1 (k = 12) 159.7 h, level 2 (k = 6) 124.8 h, against a budget of 236 h. **Provisional decision: level 0 (3 α × 6 attacks), R = 110, not truncated.** Level 0 has a margin of 41 h (17%). Part D (`ADDENDUM2.md`) replaces the costs and reruns this decision; it needs ~28 GB of free RAM, so it runs after W1 finishes.

## Sec. 6 Descriptive

Accuracy at R = 110 (%, mean ± SD over 3 seeds, unsmoothed; 5-round trailing mean at r180 in brackets), α = 0.05:

| rule | no attack | sign_flipping | trim_attack |
|---|---|---|---|
| FedAvg | 47.3 ± 1.9 [48.4] | 38.4 ± 1.0 [38.1] | 26.7 ± 1.7 [27.1] |
| FLTrust, root 100 | | 35.1 ± 6.7 [36.2] | 16.4 ± 11.1 [20.9] |
| FLTrust, root 2500 | | 27.3 ± 4.3 [23.1] | 23.8 ± 2.1 [20.6] |

- **The larger root does not help FLTrust here.** Under `sign_flipping`, root 2,500 is below root 100 in 2 of 3 seeds and at r180 (23.1 vs 36.2). This differs from MNIST, where FLTrust's advantage grew with the root (`references/sensibilidade_root_fltrust.md`). Two cells and three seeds; descriptive only. E1 tests this question on the full grid.
- **FLTrust with root 100 under `trim_attack` is bimodal:** seeds 0 and 1 end near chance (10.9, 9.3%), seed 2 at 29.2%. This is the source of the large per-cell SDs above.
- **Ranking stability is limited.** FedAvg − FLTrust(100) at α = 0.05: under `sign_flipping`, +3.3 p.p. at R and −0.2 at r180; under `trim_attack`, +10.3 at R and +7.3 at r180, with one seed of opposite sign at both rounds. Rankings within a cell at 3 seeds are not stable, as the plan anticipated.
- Wall time per round (6 processes): 15–18 s per run, against 18–19 s with 8 processes in E0.

## For the P-0 decision and the confirmatory pre-registration

1. R = 110 (final).
2. τ_CNN = 7.86 (final).
3. σ_E0^E1 = 0.77 and σ_E0^E3 = 1.23 p.p. (final), with the limitation in Sec. 4.
4. Grid level 0 at R = 110 (provisional until part D).
