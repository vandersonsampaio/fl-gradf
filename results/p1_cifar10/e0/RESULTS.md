# E0 results (CIFAR-10 feasibility pilot, gate P-0)

Plan: `results/p1_plan/PLAN.md`, Sec. 5, E0. Descriptive only: 3 seeds (0–2), outside the confirmatory block 42–51. Small CNN (34,522 parameters), 10 clients, 2 Byzantine, root 100, local lr 0.01, CPU, 1 thread per process, deterministic ops. Runs: 2026-10-30 08:07–10:04, 8 processes, 60/60 jobs, 0 failures (`run_e0.sh`, `e0_pilot.py`). Analysis: `scripts/p1/analyze_e0.py` → `analysis.txt`, `curves_summary.csv`, `separation_summary.csv`, `separation_pairs.csv`, `curves.png`. Costs: `../e0_bench/SUMMARY.md`.

The stabilization rule and the separation check below were defined at analysis time (they are not in the plan); they are descriptive.

## E0.1 Rounds and local epochs (FedAvg, no attack)

Accuracy (%) smoothed over the last 5 rounds, mean ± SD over 3 seeds:

| α | local epochs | r30 | r50 | gain r30→r50 (p.p.) | s/round (8 procs) |
|---|---|---|---|---|---|
| 0.5 | 1 | 45.1 ± 0.1 | 50.8 ± 0.6 | +5.6 | 10.4 |
| 0.5 | 2 | 51.7 ± 1.1 | 57.0 ± 1.5 | +5.3 | 18.8 |
| 0.1 | 1 | 36.1 ± 1.4 | 41.9 ± 0.8 | +5.8 | 10.3 |
| 0.1 | 2 | 43.1 ± 1.2 | 48.1 ± 0.8 | +5.0 | 19.0 |
| 0.05 | 1 | 32.7 ± 1.5 | 39.1 ± 0.5 | +6.4 | 10.5 |
| 0.05 | 2 | 37.3 ± 1.6 | 42.2 ± 1.1 | +4.8 | 15.0 |

**Finding: accuracy has not stabilized by round 50 in any cell** (still +5 to +6 p.p. between rounds 30 and 50; the "stable round" in `analysis.txt`, 45–48, only reflects the end of the window). P-0's rule "rounds are fixed where no-attack FedAvg accuracy stabilizes" cannot be applied from these curves. Two local epochs reach at round ~30 what one epoch reaches at round 50, at about the same compute.

## E0.2 Separation of the fixed rules at α = 0.05 (2 local epochs)

Round 50, accuracy (%), mean ± SD over 3 seeds; paired by seed (same partition):

| rule | sign_flipping | trim_attack |
|---|---|---|
| fedavg | 31.9 ± 0.2 | 25.8 ± 1.6 |
| fedprox | 30.6 ± 2.2 | 22.1 ± 2.7 |
| clustering | 28.6 ± 2.8 | **32.3 ± 0.8** |
| fltrust (root 100) | 22.3 ± 2.8 | 14.2 ± 9.6 |
| trimmed_mean | 18.7 ± 1.5 | 15.8 ± 2.0 |
| median | 16.7 ± 0.6 | 15.1 ± 0.4 |
| krum | 10.0 ± 0.0 | 17.7 ± 6.7 |
| pooled across-seed SD | 1.8 | 4.7 |
| pairs whose paired 95% CI excludes 0 | 14/21 | 8/21 |

- The rules separate beyond the across-seed noise under `sign_flipping` (14/21 pairs) and partly under `trim_attack` (8/21; FLTrust and Krum vary by 7–10 p.p. across seeds). The best rule differs by attack (FedAvg vs. clustering), so there is headroom for selection.
- No rule is at the floor without attack (FedAvg 42.2%), but under attack the coordinate-wise robust rules (median, trimmed mean) sit at 15–19%, within 5–9 p.p. of chance. Krum collapses to chance under `sign_flipping`, as on MNIST at α = 0.05 (`exp9`: 9.8%).
- P-0's replacement rule for α = 0.05 ("near the floor **and** the rules do not separate") is not triggered.

## σ_E0 (across-seed SD, root mean square over cells)

| round | separation cells (14) | FedAvg no-attack, 2 epochs (3) |
|---|---|---|
| 30 | 2.7 p.p. | 1.3 p.p. |
| 50 | 3.5 p.p. | 1.2 p.p. |

With σ ≈ 3.5 p.p., the E1 sensitivity margin 0.5·σ_E0 ≈ 1.8 p.p. is wider than the primary 1.0 p.p.; with 10 seeds the primary TOST will often not close ("no evidence", as the plan foresees).

## GRADF Layer-1 threshold (D5 in `../DEVIATIONS.md`)

Honest update L2 norms, no attack, root 100, seed 0:

| model | rounds 1–3 median | late-round median | max | threshold 5.0 exceeded |
|---|---|---|---|---|
| MNIST logistic, α 0.5 / 0.1 / 0.05 (30 rounds, trimmed mean) | 1.65 / 1.85 / 1.67 | 0.60 / 0.97 / 0.94 (r28–30) | 3.4 / 4.0 / 3.5 | never |
| CIFAR-10 CNN, α 0.5 (3 rounds, `e0_bench`) | 0.73–0.81 | not measured | 1.08 | never |

On MNIST the threshold sits 1.25–1.5× above the largest honest norm. On the CNN, 5.0 is ~4.6× above it, so Layer 1 never rejects an honest update but is looser against attackers than on MNIST. Keeping MNIST's ratio would give a threshold of ≈ 1.4–1.6 (5.0 × CNN/MNIST max ratio) or ≈ 2.3 (ratio of early medians). The CNN norms are known only for rounds 1–3 at α = 0.5; the per-round norms at every α over the chosen number of rounds are still needed before fixing it.

## Open for the P-0 decision

1. Rounds: the curves must be extended (e.g. 100–150 rounds) to locate the plateau, or a fixed budget declared as a deviation from P-0 ("not converged at R"); the campaign cost grows linearly with R (the 30-round estimates in `../e0_bench/SUMMARY.md` scale by R/30).
2. Local epochs: 1 vs 2 (similar compute per accuracy point).
3. α = 0.05: kept (P-0 replacement rule not triggered).
4. Layer-1 threshold: measure CNN honest norms over the chosen rounds at the 3 α.
5. AdaAggRL cost: measured (`../e0_bench/SUMMARY.md`); the grid-reduction decision depends on R.
