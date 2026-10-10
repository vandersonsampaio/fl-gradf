# Result — B3.3: threshold a₅ sweep in the official AdaAggRL, BloodMNIST (EB)

**Date:** 2026-10-10
**Plan:** `PLANO.md` (`ab42d47`; hash `0917a0a8…` in `PLANO.sha256`, unchanged; English translation in `PLANO.en.md`). **Exploratory.**
**Addendum 1:** runner test passed; code hashes recorded before the launch. `analisar_b33.py`, `run_b33.py` and `run_grid_b33.sh` checked against those hashes before the analysis (identical).
**Grid:** 15/15 runs (a₅ ∈ {0.475; 0.75; 0.95} × seeds 145–149; BloodMNIST, q = 0.5, EB, 500 rounds), from 2026-10-09 17:19 to 2026-10-10 14:35. Queue interleaved by seed, 6 processes on the GPU.
- **Execution:** no failures and **no run repeated** (PLANO §6).
- **Window:** launched at 17:19 on a Friday as an author's exception; the controller was active from 18:00 and had no pause to apply before the grid ended (weekend).

**Analysis:** `scripts/passo2_oficial/analisar_b33.py`, run once with the complete grid. Before using each run, it checks that the recorded `a_fixed` is [0.475]×4 + [a₅]. It produced `analise.txt`, `resumo_runs.csv` and `delta_vs_centro.csv`.

---

## 1. Criterion (PLANO §5)

Δ = metric(a₅) − metric(center a₅ = 0.475), paired by seed, n = 5, t CI95 (4 d.f.). Wilcoxon is descriptive only (minimum two-sided p with n = 5 is 0.0625).

| a₅ | metric | Δ (p.p.) | CI95 | Wilcoxon p |
|---|---|---|---|---|
| 0.75 | **primary (median 401–500)** | **−0.64** | (−2.11; +0.82) | 0.31 |
| 0.75 | AUC 1–500 | +0.84 | (−0.56; +2.23) | 0.19 |
| 0.75 | AUC 251–500 | +0.05 | (−2.17; +2.28) | 1.00 |
| 0.95 | **primary (median 401–500)** | **−0.14** | (−5.14; +4.87) | 1.00 |
| 0.95 | AUC 1–500 | −0.05 | (−0.56; +0.47) | 0.63 |
| 0.95 | AUC 251–500 | +0.58 | (−1.36; +2.51) | 0.31 |

→ **VERDICT: NO CONSTANT BEATS THE CENTER** in this sweep.
- Neither a₅ has Δ > 2 p.p. with CI95 > 0 on the primary. The reset caveat therefore does not apply.
- Neither a₅ is "higher threshold is worse" either (no Δ < −2 p.p. with CI95 < 0).
- The AUCs, which are less sensitive to the timing of a single reset, agree: every |Δ| is below 1 p.p.
- The wide CI95 for a₅ = 0.95 on the primary comes from two seeds that move in opposite directions (147: 38.1%; 149: 27.6%). The AUC 1–500 for the same condition is tight (±0.5 p.p.).
- No multiplicity correction (2 comparisons), as declared.

**Termination clause (PLANO §6):** the §1 question is closed as **"there is no simple constant above the center that fixes the regime"**, within the resolution of n = 5. No other sweep is opened.

## 2. Mechanism (descriptive)

Mean across the 5 seeds:

| a₅ | primary | AUC 1–500 | resets per run | median interval between resets | runs with a reset in 401–500 | clients excluded per round | attackers excluded per round | weight mass on attackers |
|---|---|---|---|---|---|---|---|---|
| 0.475 (center) | 32.0% | 30.7% | 20.8 | 23.0 rounds | 5/5 | 5.38 | 1.25 | 0.016 |
| 0.75 | 31.4% | 31.5% | 15.8 | 29.5 rounds | 5/5 | 5.98 | 1.45 | 0.016 |
| 0.95 | 31.9% | 30.7% | 10.0 | 41.8 rounds | 4/5 | 7.30 | 1.64 | 0.020 |

- **The threshold acts on the mechanism as expected:** a higher a₅ excludes more clients per round (5.4 → 7.3 of 10) and more of the attackers (1.25 → 1.64).
- **Resets become rarer:** with a₅ = 0.95 every seed has fewer resets than the center (7–13 vs. 19–24), and the interval between them almost doubles. They do not disappear: 4/5 runs still reset within rounds 401–500.
- **Accuracy does not move:** all three conditions stay at ~31%, the same degraded level as B3.1 (fixed, EB: 32.7%; FedAvg without attack: 77.8%).
- **The weight mass on attackers does not fall** (0.016–0.020) even though more attackers are excluded: the extra exclusions also remove honest clients, and what remains still lets weight through to attackers.

The reading is exploratory: filtering more aggressively spaces the resets but does not change the operating level. The degradation under EB is not explained by a threshold that is too low.

## 3. Consequences

- **PLANO §7 (criterion not met):** neither the central action nor higher thresholds defend BloodMNIST under EB in the official environment. P2 reports this as a **limit of the published mechanism on that dataset**, not as a tuning problem.
- Together with B3.1/B3.2: on BloodMNIST, TD3 adds nothing to the fixed action (B3.1), the policy does not move (B3.2), and no constant threshold above the center would have been a better target for it to find (B3.3). The "RL at most tunes a constant" reading from B2.7b does not gain new support here, because there is no better constant to tune towards within this range.
- **Caveats when citing B3.3:**
  1. Exploratory, n = 5, no multiplicity correction: it rules out large gains (> 2 p.p. with CI95 > 0), not small ones.
  2. EB only; LMP was not swept.
  3. Only values above the center were tested (values below 0.25 collapse on MNIST, B2.4r).
  4. The official code on GPU is not bit-reproducible; pairing by seed pairs the configuration, not the trajectory.

## 4. Files

- `analise.txt`: full output of the pre-written analysis.
- `resumo_runs.csv`: per run, the primary, the AUCs, resets, median interval between resets, reset in 401–500, weight mass on attackers, and excluded clients/attackers per round.
- `delta_vs_centro.csv`: Δ vs. the center with t CI95 and descriptive Wilcoxon, per a₅ and metric.
- `raw/a5_<value>/`: the 15 final JSONs and the observed states (`obs/`). There are no actor checkpoints (fixed action only). Logs and partial checkpoints are not versioned (PLANO §8).
