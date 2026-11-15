# Front 1 under FLTrust step parity — result

**Date:** 2026-11-15. **Status: descriptive check** (`PLAN.md`; `PLAN.sha256` checked out at analysis time). Analysis: `scripts/f1_server_parity.py analyze` → `analysis.txt`, `auc.csv`.

**Short answer.** Front 1's server update (20 steps on the root) understated `cos_server`. Under step parity (752–917 steps), both pre-fixed readings involving it change:
- **Step 1 (Q1, affected):**
  - `cos_server`'s mean AUC rises from 0.704 to 0.776 over 21 cells, and from 0.821 to 0.905 over the 18 cells without `fltrust_aligned`.
  - It moves from below S_R (0.745) to above it, and it becomes the best detector of the step.
- **Grid (Q3, changes):**
  - `cosserver_only` with parity beats `sr_only` by +2.34 p.p. (90% CI +1.96 to +2.73), beyond the ±1 p.p. margin.
  - With the native update it was at `sr_only`'s level (+0.14).
  - "Practically at the level of `sr_only`" no longer holds. Under parity, the root-based detector without gradient inversion is better.
- **`fltrust_aligned` (Q2):**
  - With the native update, the attacker's clone has 13–20% of the honest norm.
  - Under parity it has 1.4–2.1 times the honest norm, because a 750–900-step fit on 100 samples overfits the root.
  - The attack changes character: from a small clone to a large clone that `norm` and `l2_median` detect (AUC 0.76–1.00).
  - `cos_server` still cannot detect it (AUC ≈ 0) in either version, since the attacker copies the server direction by construction.

## Checks

- The detector variant reproduces all 21 of Front 1's stored step-1 CSVs exactly in Front 1's columns (max |diff| = 0). Before the freeze:
  - the native `cosserver_only` reproduced 2 stored accuracies exactly;
  - the parity learner replaced the score's server fit once per round.
- The grid's native difference `cosserver_only − sr_only` is +0.0014 (90% CI −0.0079 to +0.0100), the value Front 1 reported.
- 234/234 runs, no failures: detector 21, attack 3, grid 210.

## Q1 — step 1, AUC along the stored td3 trajectory (seed 42)

| detector | mean AUC, 21 cells | mean AUC, 18 cells (no FA) | cells with AUC ≥ 0.70 |
|---|---|---|---|
| **`cos_server_parity`** | **0.776** | **0.905** | **15** |
| S_R | 0.745 | 0.772 | 12 |
| S_cg | 0.716 | 0.741 | 11 |
| S_lg | 0.713 | 0.720 | 11 |
| `cos_server` (native, Front 1) | 0.704 | 0.821 | 13 |
| S_cl | 0.661 | 0.636 | 13 |
| `l2_median` | 0.633 | 0.733 | 11 |
| `norm` | 0.613 | 0.714 | 10 |
| `cos_median` | 0.465 | 0.499 | 6 |

- The gain is broad: `gaussian_noise` goes from 0.62–0.69 to 0.84–0.92; `krum_collusion`, `label_flipping`, `sign_flipping` and `trim_attack` reach 0.94–1.00 in every cell.
- `low_mag_backdoor` is the exception: 0.42/0.65/0.60 against the native 0.40/0.73/0.68.
- **Reading (fixed):** Δ = +0.072 > 0.03, and `cos_server` changes position relative to S_R, so the step-1 reading is **affected**.

## Q2 — `fltrust_aligned` with the attacker cloning the parity update

| α | Byzantine/honest norm, native | under parity | AUC `norm` | AUC `l2_median` | AUC S_R | AUC `cos_server_parity` |
|---|---|---|---|---|---|---|
| 0.05 | 0.200 | 1.719 | 0.90 | 0.87 | 0.78 | 0.00 |
| 0.1 | 0.201 | 1.386 | 0.83 | 0.76 | 0.69 | 0.00 |
| 0.5 | 0.127 | 2.065 | 0.97 | 1.00 | 0.60 | 0.00 |

- The 21-cell means with these 3 cells replaced are `cos_server_parity` 0.776, S_R 0.760, `l2_median` 0.754 and `norm` 0.740.
- Under parity, the server's root update has a median norm of about 1.86 on seed 42, against honest updates near 1. A step count matched to the clients over 100 samples drawn with replacement overfits the root. The attack therefore becomes a large-norm clone.
- Neither version is the "honest-looking" clone the attack's design suggests. That would need the server update rescaled to the honest norm, which neither P1 nor P2 does.

## Q3 — grid, `cosserver_only` (21 cells × seeds 42–51)

| comparison | mean | 90% seed-bootstrap CI |
|---|---|---|
| `cosserver_only` (native) − `sr_only` | +0.0014 | (−0.0079; +0.0100) |
| **`cosserver_only` (parity) − `sr_only`** | **+0.0234** | (+0.0196; +0.0273) |
| `cosserver_only`: parity − native | +0.0221 | (+0.0138; +0.0303) |

**Per attack, minus `sr_only`:**

| attack | native | parity |
|---|---|---|
| `fltrust_aligned` | −0.037 | −0.037 |
| `gaussian_noise` | −0.044 | −0.031 |
| `krum_collusion` | −0.015 | +0.002 |
| `label_flipping` | +0.092 | +0.135 |
| `low_mag_backdoor` | +0.017 | +0.032 |
| `sign_flipping` | +0.014 | +0.053 |
| `trim_attack` | −0.019 | +0.011 |

The grid keeps the native attack reference, so `fltrust_aligned` does not change.

**Reading (fixed):** the CI lies entirely above +0.01, so "practically at the level of `sr_only`" **changes**: with parity, `cosserver_only` is better.

## Consequences for P2 (for the author)

- **Front 1's conclusions on `cos_server` were measured with an under-trained server update.** Under parity:
  - the cheapest detector, cosine to the root update with no gradient inversion, is the best detector of step 1;
  - it beats `sr_only` on the grid by +2.3 p.p.;
  - the Front 2 trade-off ("data reconstruction is traded for a root set") tilts further toward the root, which is where the LGPD question sits.
- **`fltrust_aligned` in every P2 experiment** (B2.x, C.0, W1–W3, Front 1) is a small-norm clone (13–20% of the honest norm). It is not the attack its name suggests, and under parity it would be a large-norm clone. Neither is norm-matched.
- **Limits:**
  - one seed for step 1 and the attack variant;
  - MNIST and the logistic model only;
  - R = 15, as Front 1;
  - the grid is descriptive, with no Holm correction;
  - the `cosserver_only` variant was already exploratory in Front 1, chosen after the step-1 AUCs.
