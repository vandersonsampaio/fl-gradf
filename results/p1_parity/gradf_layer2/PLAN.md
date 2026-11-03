# Plan — splitting Layer 2 into clipping and noise, MNIST under step parity

**Date:** 2026-11-03. **Status: frozen before the grid, after the checks of Sec. 4 passed (`checks.txt`).** Changes only through dated addenda. **Authorized by the author on 2026-11-03.** It runs while the W1 rerun is paused, so nothing overlaps.

**Why:** the GRADF ablation (`../gradf_ablation/RESULTS.md`) found Layer 2 to be the largest effect on fixed FLTrust (+5.18 p.p. as a main effect; F010 − F000 = +8.72 p.p.), but it had one switch for clipping and noise. Whether Layer 2 can be presented as a defense depends on which part acts. The hypothesis stated there is that the noise lowers the server's effective step, since it is added to client updates and not to g₀.

**Status of the evidence:** re-analysis on seeds already seen, MNIST only. It informs the P1 pre-registration and is not confirmatory.

**Seen before freezing:** the checks ran 4 arms on seed 42, α 0.05, in 2 cells (`checks.txt`). In those cells the noise-only arm is close to F010 and the clipping-only arm is close to F000. The reading rule below was written as designed before those runs and was not changed after them.

## 1. Questions

- **Q1 (primary):** is the gain of Layer 2 on fixed FLTrust due to the noise, to the clipping, or to both?
- **Q2 (descriptive):** does the mechanism fit a smaller effective step?

## 2. Arms

All with fixed FLTrust, Layers 1 and 4 off, and the configuration of the ablation: exp10 seeded, root 100, `server_steps = median`, 21 cells, 15 rounds, seeds 42–51, `GroundTruthClassifierStub`.

| arm | clipping (±2.0 per coordinate) | Laplace noise (scale 1/50 per coordinate) | purpose |
|---|---|---|---|
| F000 | off | off | reference with the new logging; must equal the ablation |
| C010 | on | off | clipping alone |
| N010 | off | on | noise alone |
| F010 | on | on | reference with the new logging; must equal the ablation |

**4 arms × 21 cells × 10 seeds = 840 runs.** F000 and F010 are rerun only because the ablation did not log the norms, and they double as a regression check.

**Not in this plan:** an arm that shrinks the server's step without noise. Its factor should come from the norms measured here, so it would need its own addendum.

## 3. Implementation

- **`scripts/p1/gradf_layer2_split.py`**, with no change to `src/` or to `scripts/p1/gradf_ablation.py`. A subclass of `AblatedPipeline` switches clipping and noise separately. All arms share a seed, as in the ablation, and noise is drawn in the same order as in `HardeningPipeline`.
- **Logged per round:** ‖g₀‖; mean ‖update‖ of honest clients before Layer 2; ‖Δ‖ of the aggregate; the step along g₀, ⟨Δ, g₀⟩ / ‖g₀‖²; mean trust of honest and of Byzantine clients; the fraction of coordinates above the clipping bound.
- **Launcher:** `scripts/p1/run_gradf_layer2.sh [N]`: one job per seed, resumable, a job starts only with at least 6 GB available, CPU, 1 thread per process. Outputs in `raw/`.

## 4. Checks before freezing (seed 42, α 0.05, `sign_flipping` and `gaussian_noise`)

1. F000 and F010 equal the ablation exactly in both cells.
2. Clipping alone clips and draws no random numbers. Noise alone adds the same draw and does not clip.
3. All fields are written, and every round votes FLTrust and is accepted.

If a check fails, the grid does not start and the author is told.

## 5. Regression during analysis

F000 and F010 must equal `../gradf_ablation/raw` in all 420 pairs. Any mismatch is reported, not explained away, and the attribution below is then not made until it is understood.

## 6. Reading rule

On the 18 confirmatory cells, per seed, using `inference.describe` (CI95 for superiority or inferiority, CI90 inside ±1.0 p.p. for equivalence). The 21-cell values are reported alongside.

Contrasts: N010 − F000, C010 − F000, F010 − N010, F010 − C010.

- **Noise:** N010 − F000 is superior and F010 − N010 is equivalent.
- **Clipping:** C010 − F000 is superior and F010 − C010 is equivalent.
- **Both:** N010 − F000 and C010 − F000 are both superior, whether or not the equivalences hold.
- **Anything else:** inconclusive. It is not settled by more runs without a new plan.

**Q2 (descriptive, no test).** Per arm, by α and by attack: means of the logged quantities, over all rounds and over the last 5. The hypothesis fits if the step along g₀ under N010 is well below F000 while ‖g₀‖ and the honest clients' norm are similar. This is consistent with the hypothesis, and it does not prove that a smaller step causes the gain. The contrasts are also reported per α and per attack.

**Not attributed.** The result concerns FLTrust without Layers 1 and 4 on MNIST's logistic model, about 7,850 parameters. It does not extend by itself to other rules, to GRADF or to the CIFAR-10 CNN.

## 7. What it decides

A proposal for the author, which does not change on its own any frozen plan:
- **Noise:** Layer 2 should not be presented as a defense by itself. The next step is an addendum with a noise-free arm that shrinks the step by a factor taken from these norms. The CIFAR-10 CNN should be checked separately, since the noise's norm grows with the dimension.
- **Clipping:** Layer 2 can be presented as a clipping defense, with the fraction of clipped coordinates reported.
- **Both or inconclusive:** both parts stay in the hardened arms, and the attribution stays open in the manuscript.

## 8. Cost

Measured in the checks with the machine free: about 10.3 s per run, 1 thread.

| | jobs | runs per job | wall time |
|---|---|---|---|
| 10 processes, W1 paused | 10 | 84 | **about 15–25 min** |

Total CPU: about 2.5–4 h. Analysis and write-up: about 1 h.

## 9. Rules

- Seeds 42–51 only. Seeds 62–71 and 200–209 stay reserved, and 220–239 belong to W1.
- W1 is not resumed by this plan. If the author resumes it before the grid ends, the overlap is reported.
- No commit without the author's request.

## Code hashes (sha256)

```
62f57d1f60888b718f14315ef65d6044432cf78797674f0713ac377bfa2db117  scripts/p1/gradf_layer2_split.py
e4b68132b893d7c7b9b627574eecb6779390cc15a71737fc3ea83a19f8db2d76  scripts/p1/run_gradf_layer2.sh
764c8a8226fe28217ec05d68d5a3bea6d879607682f2352f89e0e52349f6cf23  scripts/p1/gradf_ablation.py
```
