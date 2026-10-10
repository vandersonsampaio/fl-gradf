# Result — B2.3: TD3 steelman in the official AdaAggRL

**Date:** 2026-09-30
**Plan:** `PLANO.md` (English translation in `PLANO.en.md`; hash in `PLANO.sha256`; `PLANO.md`, `analisar_b23.py`, `analisar_b21.py` and `analisar.py` checked against the hashes before running: OK). Scheduling: `EXECUCAO.md`.
**Grid:** 10/10 runs (seeds 100–104 × LMP/EB; TD3 with lr 1e-3 and `learning_starts` 10), 2026-09-29 21:01 → 2026-09-30 16:47, 0 failures. Paired with Step 2's `fixed` and `td3`.
**Analysis:** `scripts/passo2_oficial/analisar_b23.py` → `analise.txt`, `resumo_runs.csv`, `mecanismo.csv`, `drift_por_checkpoint.csv`. **Exploratory** (already-used seeds).

---

## 1. Gate B-a (criterion fixed in the PLANO before the data)

| condition | result |
|---|---|
| 1. C1 steelman − fixed with Δ > 0 and p < 0.05 | **NO**: Δ = **−10.22 p.p.** (CI95 −24.3 to +3.8), Wilcoxon p = 0.084 |
| 2. the policy moved away from the initial one (drift_500 > σ_a = 0.0475) | **YES**: median 0.470 (~10× the noise) |
| 3. the policy uses its input (S_swap > σ_a) | **NO**: median 0.0011 |

→ **VERDICT: the steelman does NOT beat the fixed action. The P2 thesis is strengthened.**

Detail per attack (primary, median 401–500):
- **LMP:** steelman ≈ fixed (Δ = −0.09 p.p.) and ≈ td3 (Δ = −0.23 p.p.). The action is irrelevant under LMP, as in Step 2.
- **EB:** Δ = −20.3 p.p. vs. fixed, driven by **two collapses** (seeds 100 and 103: 0.476 and 0.436). In the other 3 seeds the steelman stays at 0.88–0.97.
- C2 (steelman − official td3): Δ = −10.4 p.p., Wilcoxon p = 0.002. The steelman is **worse** than the published configuration.
- Extra resets under EB: steelman 15.8 per run, vs. td3 2.8 and fixed 0.6.

## 2. Mechanism: the policy learns a constant corner action (exploratory)

- **The drift grows fast and saturates:** median 0.10 at step 50, 0.31 at 100 and 0.47 from step 200 on. The policy reaches the edge of the Box in ~200 rounds and stays there.
- **The final action is a corner of the Box [0; 0.95]^5, the same for any state:** the standard deviation of the action over the states of rounds 401–500 is between 0.0001 and 0.02. Examples: LMP s100 = [0.95; 0.95; 0; 0.95; 0], EB s103 = [0; 0; 0; 0; 0], EB s104 = [0; 0; 0; 0.95; 0.95].
- **The collapse under EB coincides with a threshold a₅ ≈ 0:** EB s100 (a₅ = 0.00) and EB s103 (a₅ = 0.00) collapsed, with 38 and 36 resets. With δ = max(k)·a₅ ≈ 0, the filter only excludes the lowest-scoring client, and the (boosted) EB attackers enter the aggregation. When the learned corner has a high a₅ (EB s102: 0.94; s104: 0.95) or an intermediate one (s101: 0.25), the defense works. **Under LMP**, even a₅ ≈ 0 does not collapse (s100, s103), consistent with the near-zero mass on LMP attackers in all conditions.
- **S_swap ≈ 0.001:** the final policy responds to the state **even less** than B2.2's initial actor (0.0055), which is typical of tanh saturation.

**Reading:** giving TD3 a real learning budget (100× lr, 10× shorter warm-up) does not produce a state-conditioned policy. It produces a **degenerate constant action**, chosen almost at random among the corners of the Box (it varies by seed), which in 2 of 5 seeds under EB switches the filtering off.

## 3. Consequences (Gate B-a)

- **B-a:** "the steelman does not beat the fixed action" → the P2 thesis becomes strong. Together with B2.1 + B2.2 (equivalence confirmed; the published policy neither learns nor uses its input), the picture is:
  - **published budget:** TD3 does not leave its initialization, which is equivalent to a fixed action at the center;
  - **generous budget:** TD3 leaves its initialization, but towards a **constant corner**, also input-independent, and with a risk of collapse.
- In neither regime does TD3 learn what would justify using RL (a state-dependent action).

## 4. Declared limitations

- **Exploratory:** already-used seeds, n = 10 pairs, Step 2 baselines already inspected.
- **A single configuration.** The saturation at the corners suggests a conditioning problem, probably the reward scale: the official code uses the **sum** of the loss over ~156 test batches, with magnitudes of tens to hundreds, without normalization. A reviewer may ask for a steelman with a **normalized reward** (or normalized observations, or an lr below 1e-3). The correct claim is "this favorable configuration does not beat the fixed action; learning degenerates into a constant corner action", not "no configuration beats it".
- **Possible next step (to be decided, not pre-registered):** B2.3b with a normalized reward and lr 1e-4, same seeds, ~16 h.
