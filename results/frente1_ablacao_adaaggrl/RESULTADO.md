# Result: AdaAggRL ablation (Front 1, after V0)

**Date:** 2026-09-24. **Pre-registration:** `PREREGISTRO.md` (English translation in `PREREGISTRO.en.md`), written before the grid.
**Code:** `scripts/frente1_ablacao_adaaggrl.py` (subcommands `run`, `auc`, `analyze`). Nothing was changed in `src/`.
**Grid:** 4 variants × 21 cells × 10 seeds = 840 runs, complete (210 per variant). The `td3_ref` reference and Random were reused from `exp10_*_full_seed*_raw.csv`.

**Files:**
- `grade_combined_raw.csv`
- `grade_pooled.csv`
- `grade_por_ataque.csv`
- `grade_analise.txt` (full output)
- `passo1_auc_detectores.csv` and `detectores/` (step 1)
- `raw/`

**Methodology checks, done before the grid:**
- the script's `td3` trajectory reproduces the official reference with a difference of 0.0 (seed 42, 21 cells);
- the headroom function reproduces the documented 9/21.

## Summary

| variant | mean accuracy | cells with headroom (of 21) | Δ vs. td3_ref (90% CI) | Wilcoxon p | pre-registered reading |
|---|---|---|---|---|---|
| td3_ref (AdaAggRL) | 0.7924 | 9 | — | — | — |
| **fixed** (a=0.5, b=0.5, no TD3) | **0.8103** | 11 | **+0.0179** [+0.006; +0.030] | **0.049** | **H1 refuted in the strong sense: removing TD3 improves** |
| **sr_only** (S_R only) | 0.8058 | **12** | +0.0134 [−0.005; +0.032] | 0.28 | H2 **inconclusive** (neither equivalent nor different) |
| cosmed_only (cosine to the median) | 0.7331 | 6 | −0.0593 [−0.078; −0.041] | 0.002 | H3 **fails**, as predicted after step 1 |
| cosserver_only *(exploratory)* | 0.8071 | 6 | +0.0147 [−0.009; +0.039] | 0.28 | — |
| Random | 0.7622 | 3 | — | — | — |

## Reading

**H1. TD3 does not contribute.** The result is stronger than the hypothesis. By the pre-registered rule, `fixed` better than the reference with p < 0.05 means "TD3 hurts". The aggregate effect is +1.8 p.p. (d = 0.86), but p = 0.049 is borderline.
- Per attack, no difference survives the Holm correction.
- The gain comes mainly from `label_flipping` (+0.078; uncorrected Wilcoxon p = 0.010, Holm 0.068).
- There is formal equivalence (±0.01, Holm) on `gaussian_noise` and `krum_collusion`.
- **V0's finding stops being observational:** in this regime (15 rounds), AdaAggRL's performance comes from the fixed filter, and TD3's actions are, at most, noise that costs accuracy.

**H2. The MMD cues do not contribute beyond S_R.** Formally inconclusive: the 90% CI crosses +0.01. But there is no sign that the cues help.
- `sr_only` is numerically above the reference (+1.3 p.p.) and has the **largest number of cells with headroom (12/21)**.
- The difference to `fixed` (0.8058 vs. 0.8103) is small. That comparison is descriptive and was not pre-registered.

**H3. A trivial detector is enough.** It fails with `cos_median`: −5.9 p.p. vs. the reference and −7.3 p.p. vs. `sr_only`, with a collapse on `trim_attack` (0.503, vs. 0.835 for the reference). This is what step 1 anticipated: `trim_attack` and `krum_collusion` are built to look like the median.

**Exploratory: `cosserver_only` sits practically at the level of `sr_only`** (Δ = +0.0014, 90% CI [−0.009; +0.012]; the TOST narrowly fails to close, p = 0.08), but with a **complementary profile**:
- it loses to `sr_only` on `fltrust_aligned` (−0.037) and on `gaussian_noise` (−0.044), with Holm < 0.05;
- it gains a lot on `label_flipping` (+0.092, Holm < 0.05);
- both beat Random comfortably (+4.5 and +4.4 p.p.).

It is exploratory because the detector was chosen after seeing the seed-42 AUCs. It needs replication on new seeds before becoming a claim.

## Implications

1. **The AdaAggRL reference for the new paper.** The `fixed` variant (and `sr_only`) is stronger than AdaAggRL itself in this regime. Citing AdaAggRL as "the continuous-RL baseline" without this ablation would be misleading. The honest target to beat, in Front 2 or in any comparison, becomes **S_R filter + penalty without RL**, with 0.806–0.810 and 11–12/21 cells with headroom. This decision is the author's.
2. **For Front 1:** the "headroom" AdaAggRL captures does not come from learning. It comes from a heuristic per-client filter with memory (penalty λ^h). This reinforces V0: the relevant difference between AdaAggRL and discrete selection is **per-client weighting with memory**, not "continuous vs. discrete" nor "RL vs. bandit".
3. **For Front 2 (privacy differentiator):** the signal is positive, but it has a cost.
   - A detector without gradient inversion (`cos_server`) reaches S_R's level in the aggregate.
   - But it depends on a **root dataset on the server**, which is exactly the open data-protection (LGPD) question about the root dataset. Data reconstruction is traded for a root set, and the differentiator is not free.
   - The trivial detector without a root (`cos_median`) is refuted.
   - The per-attack complementarity between S_R and `cos_server` suggests that combining the two could beat both (a new hypothesis, not tested).

## Caveats

- The values b = 0.5 and λ = 2 were fixed without a search. Other values may change the magnitude, especially on `label_flipping`, where the variation across seeds is large.
- H1's p = 0.049 is borderline. With n = 10, the minimum Wilcoxon p is 0.002, so the result is real, but fragile.
- The reference and Random were reused from earlier runs. Pairing by seed is valid, because the script's `td3` trajectory reproduces the reference exactly.
- MNIST and 15 rounds only. The conclusion "TD3 does not learn" holds for this horizon. With long horizons, TD3 might start contributing (not tested).
