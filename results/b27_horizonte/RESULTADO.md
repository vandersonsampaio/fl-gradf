# Result — B2.7: does the gain from learning grow with the horizon?

**Date:** 2026-10-02. Plan frozen in `PLANO.md` (English translation in `PLANO.en.md`; hash in `PLANO.sha256`, commit `8ab747f`) before the grid.
**Grid:** 80 cell × seed jobs (20 systems × 150 rounds) + 30 ceilings, from 10/01 12:39 to 10/02 15:56. No failures.
**Outputs:** `analise.txt`, `delta_por_celula.csv`, `grade_raw.csv`, `tetos_raw.csv`, `verificacao.txt`.

## Short answer

**No.** By the plan's criterion (Δ > 0 with CI95 > 0 in ≥ 3 of the 8 cells), **no agent pays off at any horizon**. The only positive signal (TD3, 1/8 cells at H = 150) was explained by B2.7a/b as non-learning factors (exploration noise and/or a shifted constant, not separable), not as learning (see the TD3 reading below).

| agent | fixed reference | H = 15 | H = 50 | H = 150 | mean Δ over the cells (15 → 50 → 150) |
|---|---|---|---|---|---|
| TD3 (AdaAggRL) | `fixed`, same skeleton | 0/8 | 0/8 | 1/8 | −1.65 → −1.12 → **+1.33** p.p. |
| LinUCB (FedStrategist b) | best `arm_rule` in hindsight | 0/8 | 0/8 | 0/8 | −11.55 → −8.23 → −8.62 p.p. |
| DQN (GRADF v1) | best `arm_gradf` in hindsight | 0/8 | 0/8 | 0/8 | −13.27 → −11.61 → −9.54 p.p. |

## Reading per agent

**TD3: a positive trend in Δ vs. the center, concentrated in three cells and not passing the criterion. B2.7a/b showed that it is not learning.**
- The only cell that passes is `label_flipping` α 0.05 at H = 150: +6.65 p.p., CI95 (+2.84; +10.46). At H = 15 and 50, the same cell was at −5.8 and −5.3 p.p.
- The turn in the mean Δ comes from 3 cells:
  - `label_flipping` α 0.05: slope +5.3 p.p. per log H;
  - `label_flipping` α 0.1: +2.2, with H = 150 at +2.7 and a CI95 crossing zero;
  - `low_mag_backdoor` α 0.05: +2.4, with H = 150 at +3.1 and a CI95 crossing zero.
- In the other 5 cells, Δ stays within ±0.1 p.p. (`gaussian_noise`, `krum_collusion`, `trim_attack`) or oscillates (`sign_flipping` α 0.05: −2.0 at H = 150).
- **Reading (updated on 2026-10-04 with B2.7a/b, `results/b27b_td3_constante/RESULTADO.md`):** TD3 does **not** start using the signal at a long horizon. In the 3 cells with a positive slope:
  - the policy does not depend on the state: the sd_estados of π₁₅₀ (~0.0005) equals the initial network's and is ~300× smaller than σ_a = 0.15;
  - TD3 with a frozen actor (same noise, no learning) is indistinguishable from `td3_ref` (CI95 contains 0 in the 3 cells);
  - the constant that TD3 itself learned is indistinguishable from `td3_ref` (wide CI95, ±6 p.p.; this is not equivalence);
  - **a constant with b = 0.25 beats TD3 by 4.4 to 6.9 p.p. in the 3 cells**, with the whole CI95 below 0.

  The gain over the center comes from non-learning factors (exploration noise and/or a constant shifted to b ≈ 0.45, not separable: B2.7b's criterion 1 failed narrowly, and criterion 2 only shows that there is no detectable difference). **B2.7c was cancelled** (condition 1 not met). *The earlier reading ("a hint that TD3 starts using the signal… deserves confirmation") is superseded.*

**LinUCB: far below the best fixed arm at every horizon.**
- Δ between −0.6 and −26.4 p.p. The difference shrinks with H in 4 cells and grows in others (`sign_flipping` α 0.05: −19.4 → −26.4).
- Against the random arm, Δ stays near zero (−8 to +7 p.p.). LinUCB is indistinguishable from picking the rule at random.

**DQN: far below the best fixed arm and, in most cells, below its own random arm.**
- Δ from −1.2 to −24.0 p.p. The slope per log H is positive in 7 of 8 cells (mean Δ −13.3 → −9.5), but stays far from zero.
- Against `rand_gradf`, Δ is negative in 22 of the 24 cell × H combinations. On `low_mag_backdoor` α 0.05, it reaches −18 to −23 p.p.: the learned policy is **worse than choosing at random**.

**Caveat (foreseen in the plan):** the best arm is chosen in hindsight and is optimistic for the fixed side. But LinUCB and DQN also lose to the random arm or tie with it, so the conclusion does not depend on this caveat.

## Robustness vs. convergence (the C0 question)

The ceilings rise with the horizon, especially the FedAvg-8 oracle at low α. Oracle values **corrected** on 2026-10-02 (see the correction at the end of this file):

| α | FedAvg-10 without attack (15 → 50 → 150) | FedAvg-8 oracle (15 → 50 → 150) |
|---|---|---|
| 0.05 | 85.3 → 88.5 → 89.5% | 77.1 → 84.0 → 86.5% |
| 0.1 | 85.5 → 88.5 → 89.4% | 79.0 → 82.5 → 85.0% |
| 0.5 | 90.5 → 91.6 → 91.9% | 89.9 → 91.1 → 91.7% |

- **Note (2026-10-04):** the gaps in this section are against the **sample-size-weighted** FedAvg-8 oracle. With α ≤ 0.1, the negative gaps **mix convergence and sample-size weighting**: the uniformly weighted oracle-8 is ~4.5 p.p. above the weighted one at α 0.1 (H = 150) and is the appropriate reference ceiling (see `results/c0b_espaco_h150/RESULTADO.md`, sensitivity section, and `VERIFICACOES.md`).
- **The FedAvg-8 oracle does not converge in 15 rounds with α ≤ 0.1.** It gains 6.0–9.4 p.p. up to H = 150. So part of the C0 negative gaps (methods under attack above the oracle at 15 rounds) is a **ceiling convergence artifact**:
  - TD3 on `gaussian_noise` α 0.05: gap to the oracle −3.2 → −1.4 p.p.
- In other cells the negative gap **persists** with the horizon:
  - TD3 on `krum_collusion` α 0.1: −4.5 → −4.4;
  - TD3 on `sign_flipping` α 0.1: −1.2 → −2.7.
  - In those cells, the skeleton excludes the attackers (A0(a): mass ≈ 0) and weights the honest clients uniformly, while the FedAvg-8 oracle weights by sample size. Since the metric is the accuracy on the global, IID and balanced test set, and the labels are skewed per client, sample-size weighting unbalances the classes, and this is enough for the skeleton to beat the oracle (mechanism corrected on 2026-10-05). (Corrected on 2026-10-04; see `results/c0b_espaco_h150/VERIFICACOES.md`: the sample-weighted oracle underestimates the ceiling at α ≤ 0.1.)
- **`label_flipping` α ≤ 0.1, the only region with headroom in C0, still has a large gap at H = 150 against TD3 and the reference skeleton** (TD3: +18.0 p.p. to the oracle at α 0.05; +10.9 at α 0.1). **Against the best existing method, the headroom closes at α 0.1 and +2.8 p.p. remain at α 0.05** (C0b, uniform oracle; the best existing method is FLTrust).

## Checks (`verificacao.txt`)

**1. Nesting:** identical (|Δ| = 0) for 7 of the 8 systems tested (`td3_ref`, `fixed`, `linucb`, `rand_rule`, `rand_gradf`, `arm_rule_median`, `arm_gradf_median`). `dqn` differed by 4.15 p.p. (H = 15) and 3.74 p.p. (H = 50).
- **Diagnosed cause:** not a horizon dependence. Two processes with an identical configuration (15 rounds, same system order) give the same 4.15 p.p. difference for `dqn` and |Δ| = 0 for the others.
- **GRADF's DQN is non-deterministic from run to run**, even with the seed fixed (probably TF state or non-deterministic operations in the selector's online training).
- **Declared deviation:** the plan (§6.1) foresaw redoing the horizons as separate runs if the nesting failed. This was not done, because the cause is not the nesting and separate runs would not remove the non-determinism.
- **Consequence:** `dqn` carries extra run-to-run noise, absorbed into the variance across seeds. D2 (`results/d2_variancia_p1/`) measured this noise in the P1 code: a mean SD of 9.3 p.p. per cell × seed (range up to 63 p.p.); on the 21-cell aggregate, SD 1.8 p.p. The DQN Δs (−9 to −13 p.p. on average) are much larger than that noise; the conclusion does not change.

**2. Reproduction at H = 15 against the earlier runs:**
- `td3_ref` and `fixed`: identical (80/80).
- `linucb`: mean |Δ| 0.9 p.p. (max. 23.6).
- `dqn`: mean |Δ| 17.4 p.p. (max. 61.9).
- `rand_gradf`: mean |Δ| 1.6 p.p.
- Arms vs. exp9: mean |Δ| ≤ 2.2 p.p.
- The differences come from the order dependence of the global RNG (fixed in B2.7, PLANO §6.3) and, for the DQN, from the non-determinism above. **The DQN/GRADF numbers from the earlier experiments (exp10 etc.) therefore have unreported run-to-run variance.**

## Implications

1. **P2:** "learning does not pay off" extends to horizons 10× longer (150 rounds) for LinUCB and DQN, against the fixed version in the same action space and against random. **TD3 also does not pay off up to 150 rounds:** the gain over the center on `label_flipping` and `low_mag_backdoor` comes from non-learning factors (exploration noise and/or a shifted constant, not separable; B2.7a/b), and the best constant (b = 0.25) beats it.
2. **C0:** part of the negative gaps was ceiling convergence, and part the oracle's sample-size weighting (C0b). The headroom on `label_flipping` α ≤ 0.1 persists against TD3 and the reference skeleton. Against the best existing method, it closes at α 0.1 and +2.8 p.p. remain at α 0.05 (C0b, uniform oracle).
3. **Reproducibility:** GRADF v1's DQN is non-deterministic across runs, and the framework's global RNG depends on the execution order. This should be fixed before GRADF-v2 (P3) and declared as a limitation of the P1 results.

## Correction of the FedAvg-8 oracle ceiling (2026-10-02)

- **Bug:** `run_ceiling` recorded the FedAvg-8 oracle when `round_num + 1 ∈ {15, 50, 150}`, but `train` numbers rounds starting from 1. The values labeled H = 15, 50 and 150 were those of rounds 14, 49 and 149.
- **The FedAvg-10 ceiling and all systems were correct**, because they use `results[H − 1]`.
- **Fix:** `if round_num in HORIZONS` in `scripts/b27_horizonte.py`. The 30 ceiling jobs were re-run into `raw_teto_corrigido/`; the files with the bug stay in `raw/teto_*` and the earlier analysis in `v1_teto_bug_*`.
- **Effect:**
  - FedAvg-10 comes out identical (|Δ| = 0);
  - the oracle changes by +0.25 p.p. on average at H = 15 (max. 0.56), +0.05 at H = 50 and +0.01 at H = 150;
  - the gaps to the oracle change by at most 0.46 p.p.;
  - **the agents' Δs and all criterion verdicts are identical.**
- **Check:** the corrected oracle at H = 15 is within 0.05 p.p. (mean; max. 0.13) of the C0 oracle ceiling, which was computed in another process.
