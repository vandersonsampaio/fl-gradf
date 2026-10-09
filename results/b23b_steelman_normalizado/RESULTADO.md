# Result — B2.3b: last TD3 steelman (normalized reward)

**Date:** 2026-10-01
**Plan:** `PLANO.md` (English translation in `PLANO.en.md`; hash in `PLANO.sha256`; `PLANO.md`, `run_b23b.py`, `analisar_b23b.py`, `analisar_b23.py` and `analisar_b21.py` checked against the hashes before the analysis: OK).
**Grid:** 10/10 runs (seeds 100–104 × LMP/EB; `VecNormalize` on the reward only, lr 1e-4, `learning_starts` 10), 09/30 18:55 → 10/01 09:50, 0 failures. Paired with Step 2's `fixed` and `td3` and with the B2.3 steelman.
**Analysis:** `scripts/passo2_oficial/analisar_b23b.py` → `analise.txt`, `resumo_runs.csv`, `mecanismo.csv`, `drift_por_checkpoint.csv`. **Exploratory** (already-used seeds).

---

## 1. Definitive Gate B-a (criterion fixed in PLANO §5)

| condition | result |
|---|---|
| 1. C1 B2.3b − fixed with Δ > 0 and p < 0.05 | **NO**: Δ = +0.95 p.p. (CI95 −0.87 to +2.78), Wilcoxon p = 0.066 |
| 2. the policy moved away from the initial one (drift_500 > σ_a = 0.0475) | **YES**: median 0.174 |
| 3. the policy depends on the state (sd_estados > σ_a) | **NO**: median 0.0134 (max. 0.0231) |

→ **VERDICT: the steelman does NOT beat the fixed action. Gate B-a becomes definitive and the configuration search ends** ("last steelman" clause). **B2.3c does not run.**

Comparisons (primary, median 401–500; n = 10 pairs):
- **C1 vs. fixed:** +0.95 p.p., not significant. On the secondary (mean 451–500): **Δ = 0.00 p.p.** (CI95 ±0.42).
- **C2 vs. the published td3:** +0.77 p.p., p = 0.56.
- **C3 vs. the B2.3 steelman:** +11.2 p.p., p = 0.010. The normalization **eliminated the B2.3 collapses**: extra resets under EB fell from 15.8 to 0.8 per run, the level of the fixed action (0.6).

In short, well conditioned, TD3 sits **at the level of the fixed action and of the published TD3**, with no gain and no collapse.

## 2. Mechanism (exploratory)

- **The policy moves continuously and does not saturate:** the median drift grows almost linearly over training (0.009 at step 50; 0.047 at 200; 0.13 at 400; **0.174 at 500**). The final action is an **interior point** of the Box that varies by seed (e.g. LMP s100 [0.79; 0.71; 0.35; 0.75; 0.31]), not a corner as in B2.3.
- **State dependence grows, but stays small:** the sd_estados (median) per checkpoint, computed over the same states of rounds 401–500, was:

  | step | 0 | 100 | 200 | 300 | 400 | 500 |
  |---|---|---|---|---|---|---|
  | sd_estados | 0.0045 | 0.0048 | 0.0059 | 0.0085 | 0.0113 | **0.0134** |

  It **triples** in 500 rounds, but stays at **~1/3.5 of the exploration noise** (σ_a = 0.0475). Step 0 corresponds to the sensitivity of a random network.
- **Extrapolation (speculative):** at the pace of the last 200 rounds (~+0.0025 every 100 rounds), sd_estados would only reach σ_a around round ~1,900. It **cannot be ruled out** that, in this configuration, a state dependence emerges at a horizon much longer than the published one. Even if it does, nothing indicates it would bring an accuracy gain: on the secondary, Δ = 0.00 vs. the fixed action.

## 3. The two steelmen side by side (for P2)

| | published TD3 (B2.1/B2.2) | B2.3 steelman (lr 1e-3, raw reward) | **B2.3b steelman (lr 1e-4, normalized reward)** |
|---|---|---|---|
| vs. fixed | equivalent (TOST) | −10.2 p.p. (collapses) | +0.95 p.p., n.s. (secondary: 0.00) |
| drift_500 | 0.0097 | 0.47 (saturates in ~200 rounds) | 0.17 (still growing) |
| state dependence | equal to a random network's (S_swap 0.0055) | none (constant corner, sd 0.001) | small and growing (sd 0.0134; 3× the initial) |
| resets under EB | 2.8 | 15.8 | 0.8 |

**Reading for P2:** in three learning regimes, TD3 **never beats the fixed action** in 500 rounds. The reason varies:
- in the published regime, it does not learn;
- with a high lr and a raw reward, it degenerates into a corner and switches the defense off;
- well conditioned, it slowly learns an almost constant policy, which ties with the fixed one.

The only sign of state-dependent learning (B2.3b) is small, grows slowly and does not turn into a gain. This reinforces B2.7 (horizon) as a gap to report: the question "from which N does RL pay off?" is not answered for budgets much larger than the published one.

## 4. Limitations

- **Exploratory:** already-used seeds; n = 10 pairs; Step 2 baselines already inspected.
- **One configuration.** By the termination clause, no other will be tested in this line.
- The per-checkpoint sd_estados and the extrapolation are **not pre-registered** analyses, done after the grid was complete.
- **500-round horizon,** the one in the published code.
