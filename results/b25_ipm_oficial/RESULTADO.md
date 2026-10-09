# Result — B2.5: real IPM in the official AdaAggRL (fixed vs. td3)

**Date:** 2026-10-02.
- Plan: `PLANO.md` (English translation in `PLANO.en.md`), commit `1e92014`.
- Addendum 1 (ε = 2): `ADENDO1.md` (English translation in `ADENDO1.en.md`), commit `4faad80`.
- Grid: 20 runs (seeds 105–114 × {td3, fixed}, 500 rounds), from 10/01 14:12 to 10/02 19:36. No failures.
- Outputs: `analise.txt`, `resumo_runs.csv`, `mecanismo.csv`, `raw/`.

## H1 (pre-registered): INCONCLUSIVE

| metric | Δ fixed − td3 | CI90 | CI95 | TOST ±1 p.p. | Wilcoxon |
|---|---|---|---|---|---|
| **primary:** median 401–500 | +2.78 p.p. | −4.63 to +10.19 | −6.37 to +11.92 | p = 0.66 | p = 0.63 |
| secondary: mean 451–500 | +4.23 p.p. | −5.65 to +14.12 | −7.96 to +16.43 | p = 0.72 | p = 0.56 |

- Neither equivalence nor a difference. The sign favors **fixed**, without significance.
- **There is no evidence that TD3 contributes under IPM.**

## Why inconclusive: reset cascades

- The sd of the per-seed differences is **12.8 p.p.**, vs. ~0.3 p.p. in B2.1.
- IPM with ε = 2 triggers many resets of the official environment (reward < −80 → model re-initialization): a median of **2.5 extra resets per run in both conditions**.
- Two seeds have **late cascades** inside the 401–500 window:
  - **110, td3:** 7 resets, 5 of them between rounds 414 and 494 → median 57.9%;
  - **111, fixed:** 7 resets, 4 of them between rounds 402 and 435 → 68.4%;
  - **111, td3:** resets at 412 and 428 → 81.8%.
- Since the reset is a consequence of the action, the plan does not allow excluding it. The 100-round median does not withstand cascades with more than ~50 contaminated rounds, unlike B2.1, where resets were isolated.
- **Post hoc sensitivity**, with no confirmatory value: excluding seeds 110 and 111, Δ = +0.58 p.p., CI90 from −0.99 to +2.15, sd 2.35. The result still shows no equivalence at ±1 p.p. and no difference.

## Mechanism (descriptive): TD3 still does not learn

| | median | max. |
|---|---|---|
| drift \|π₅₀₀ − π₀\| | 0.0107 | 0.0168 |
| sd_estados of π₅₀₀ | 0.0029 | 0.0106 |

- Both stay well below the exploration noise (σ_a = 0.0475), as in B2.1 (drift ~0.01).
- Under IPM, the final policy is practically the initial one and does not depend on the state.
- The accuracy difference between the conditions comes from the reset dynamics, not from a learned policy.

**Mass on attackers** (median across seeds, rounds with real attackers): fixed 0.089, td3 0.096. With ε = 2, both filters give appreciable weight to the attackers, as the sanity check anticipated.

## Audit findings

1. **The official code's IPM is a null update** (`teste_unitario.txt`): the network is set to `old_weights` before the `craft`, so the attacker sends the global model itself. The original paper's IPM results do not test Xie et al.'s IPM.
2. The real IPM (via the LMP hook, a declared deviation) is an effective attack in the official code: it degrades FedAvg by 5.9 p.p. with ε = 2 and by 63.5 p.p. with ε = 10 (sanity). It also triggers reset cascades in the environment's re-initialization mechanism.

## Reading for the thesis

- The confirmatory P2 conclusion ("fixed ≡ TD3") still holds **for LMP and EB** (B2.1).
- **Under real IPM:**
  - no equivalence is established, because of the reset-induced variance;
  - there is no evidence of a TD3 contribution: Δ favors fixed, without significance;
  - the mechanism (policy with no drift and independent of the state) holds.
- Suggested wording: "under IPM, the TD3 policy does not learn (drift ≪ exploration noise) and there is no evidence of a gain over the fixed action; equivalence could not be established due to reset cascades in the official environment".
- **Limitation:** n = 10, one cell (MNIST, q = 0.5, ε = 2), primary metric sensitive to long reset cascades.
