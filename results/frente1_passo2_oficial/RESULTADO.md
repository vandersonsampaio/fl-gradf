# Result — Step 2: does TD3 contribute in the official AdaAggRL?

**Date:** 2026-09-27
**Pre-registration:** `PREREGISTRO.md` (English translation in `PREREGISTRO.en.md`; original hash and Addendum 1 hash in `PREREGISTRO.sha256`)
**Grid:** 30/30 runs (LMP, EB × td3/fixed/random × seeds 100–104; MNIST, q=0.5, 500 rounds), 2026-09-25 12:46 → 2026-09-27 09:51, 0 failures.
**Analysis:** `scripts/passo2_oficial/analisar.py` → `analise.txt`, `resumo_runs.csv`, `curvas_acc.csv`. Exploratory checks run separately (see §4b).

---

## 1. Confirmatory result (as pre-registered)

Primary metric: mean accuracy over rounds 451–500 (runs truncated at 500 steps, Addendum 1).

| attack | seed | fixed | random | td3 |
|---|---|---|---|---|
| EB | 100 | 0.8811 | 0.9163 | 0.8250 |
| EB | 101 | 0.9674 | 0.6995 | 0.9656 |
| EB | 102 | 0.9671 | 0.8142 | 0.9660 |
| EB | 103 | 0.9639 | 0.7969 | 0.9664 |
| EB | 104 | 0.9622 | 0.6630 | 0.9679 |
| LMP | 100 | 0.9659 | 0.9686 | 0.9648 |
| LMP | 101 | 0.9654 | 0.9663 | 0.9696 |
| LMP | 102 | 0.9680 | 0.9679 | 0.9657 |
| LMP | 103 | 0.9631 | 0.9641 | 0.9668 |
| LMP | 104 | 0.9655 | 0.9687 | 0.9675 |

**H1 (fixed − td3), pooled n=10:** Δ = **+0.45 p.p.**, CI95 (−0.87; +1.76), d = +0.24. TOST ±1.0 p.p.: p = 0.18. Wilcoxon: p = 0.56.
→ **PRE-REGISTERED VERDICT: INCONCLUSIVE.** There is no evidence that TD3 is better than the fixed action (the point estimate favors the fixed one), **but equivalence within ±1 p.p. was not shown**.
- LMP: Δ = −0.13 p.p., CI95 (−0.49; +0.23), Holm p = 0.88.
- EB: Δ = +1.02 p.p., CI95 (−2.19; +4.23), Holm p = 1.00. The width comes from **one** pair (seed 100, see §2.1).

**H2 (random − td3), pooled n=10:** Δ = −8.00 p.p., CI95 (−17.53; +1.54), d = −0.60. TOST p = 0.93, Wilcoxon p = 0.19 → **INCONCLUSIVE**.
- LMP: Δ = +0.02 p.p. (the action is irrelevant under LMP).
- EB: Δ = −16.02 p.p., Holm p = 0.25 (a random action collapses under EB in 4/5 seeds, but n=5 is not enough for Wilcoxon + Holm).

## 2. Exploratory results (not confirmatory)

### 2.1 The only discrepant pair is a late-reset artifact
The official environment re-initializes the FL model from scratch when the reward is < −80. In EB, seed 100, **fixed reset at round 421 and td3 at 441** — the 451–500 window measures the recovery of a freshly re-initialized model, not the stable regime (fixed: min 0.72 in the window; td3: min 0.56). In the other 9 pairs, |Δ| ≤ 0.5 p.p. The pre-registration does not allow excluding the pair; it is recorded as the cause of the CI width.

Extra-reset counts (5 seeds summed):

| | fixed | td3 | random |
|---|---|---|---|
| EB | 3 | 14 | 60 |
| LMP | 1 | 3 | 0 |

The td3 resets concentrate before round 122 — in SB3's 100-round warm-up phase with uniformly random actions. After the warm-up, only 1 reset (s100, round 441). **TD3's cost under EB is the random warm-up, not the learned policy.**

### 2.2 The TD3 policy does not move away from its starting point and does not depend on the attack
- Mean drift of the action relative to the Box center (0.475) over rounds 101–500: ≤ 0.023 per dimension (mean across seeds); maximum per seed/dimension 0.079. No trend across 100-round windows.
- **Same seed, different attacks → almost the same action sequence:** step-by-step correlation of the actions after round 100 between EB and LMP = **0.91 / 0.99 / 0.99 / 0.94 / 0.99** (seeds 100–104), with mean actions equal to the 3rd decimal place. States and rewards differ a lot between the attacks (e.g. the `sim_lc ≥ 0.9` rule fires 1.4×/round under EB and ~0 under LMP). The trajectory is determined by the network initialization and the noise sequence (both fixed by the seed), **not by what the agent observes**.
- Consistent with the learning budget of the published configuration: ~133 critic steps / ~66 actor steps at lr 1e-5.

### 2.3 The memory filter excludes the attackers in every condition
Weight mass aggregated onto real attackers, mean per round: ≤ 0.0011 in all 6 combinations (including `random`). Under LMP the action does not matter (the three conditions tie); under EB, a random action hurts through resets, and the fixed one is as good as TD3.

## 3. Reading and consequences for the plan

**What can be claimed:**
1. (confirmatory) In the published code and horizon, **there is no evidence that TD3 beats a fixed action**; the point estimate favors the fixed one (+0.45 p.p.).
2. (confirmatory) Equivalence within ±1 p.p. **was not established** with 5 seeds — mainly because of one late reset under EB.
3. (exploratory, strong) The learned policy **is essentially the initial policy plus noise**: it does not move away from the center and is almost identical across attacks with very different states and rewards.

**What cannot be claimed:** "TD3 is equivalent to the fixed action" (TOST failed); "TD3 hurts" (no significant test).

**Branches of the plan (§6 of the pre-registration):** neither pre-registered outcome materialized cleanly. The most honest reading is "RL does not contribute at this horizon, and the mechanism is that it never gets to learn", which supports the central thesis on an **exploratory** basis, and not the branch "RL only pays off after N rounds" (there is no learning trend over the 500 rounds).

## 4. Next-step options (author's decision)

1. **Confirmatory replication of the equivalence**, with a new pre-registration: new seeds (e.g. 105–114), same margin. Declare it as a new study, never as a sequential extension of seeds 100–104 without correction. Consider also pre-registering a reset-robust metric (e.g. a stable window: rounds with no reset in the previous 50), justified by §2.1. Cost: ~9 h per batch of 6 runs (fixed and td3 only: 20 runs ≈ 30 h).
2. **Confirm the mechanistic finding (§2.2)** as a pre-registered hypothesis: "EB×LMP action correlation ≥ 0.9 per seed" and "|drift| < exploration noise". Cheap: the data already exist for seeds 100–104 (exploratory); confirmation would need new seeds — it can run together with option 1.
3. **Move on to Steps 3–5**, treating Step 2 as: "no evidence of a TD3 contribution; the policy does not learn within the published budget" (exploratory), and keep option 1 for before submission.

## 4b. Commands for the exploratory checks

The §2 checks were run ad hoc on `raw/*.json` (resets per run, action drift per window, EB×LMP action correlation per seed). To reproduce, read `steps[:500]` of each JSON: fields `action`, `acc`, and `resets[1:]`.
