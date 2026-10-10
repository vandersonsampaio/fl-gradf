> English translation of `PLANO.md`. The Portuguese original is the frozen record (its hash is in `PLANO.sha256`); if the two ever disagree, the original prevails.

# Plan — B3.3: short sweep of the threshold a₅ on BloodMNIST (official AdaAggRL, EB)

**Status:** FINAL before any run (hash in `PLANO.sha256`). **Exploratory.** The code does not exist yet: it will be written, tested (§4) and have its hash recorded in an addendum **before** the launch, without changing anything in this plan.
**Date:** 2026-10-09
**Background:**
- **B2.4r** (MNIST, EB; `results/b24r_limiar_oficial/`): the threshold is causal in the published code. a₅ ≤ 0.1 collapses or is costly, and a₅ ∈ [0.25; 0.95] forms a plateau within 0.4 p.p. of the center. No constant beats the center (Q2: no).
- **B3.1** (BloodMNIST; `results/b31_medmnist_oficial/RESULTADO.md`): fixed ≈ td3 (±3 p.p.), in a degraded, periodic-reset regime. Accuracy is ~31% (vs. 77.8% without attack), there is a reset every ~23 rounds and the weight mass on attackers is ~0.016 (vs. ≤ 0.0001 on MNIST).

## 1. Question

On BloodMNIST, under EB, does **a threshold higher than the center** (which excludes more clients per round) beat the central action? That is: is B3.1's degraded regime, at least in part, a threshold poorly calibrated for this dataset, which a simple constant would fix?

The question is about constants, not about learning. If the answer is yes, the published TD3 does not find the better constant either (B3.2: the policy stays at its initialization), and P2 gains the same reading as B2.7b: "the possible contribution of RL amounts, at most, to tuning a constant".

## 2. Design

- **Code:** official (`external/AdaAggRL`, commit `27b9c18`), venv `external/.venv_adaaggrl`, without any edit.
  - BloodMNIST adaptation: the same as B3.1 (`bloodmnist_shim.py`, extractor and data with the hashes from B3.1's Addendum 1).
  - New runner `run_b33.py`, which reuses `run_b3.run` and only replaces `R.A_FIXED` in the process, as `run_b24r.py` does with `run_oficial`.
- **Regime:** identical to B3.1 (BloodMNIST, q = 0.5, 96 clients, 10 per round, 20 attackers, lr 0.05, 500 rounds). Attack: **EB** only.
- **Conditions:** fixed action [0.475; 0.475; 0.475; 0.475; a₅] with **a₅ ∈ {0.475 (center); 0.75; 0.95}**. The official Box is [0; 0.95].
  - Only values **above** the center enter: in B2.4r, values below 0.25 collapse or are costly, and the hypothesis here is insufficient filtering (mass on attackers higher than on MNIST).
- **The center is re-run on the new seeds.** It is not reused from B3.1 (seeds 135–144), to keep the pairing by seed and the same GPU execution, which is not bit-reproducible (B2.4r ADENDO1).
- **Seeds:** **145–149** (new). Total: 3 × 5 = **15 runs**.
- **Execution:**
  - Queue **interleaved** by seed (center, 0.75, 0.95 for each seed), 6 processes on the GPU. So each batch mixes the three conditions.
  - 7h–18h Mon–Fri execution window with `scripts/janela_execucao.sh`.
  - Cost from B3.1's throughput (~0.69 run/h): **~22 GPU hours**.
- **Logging:** the same as B3.1 for `fixed` (accuracy, loss, executed action, mass on real attackers, resets, `a_fixed` in the JSON). Partial checkpoints every 25 rounds. One directory per a₅ (`raw/a5_<value>/`), because the file name does not include a₅.

## 3. Metrics

- **Primary:** median accuracy over rounds 401–500 (the same as B3.1).
- **AUC sensitivity:** mean accuracy over rounds 1–500 and over 251–500, as in B3.1.
- **Mechanism (descriptive):** number of resets per run and median interval between resets; mean weight mass on real attackers (rounds with an attacker).

## 4. Check before the grid

- **Runner test** (2 rounds, seed 100, outside the grid and discarded): for a₅ ∈ {0.475; 0.75; 0.95}, the JSON must record `a_fixed` = [0.475]×4 + [a₅] and the action executed at each step must equal `a_fixed`.
- There is no reproduction check against B3.1, because the center is re-run on the same seeds as the grid.

## 5. Criterion (fixed now; exploratory; n = 5)

For each a₅ ∈ {0.75; 0.95}: Δ = metric(a₅) − metric(center), paired by seed, with a t CI95 (4 d.f.).

- **"Some constant beats the center":** in at least one a₅, **Δ > 2 p.p. with CI95 > 0 on the primary.**
- **Reset caveat:** if the criterion is met on the primary but Δ ≤ 0 on AUC 1–500, the result is reported as **"reset-sensitive"** and is not claimed without that caveat.
- **No multiplicity correction** (2 comparisons), declared. Wilcoxon descriptive only: with n = 5, the smallest possible two-sided p is 0.0625.
- **Resets count as results** (a consequence of the action) and are never excluded.
- **Reading of the remaining cases:** Δ ≤ 2 p.p. or a CI95 that includes 0 → **"no constant beats the center"** in this sweep. Δ < −2 p.p. with CI95 < 0 is reported as "higher threshold is worse".

## 6. Termination clause

- **B3.3 is the only threshold sweep on BloodMNIST.** No value of a₅, attack or seed is added after seeing the result, whatever it is.
- **Criterion not met:** the §1 question is closed as "there is no simple constant above the center that fixes the regime" (within the resolution of n = 5), and no other sweep is opened.
- **Criterion met:** the finding stays **exploratory**. Any follow-up (e.g. confirming with new seeds or testing TD3 with the shifted threshold) requires a new pre-registration and an author's decision; it does not happen automatically.
- Runs with a complete final JSON are never re-run (B3.1 §6). Only runs that fail without a final JSON (environment error) are repeated with the same seed, and this is recorded in `grid.log` and in the RESULTADO.
- **Execution failure:** if more than 3 of the 15 runs fail without a final JSON even after one repetition, the grid is stopped and the author is consulted.

## 7. Reading

- **Criterion met:** B3.1's degraded regime is, in part, a threshold poorly calibrated for BloodMNIST. A simple constant beats the center, and the published TD3 (stuck at its initialization, B3.2) does not find it. This reinforces in P2 the reading that learning would amount, at most, to tuning a constant.
- **Criterion not met:** the degradation does not come from the threshold being too low. Together with B3.1's equivalence, this indicates that neither the central action nor the higher thresholds defend BloodMNIST under EB in the official environment. P2 reports this as a limit of the published mechanism on that dataset.

## 8. Rules

- Pre-written analysis (`analisar_b33.py`, hash in the addendum), run once with the complete grid.
- No accuracy is inspected before the grid ends. Monitoring reports only health and progress.
- **Order:**
  1. this plan committed;
  2. code (`run_b33.py`, `run_grid_b33.sh`, `analisar_b33.py`) and the §4 test;
  3. addendum with the hashes;
  4. commit;
  5. launch.
- Logs and partial checkpoints are not versioned.
