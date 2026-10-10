> English translation of `PLANO.md`. The Portuguese original is the frozen record (its hash is in `PLANO.sha256`); if the two ever disagree, the original prevails.

# Plan — B2.5: real IPM in the official AdaAggRL (fixed vs. td3)

**Status:** FINAL before any sanity or grid run (hash in `PLANO.sha256`). Changes after that only as a dated, hashed addendum.
**Date:** 2026-10-01
**Scope:** B2.5 (priority 1): the third attack of the original paper.
**Background:** B2.1 + B2.2 (`results/b21_replicacao_oficial/`): fixed equivalent to td3 (±1 p.p.) under LMP and EB; TD3 policy with no drift and no input dependence.

## 1. Question

Under an attack optimized against the mean (IPM), is the fixed action still equivalent (±1.0 p.p.) to AdaAggRL's TD3 in the published code and horizon?

## 2. Audit finding and declared deviation

- **The official IPM is null.** In `exp_environments.step`, the network is set to `old_weights` before `IPM_attack`, which returns `craft(old_weights, get_parameters(net), 5, -1)`; since `get_parameters(net) == old_weights`, the difference is zero and the attacker sends the global model itself. Check: `scripts/passo2_oficial/test_b25_ipm.py`, item 2 (norm of the crafted update = 0), output in `teste_unitario.txt`. **Reported as an audit finding**; the official IPM is not run in the grid (it would be a no-effect "free-riding" attack, not Xie et al.'s IPM).
- **Declared deviation: real IPM** (Xie et al., 2020): crafted delta = −ε × mean of the round's honest deltas (omniscient attacker, knows the honest updates); sent weights = `old_weights` + crafted delta; all real attackers of the round send the same vector.
- **Implementation without editing the official code** (`scripts/passo2_oficial/run_b25.py`): the environment runs on the LMP path (the only one that receives the list of already computed honest weights) and, in this process only, `exp_environments.LMP_attack` is replaced by the real-IPM function. The honest weights are those of the non-attackers and of attackers on their first participation (who, in the official code, train honestly), as in the official LMP. Everything else (aggregation, reward, resets, sampling) is the official code's.
- **Unit test** (item 1 of the same script): cos(crafted delta, honest mean) = −1 and norm ratio = ε, for ε = 2 and 10, on synthetic weights and on the official `MNISTClassifier`. **Passed** before this plan.

## 3. Choice of ε (sanity, seed 100, already used)

Candidates: **ε ∈ {2, 10}**. Five runs of **100 rounds**, seed 100 (`run_grid_b25.sh sanity`):
`fedavg` without attack; `fedavg` with ε = 2 and 10; `fixed` with ε = 2 and 10.

- `fedavg`: uniform aggregation (replaces `aggeregate` with `average` in this process only). Without attack: same partition and same 20 sampled attackers (RNG preserved), with the attacker list emptied right after initialization.
- **"Degrades FedAvg":** mean accuracy over rounds 81–100 at least **5 p.p.** below the no-attack run (same seed).
- **"Not fully excluded":** mean weight mass on attackers under `fixed` **> 0.01**, computed only over rounds with real attackers (reference: ≤ 0.0001 in B2.1).
- **Rule:** choose the ε that meets both criteria; if both meet them, the **smaller** one. If none meets both, the smallest ε that degrades FedAvg, **declaring that the result may be trivial** (fixed would exclude the attackers). If none degrades FedAvg, an unforeseen case: **stop and consult the author** before the grid.
- **ε is chosen by looking only at FedAvg and fixed; td3 never enters the sanity check.**
- Mechanical application: `scripts/passo2_oficial/analisar_b25.py sanity` → `sanity.txt`. The chosen ε is recorded in **Addendum 1** (hashed and committed) before launching the grid.

## 4. Grid

- MNIST, q = 0.5, 500 rounds, 100 clients, 10% per round, 20 attackers (same as B2.1).
- Attack: real IPM with the ε from Addendum 1. Conditions: `td3` (as in the official `main.py`) and `fixed` = [0.475]×5.
- **Seeds: 105–114** (confirmation in the official code; a new cell relative to B2.1).
- Total: **20 runs**, 5 in parallel on the GPU, ~27–31 h wall clock.
- Logging as in B2.1: states before each action, actor checkpoints every 50 steps, weight mass on attackers, resets. Truncation at 500 steps (SB3's td3 runs 501).

## 5. Hypothesis and criteria (same as B2.1's H1)

**Primary metric:** median test accuracy over rounds 401–500 (B2.1's, kept for comparability).
**Unit:** seed, n = 10. D = metric(fixed) − metric(td3).

**H1.** The fixed action is equivalent to TD3 under real IPM.
- Paired TOST (t), margin ±1.0 p.p., α = 0.05 → **equivalence confirmed**. Report CI90 (the TOST interval) and CI95.
- Paired Wilcoxon p < 0.05 with D < 0 → **TD3 contributes**.
- Wilcoxon p < 0.05 with D > 0 → the fixed action is better; report as "does not contribute".
- None → **inconclusive**; report Δ, CI90, CI95 and d.

**Secondary (descriptive):** mean 451–500; number of resets; mass on attackers per condition; drift |π₅₀₀ − π₀| and `sd_estados` of π₅₀₀ over the states of rounds 401–500 (compared with σ_a = 0.0475, no test).

**Power:** with n = 10, equivalence is only reachable if the sd of the differences is small (in B2.1 it was ~0.3 p.p. on the primary metric). An inconclusive result is reported as such.

## 6. Reading for the thesis

- Equivalence → the P2 conclusion ("TD3 does not contribute") extends to the third attack of the original paper, with the IPM fixed.
- TD3 contributes → limits the P2 conclusion to LMP/EB; report it prominently.
- If the chosen ε does not meet the inclusion criterion, the reading is conditioned on the declaration of possible triviality (§3).

## 7. Rules

- Pre-written analysis in `scripts/passo2_oficial/analisar_b25.py` (subcommands `sanity` and `grade`), run once with the complete runs.
- Runs that fail due to an environment error are repeated with the same seed and recorded. Collapses and resets count as results.
- No grid accuracy is inspected before the end. Monitoring reports only health and progress.
- Logs (`*.runner.log`, `*.stdout.log`) and intermediate actor checkpoints are not versioned (only steps 0 and 500).

## 8. Out of scope

**B2.7b cut** at the time: declared as a limitation. B2.4r and MedMNIST only when the author asks.
