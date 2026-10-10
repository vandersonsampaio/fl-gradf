> English translation of `PLANO.md`. The Portuguese original is the frozen record (its hash is in `PLANO.sha256`); if the two ever disagree, the original prevails.

# Plan — A0: cheap analyses

**Status:** FINAL before the runs and the analyses (hash in `PLANO.sha256`). Descriptive/exploratory.
**Date:** 2026-10-01
**Scope:** decides P3 directions 1 (signal selection) and 6 (exploit attackers instead of excluding them).

## Items

**(a) Weight mass on attackers** (`scripts/a0_analises.py massa` / `analisar`)
- B2.6 did not record per-client weights, so these are **new runs**: `sr_only` and `sr_bin` × 8 cells × seeds **52, 53 and 54** = **48 runs** (15 rounds, CPU).
- **Cells:** the 8 with a negative gap and the whole CI95 below 0 in C0 (methods under attack above the FedAvg-8 oracle): `gaussian_noise`, `krum_collusion` and `trim_attack` at α = 0.05 and 0.1; `sign_flipping` and `low_mag_backdoor` at α = 0.1.
- **Check:** the learner inherits B2.6's and only **records** weights and `is_byz`. The final accuracy of each run must exactly reproduce B2.6's (max. |Δ| < 1e-9). If it does not, item (a) is invalidated and reported as such.
- **Metric:** mass = sum of the normalized weights of the Byzantine clients per round, mean over the 15 rounds and per cell. Reference: uniform FedAvg gives 0.2.
- **Decision rule** (P3 direction 6, "exploit instead of exclude"): **real exploitation** if the mean mass is ≥ **0.05** in at least half of the 8 cells, for the variant considered. Otherwise, the C0 negative gaps are **not** explained by exploiting the attackers, and direction 6 loses its motivation.

**(b) Signal-selection ceiling** (B2.6 data, seeds 52–61, 19 valid cells)
- **Three estimates**, all as the mean over the 19 cells per seed:
  1. per-seed max between `sr_only` and `cosserver_only` (**biased upwards**: the maximum of two noisy values);
  2. per-cell signal choice, in-sample;
  3. per-cell choice **leaving the seed out** (LOSO, no selection bias).
- Compared with `sr_only`, `cosserver_only`, `sr_cosserver`, `sr_bin` and `sr_b025`, and with the best single signal per seed.
- **Use:** sizes the headroom that P3 direction 1 competes for. The **reference estimate is LOSO**.

**(c) Agreement between the per-client signals** (**preliminary**)
- **Source:** the ablation log (`results/frente1_ablacao_adaaggrl/detectores/`), which covers only the **td3 trajectory, seed 42**, on the 21 cells.
- **Measures:** mean per-round Spearman between S_R and `cos_server` across clients, and the AUC of each signal for separating the attackers.
- **Use:** candidate non-learned selector. Preliminary because it uses a single seed and another trajectory.

**(d) Summary table of the three TD3 regimes** (`scripts/passo2_oficial/a0_regimes_td3.py`, official venv)
- **Regimes:** published (B2.1), B2.3 and B2.3b.
- **Measures:** median drift and `sd_estados` per checkpoint (0–500), over the states of rounds 401–500. Uses the **local** intermediate checkpoints.
- **Use:** P2 figure.

## Cost and rules

- **(a):** 48 runs of ~62 process-seconds, in 3 processes (one per seed), `nice 19`.
- **(b) to (d):** minutes.
- **(a) runs before B2.7,** so as not to compete for CPU.
- No result is used to adjust the B2.5 and B2.7 plans, which are frozen separately.
