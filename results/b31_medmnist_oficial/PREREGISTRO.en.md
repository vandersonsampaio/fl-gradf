> English translation of `PREREGISTRO.md`. The Portuguese original is the frozen record (its hash is in `PREREGISTRO.sha256`); if the two ever disagree, the original prevails.

# Pre-registration — B3.1 + B3.2: fixed vs. TD3 in the official AdaAggRL, BloodMNIST

**Status:** FINAL, frozen **before any B3.0 or B3.1 run** (hash in `PREREGISTRO.sha256`). Changes after that only as a dated, hashed addendum.
**Date:** 2026-10-04
**Scope:** B3.0, B3.1 + B3.2 (health-dataset phase B′).
**Background:**
- B2.1 + B2.2 (`results/b21_replicacao_oficial/`): fixed ≈ td3 equivalence on MNIST; policy stuck and independent of its input.
- B2.4r (`results/b24r_limiar_oficial/`): the official code on GPU is not bit-reproducible, and a single late reset changes one seed's primary metric by ~8 p.p.

## 1. Questions

- **B3.1:** in the published code and horizon, with a health dataset (BloodMNIST), is the fixed action equivalent to AdaAggRL's TD3 within the margin M (§3)?
- **B3.2:** does the learned TD3 policy remain essentially the initial one and independent of its input (B2.2's H3–H5)?

## 2. Dataset and adaptation (done in B3.0, frozen before B3.1)

- **BloodMNIST** (MedMNIST v2): 28×28 RGB, 8 classes. Official train and test splits; the validation split is not used.
- **Official regime unchanged:** 100 clients, 10% per round, 20 attackers, q = 0.5, lr 0.05, 500 rounds, the official code's `_build_groups_by_q` partition.
- **Minimal adaptation,** done outside the official code (a shim or monkeypatch in the runner, as in the previous experiments; `external/AdaAggRL` is not edited):
  - a BloodMNIST branch in `construct_dataloaders`;
  - the official model with 3 input channels and 8 outputs;
  - **a new feature extractor** for the gradient inversion and the MMD cues, because the official `extract_feature.pt` is a 1-channel MNIST one. It is trained once in B3.0 with the same architecture and procedure as the official one (adapted to 3 channels and 8 classes), frozen, with its hash recorded in an addendum before B3.1.
- Any other necessary change is listed in the B3.0 addendum, with a justification.

## 3. What B3.0 must deliver, and the rules that use it (fixed now)

**B3.0 (sanity, development):** FedAvg with uniform aggregation (the B2.5 runner's `fedavg` mode) for 500 rounds, seeds **130–132**:
- BloodMNIST: no attack, LMP and EB (9 runs);
- **MNIST: no attack** (3 runs), as the reference for the margin rule.

Metric of each run: T = median accuracy over rounds 401–500.

**Convergence rule (B3.0 gate):**
- Uniform FedAvg without attack on BloodMNIST "converges" if the mean T over the 3 seeds is ≥ 50% (4× the 12.5% chance level) **and** the mean accuracy over rounds 451–500 differs from the mean over 401–450 by less than 2 p.p.
- If it does not converge: **stop and consult the author** before B3.1.

**Attack-inclusion rule:**
- An attack enters B3.1 if FedAvg under that attack is, on the mean over the 3 seeds, **≥ 5 p.p. below** FedAvg without attack (T). It is the same threshold as B2.5's sanity check.
- If only one attack passes, B3.1 runs with it alone (n = 10) and this is declared.
- If none passes: stop and consult the author.

**Margin rule M (equivalence):**
- On MNIST, the ±1.0 p.p. margin corresponds to the error level of an unattacked FedAvg. On BloodMNIST, it scales with the relative error:

  M = 1.0 p.p. × (100 − T_Blood) / (100 − T_MNIST)

  - T_Blood and T_MNIST are the mean Ts of FedAvg without attack over seeds 130–132.
  - M is rounded **up** to the next multiple of 0.25 p.p. and bounded to **[1.0; 3.0] p.p.**
  - If the formula gives more than 3.0, M = 3.0, with the declaration that equivalence at that margin is weak.
- M is computed **mechanically** by the B3.0 analysis script and recorded in an addendum (with hash and commit) **before** launching B3.1. No fixed or td3 run enters this computation.

## 4. B3.1 + B3.2 grid

- **Conditions:** `fixed` = [0.475]×5 (the center, as in B2.1) and `td3` (as in the official `main.py`: lr 1e-5, noise 0.1, `train_freq` 3, `batch_size` 64, `buffer_size` 1000, network [256, 128]).
- **Attacks:** LMP and EB (those that pass the §3 rule).
- **Seeds:** **135–144**. Total: 10 × 2 × 2 = **40 runs** (or 20 if only one attack passes).
- **Interleaved execution:** the queue alternates the conditions by seed and attack (td3 LMP, fixed LMP, td3 EB, fixed EB, for each seed). So each parallel GPU batch mixes fixed and td3, and both conditions are subject to the same variations of load, temperature and order. Running all runs of one condition before the other is forbidden.
- **Logging:** as in B2.1:
  - states observed before each action;
  - actor checkpoints at steps 0, 50, …, 500;
  - weight mass on attackers;
  - resets;
  - partials every 25 rounds.
- **Truncation:** 500 steps (SB3's td3 runs 501).

## 5. Metrics and hypotheses

**Unit:** attack × seed pair (n = 20; or n = 10 with a single attack). D = metric(fixed) − metric(td3).

**Primary metric:** median accuracy over rounds 401–500 (the same as B2.1).

**AUC sensitivity (mandatory, reported alongside):** area under the accuracy curve, normalized:
- mean accuracy over rounds 1–500;
- mean over rounds 251–500.

It captures the cost of resets over the whole run and does not depend on where the window falls.

**H1 (B3.1).** The fixed action is equivalent to TD3.
- Paired TOST (t), margin ±M, α = 0.05, on the primary → **equivalence confirmed**. Report CI90 and CI95.
- **Sensitivity:** the same TOST on both AUCs. If the primary and the AUCs disagree on the verdict, the result is reported as **"reset-sensitive"**, and equivalence is only claimed with that caveat.
- Paired Wilcoxon p < 0.05 with D < 0 → **TD3 contributes**; with D > 0 → the fixed action is better ("does not contribute").
- None → **inconclusive**; report Δ, CI90, CI95 and d.
- **Per attack** (descriptive): Δ, CI95, d, Wilcoxon with Holm.

**B3.2 (mechanism, family H3–H5, Holm, α = 0.05, one-sided; σ_a = 0.1 × 0.95 / 2 = 0.0475; identical to B2.2):**
- **H3:** correlation of the executed actions between EB and LMP of the same seed (rounds 101–500) > 0.9. Only if both attacks enter.
- **H4:** drift = mean |π₅₀₀(s) − π₀(s)| over the states of rounds 401–500 < σ_a.
- **H5:** S_swap = mean |π₅₀₀(s_t) − π₅₀₀(s′_t)| with the other attack's state < σ_a. Only if both attacks enter; otherwise, report S_shuffle as descriptive.

**Resets:** they are a consequence of the action and **count as results**. They are never excluded from the primary or the AUCs. Report, per condition, the number of resets per run and the fraction of runs with a reset inside the 401–500 window, with a descriptive paired Wilcoxon of the number of resets.

## 6. GPU non-reproducibility clause

- The official code on GPU is **not bit-reproducible** across executions (B2.4r ADENDO1 and RESULTADO). The seed fixes the initialization and the partition, not the trajectory. Pairing by seed pairs the **configuration**, and each run is a sample from that configuration's distribution.
- **No run with a complete final JSON is re-run**, for any reason (unexpected result, reset, collapse). Only runs that **fail without a final JSON** (environment error, killed process) are repeated with the same seed, and each repetition is recorded in `grid.log` and in the RESULTADO.
- The run-to-run variability (in B2.4r, ~8.6 p.p. on one seed's primary because of a late reset) is declared as a source of variance included in the CIs. It is also why the AUC sensitivity is mandatory.

## 7. Gate B′

- **Equivalence (H1) and mechanism (H4, H5) repeat** → P2 states the result on two datasets, one of them a health dataset.
- **They do not repeat** → report as a limit on the scope of the finding, with the observed direction and mechanism.
- **"Reset-sensitive"** → P2 states the mechanism and reports the equivalence with the caveat.

## 8. Rules

- The B3.1 + B3.2 analysis is pre-written (script with its hash in an addendum before the launch) and run once, with the complete grid.
- No B3.1 accuracy is inspected before the grid ends. Monitoring reports only health and progress.
- **Mandatory order:**
  1. this pre-registration committed;
  2. B3.0 implemented and run;
  3. addendum with dataset, extractor, included attacks, M and hashes;
  4. commit;
  5. B3.1 launch.
- Cost: B3.0's cost calibrates B3.1's (estimate: ~42 GPU hours for BloodMNIST).
- Logs and intermediate actor checkpoints are not versioned (only steps 0 and 500).
