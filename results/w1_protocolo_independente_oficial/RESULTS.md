# Result — W1: independent evaluation protocol in the official AdaAggRL

**Date:** 2026-11-01.

**Status: NO VALID CONFIRMATORY VERDICT.** The grid finished, but the held-out series does not measure the global model, so H1 and H2 were not tested as registered. The verdicts printed in `analise.txt` must not be cited. W1 has to be rerun with the corrected evaluation to answer its question. The rerun is planned in `../w1_rerun_global_eval/PLAN.md`; the scripts named here have since been changed for it, and the versions that produced this grid are in commit `ba98905`.

**Pre-registration:** `PLANO.md` (frozen 2026-10-23, `PLANO.sha256` checks out) and `ADENDO1.md` (parts A+B, 2026-10-25, before the grid). The hashes of the code in the addendum check out against the code that ran.

**Grid:** 86/86 runs, 0 failures, from 2026-10-25 09:07 to 2026-11-01 12:38, on the GPU. Six in-progress runs (seeds 165 and 175) were lost in the 2026-10-29 reboot after an OOM caused by other work, and were rerun from scratch by the resumable launcher. Every run has 500 rounds, the official commit `27b9c18` and `rng_intact` = true.

## 1. The defect

PLANO §1 requires the official test set to evaluate "the current global model after every environment step". `HeldOutAdapter.step` (`scripts/passo2_oficial/run_w1.py:136-145`) evaluates `self.inner.net` after the environment's step returns. By then `inner.net` no longer holds the global model:

1. `FL_mnist.step` aggregates, loads the result into `self.net` and evaluates it on validation (`external/AdaAggRL/exp_environments.py`). This is the `acc` logged in `steps`, and it is correct.
2. It then samples the next round's clients and trains them.
3. Its "Server's Operation" loop calls `set_parameters(self.net, weight_cid)` for each sampled client. Nothing restores the global weights afterwards.

So **the held-out series is the accuracy of the last sampled client's update for the next round**, which is an attacker's update when that client is an attacker.

## 2. The evidence

`scripts/passo2_oficial/diagnose_w1_heldout.py` (raw JSONs only, output in `heldout_diagnosis.txt`).

With 10 clients per round and the last one drawn at random, this defect predicts P(held-out accuracy < 50%) = n/10, where n is the number of real attackers aggregated at the **next** round. MNIST, attacked runs, rounds ≥ 100, validation accuracy > 90% at both rounds:

| n | rounds | P(held-out < 50% \| n at round i+1) | P(held-out < 50% \| n at round i) |
|---|---|---|---|
| 0 | 1,444 | **0.000** | 0.188 |
| 1 | 4,136 | **0.100** | 0.202 |
| 2 | 4,810 | **0.202** | 0.217 |
| 3 | 3,153 | **0.303** | 0.188 |
| 4 | 1,251 | **0.413** | 0.201 |
| 5 | 352 | **0.514** | 0.196 |
| 6 | 42 | 0.667 | 0.122 |
| 7 | 5 | 0.800 | 0.000 |

- The relation with the next round's sample matches the prediction, including zero low rounds out of 1,444 when no attacker is sampled. There is no relation with the same round's sample.
- Under `fedavg`/`none`, which has no attackers, the held-out series is below 50% in 1 of 1,200 rounds.
- In a single run (MNIST EB fixed, seed 160, rounds 471–500) the held-out series jumps between about 0.95 and about 0.09 while validation stays at 0.96–0.97.

A direct check, comparing `inner.net` with `inner.aggregate_weights` after a step, **was not run**. Available RAM was about 4 GB with the P1 parity chain running, and an earlier OOM froze this machine. The conclusion rests on the code and on the table above. The direct check is part of the correction (Sec. 6).

## 3. What is invalid

- **H1 and H2 as registered**, on both datasets, because the primary and the secondary metric come from this series.
- **The val–test gap** of PLANO §5. It compares the global model on validation with a client's model on test. The gap of about 8 p.p. on BloodMNIST under `fedavg`/`none` cannot be read as a difference between the two sets.
- **The per-attack Δ** in `analise.txt`.
- In `analise.txt` and `resumo_runs.csv`, every column named `primary` or `auc_1_500`, and `gap_val_minus_test`.

For the record, `analise.txt` prints EQUIVALENT for H1 on both datasets, "fixed BETTER" for H2 on MNIST and EQUIVALENT for H2 on BloodMNIST. Those labels describe client models, not the question of the plan.

## 4. What remains valid

The evaluation does not disturb the runs: the RNG states are restored (`rng_intact`), and the next step reloads the global weights before any use of the network. Therefore:

- The runs follow the protocol. The reward, the resets and the EB attacker's data use validation, and the policy never saw the official test set.
- `steps[].acc` is the global model on validation.
- Resets and the attackers' weight mass are valid.
- The TD3 behaviour metrics (`mecanismo.csv`) are valid.

## 5. Descriptive reading from the valid fields

**This is post hoc and is not a substitute for W1.** Validation feeds the reward, the resets and the EB attacker, so it cannot answer "does the conclusion hold when the evaluation set takes no part in learning?". Structurally this is the old protocol with a different set and fewer client samples.

Accuracy of the global model on validation (mean over seeds):

| dataset | attack | condition | median 401–500 | mean 1–500 |
|---|---|---|---|---|
| MNIST | LMP | fixed | 95.88 | 90.77 |
| | | td3 | 96.09 | 89.80 |
| | EB | fixed | 96.00 | 90.80 |
| | | td3 | 95.89 | 85.90 |
| | none | fedavg (n = 3) | 96.37 | 92.68 |
| BloodMNIST | LMP | fixed | 29.00 | 27.19 |
| | | td3 | 29.51 | 27.46 |
| | EB | fixed | 31.02 | 29.20 |
| | | td3 | 31.20 | 29.33 |
| | none | fedavg (n = 3) | 74.71 | 60.57 |

D = fixed − td3, paired, n = 20 per dataset (p.p.):

| dataset | metric | D | CI90 | CI95 | t p |
|---|---|---|---|---|---|
| MNIST | median 401–500 | −0.05 | (−0.16; +0.06) | (−0.18; +0.08) | 0.44 |
| MNIST | mean 1–500 | +2.93 | (+1.48; +4.38) | (+1.18; +4.69) | 0.002 |
| BloodMNIST | median 401–500 | −0.34 | (−1.38; +0.69) | (−1.59; +0.91) | 0.57 |
| BloodMNIST | mean 1–500 | −0.20 | (−0.52; +0.12) | (−0.59; +0.19) | 0.29 |

- These numbers point the same way as B2.1 and B3.1: no TD3 advantage, and on MNIST a slower start. No test or margin was registered for them.
- **BloodMNIST is a joint collapse.** Both conditions stay near 30%, against 75% for FedAvg without attack, with 17–24 resets per run and a reset inside rounds 401–500 in every run. The two conditions failing alike says nothing about either one defending.
- **On MNIST the final median has little room to separate the conditions:** both are within about 0.5 p.p. of FedAvg without attack.
- **TD3 behaviour** repeats B2.2/B3.2. The median drift is 0.010–0.013 against σ_a = 0.0475. The only sd_estados above σ_a is on BloodMNIST EB (0.084), and it is already there at initialization (0.083).

## 6. Why the checks missed it, and the correction

**Why it was missed.**
- Check 2 tests whether the evaluation disturbs the run, not what it measures.
- The GPU smoke test inspected the fields and deliberately not the held-out values.
- Nothing compared the evaluation against the logged validation accuracy.

**Correction (applied for the rerun, not to this grid):**
- Evaluate `inner.aggregate_weights` on a separate copy of the network, so that `inner.net` is left untouched.
- **New mandatory check:** the same evaluation path, pointed at validation, must reproduce `history["acc"][-1]` at every step of a short run, up to the GPU's numerical noise. `inner.net` versus `inner.aggregate_weights` is recorded in the same run.
- **Guard in the runner:** run that comparison at a fixed interval during the grid and abort on a mismatch, so a recurrence is caught early.

**Scope of the defect.** `run_w1.py` is the only script under `scripts/` that references `inner.net`. B2.x and B3.x read `history_acc`, which is evaluated right after aggregation. W2 and W3 use the in-house framework. Their results are not affected by this defect.

**Rerun.** All 86 runs are needed, because the global model's weights were not saved. The cost was about 7 calendar days this time, or about 130 GPU hours. Seeds 160–179 have now been looked at, both on validation and through the client-model series, so a fresh block (after checking the seed registry) is the cleaner confirmation.

## Outputs

- `analise.txt`, `resumo_runs.csv`: the frozen analysis as run once at 86/86. Kept as evidence; see Sec. 3.
- `mecanismo.csv`: valid.
- `heldout_diagnosis.txt`: output of the diagnostic.
- `raw/`: runs, with `held_out` as recorded.
