# Plan — W1: Independent Evaluation Protocol in the official AdaAggRL (reward on validation, final test held out, constant selected on disjoint seeds)

**Status:** **CONFIRMATORY**. Final plan, frozen on 2026-10-23 before any code for this experiment was written (hash in `PLANO.sha256`). Changes only through dated addenda.
**Date:** 2026-10-23
**Batch:** W1–W4 (`docs/experimentos_prioritarios_adaaggrl.md`, items 1 and 2, merged into a single confirmatory protocol as that document recommends).
**Environment:** official AdaAggRL code (`external/AdaAggRL`, commit 27b9c18, unedited), isolated venv, GPU, via hooks outside the official code (the same approach as `run_oficial.py` and `bloodmnist_shim.py`).

## 0. Why this experiment

In the official code, the **same test set** plays four roles:

1. **Reward:** `exp_environments.py:142`, r = loss_{t−1} − loss_t on `self.testloader`.
2. **Resets:** `exp_environments.py:267–272`. The environment resets when the reward falls below −80, so the reset trigger is also a test-set quantity.
3. **The EB attacker's data:** `attack_utilities.py:399–415`. `EB_attack` trains on `testloaders[cid]`, which are q-grouped slices of the test set.
4. **The reported accuracy:** `history_acc`, from which the primary metric (median over rounds 401–500) is computed.

So the set that measures the final result takes part in training the policy, in triggering resets, and in building the attack.

> Does the conclusion TD3 ≈ fixed still hold when the test set no longer takes part in learning?

Item 2 of the document, a constant chosen independently, is handled in §2.

In the in-house framework this problem does not exist. The reward uses the server root (`server_val`, 100 samples), and accuracy uses the clients' test shards (`exp10_selector_comparison.py:377–382`). W1 is therefore only about the official code.

## 1. The protocol (hooks, official code untouched)

**Validation set:**
- It is a held-out part of the official **non-test** pool, with the same size as the official test set, so that the reward keeps its scale. The reward sums the per-batch losses (it is not normalized), so it scales with the number of batches (156 batches of 64 for 10,000 samples). The −80 reset threshold, which is calibrated to that scale, stays meaningful.
  - **MNIST:** 10,000 from the 60,000 training samples, leaving 50,000 for the clients.
  - **BloodMNIST:** 3,421 (= |test|) from the official train + val pool (11,959 + 1,712 = 13,671), leaving 10,250 for the clients.
- The indices are drawn with `np.random.RandomState(20261023)`, independent of the run seed and of Python's `random`, whose state after `random.seed(150)` drives the official partition.
- The validation set uses the **non-augmented** transform (the official test transform). Note that `_build_groups_by_q` materializes the transformed tensors once, so an augmented validation set would be frozen with one random crop.

**What the environment receives:** (train_rest, validation) in the place of (trainset, testset), by wrapping `construct_dataloaders`. As a result, the reward, the resets, the EB attacker's slices and `history_acc` all use **validation**.

**Held-out test (official test set):** it is used only to evaluate the current global model after every environment step (accuracy and loss). This is logging only, and the evaluation must not consume the global RNG:
- the DataLoader gets a dedicated `torch.Generator`;
- the torch CPU/CUDA RNG states are saved and restored around the evaluation, and checked in the smoke test.

**Unchanged:** the TD3 configuration, the actions, the attacks (LMP and EB, official), q = 0.5, 500 rounds, attackers via `random.seed(150)`, and `set_random_seed(seed)` as in `run_oficial.py`/`run_b3.py`.

## 2. Selection of the constant (item 2): Phase A from existing development sweeps

**Rule (fixed in this plan):** b\* = the a₅ with the highest mean primary metric (median accuracy over rounds 401–500) in the development sweep of each dataset, with the other four action components at the center (0.475).

| dataset | sweep | seeds | a₅ values | mean primary | b\* |
|---|---|---|---|---|---|
| MNIST | B2.4r (EB) | 100–104 | {0; 0.1; 0.25; 0.475; 0.95} | 0.475 = 96.7%, 0.25 = 96.6%, 0.95 = 96.3% | **0.475** |
| BloodMNIST | B3.3 (EB) | 145–149 | {0.475; 0.75; 0.95} | the center beats 0.75 (−0.64) and 0.95 (−0.14) | **0.475** |

So **b\* = the center on both datasets.** In the official environment, "fixed action" and "best constant selected on disjoint seeds" coincide, and W1 tests both items with the same arm.

Limitations, stated here and to be stated in the paper:
- (a) This rule is written down after both sweeps were seen.
- (b) The sweeps ran under the old protocol, with test-based resets and the EB attacker's data taken from the test set.
- (c) They cover EB only.
- (d) The BloodMNIST sweep only covers a₅ ≥ center.
- (e) A different rule would change b\* on BloodMNIST: the mean AUC over rounds 1–500 would pick 0.75 (+0.84 p.p., CI95 −0.56; +2.23).

What makes the selection legitimate is that it uses seeds disjoint from the confirmation and that the rule is the paper's primary metric. A fresh Phase A under the new protocol would cost ~60 GPU hours (≥ 40 runs) and is **not** run, unless the author decides otherwise.

## 3. Conditions, seeds and grid (Phase B, confirmation)

| condition | definition |
|---|---|
| `td3` | SB3 TD3 exactly as in the official `main.py` (as in `run_oficial.py`) |
| `fixed` | the constant action [0.475]×5 (= b\*) |
| `fedavg_none` | uniform aggregation, attack `none` (as in B3.0). It is the no-attack reference, descriptive only |

- **Attacks:** LMP and EB.
- **Datasets:** MNIST and BloodMNIST. BloodMNIST uses `bloodmnist_shim`, unchanged.
- **Seeds:**
  - MNIST **160–169**;
  - BloodMNIST **170–179** (n = 10 per attack, n = 20 per dataset when the attacks are pooled);
  - `fedavg_none` on 160–162 (MNIST) and 170–172 (BloodMNIST).
  - All these ranges are free in the seed registry.
- **Grid:** 2 datasets × 2 attacks × 2 conditions × 10 seeds = 80 runs, plus 6 `fedavg_none` runs, for **86 runs** in total.

## 4. Metrics

- **Primary:** the median **held-out test** accuracy over rounds 401–500.
- **Secondary:** the mean held-out test accuracy over rounds 1–500 (AUC).
- **Also recorded:**
  - the validation accuracy and loss (what the agent sees);
  - the val–test gap (descriptive, to quantify how much the reuse of the test set mattered);
  - resets, the attackers' weight mass, and the behavioural metrics (drift and sd_estados vs. σ_a = 0.0475, computed with `predict(deterministic=True)` as in B2.2/B3.2).

## 5. Hypotheses and tests (fixed now)

Δ = primary(`fixed`) − primary(`td3`), paired by seed. Positive values favor the fixed action, which is the paper's convention for the official code.

**H1 (primary, per dataset, LMP + EB pooled, n = 20):** equivalence by TOST (two one-sided tests at α = 0.05; CI90 inside the margin).
- Margins: **±1 p.p. (MNIST)** and **±3 p.p. (BloodMNIST)**, the paper's pre-registered margins.
- Holm–Bonferroni over the 2 datasets.
- The result is classified with the paper's convention: superiority or inferiority by the CI95, equivalence by the CI90, otherwise inconclusive.

**H2 (secondary):** the same test on the AUC 1–500, with no correction.

**Descriptive:**
- per-attack Δ;
- the val–test gap per condition;
- drift and sd_estados;
- resets;
- the comparison with B2.1 (MNIST, seeds 105–114) and B3.1 (BloodMNIST, 135–144) under the old protocol.

That comparison is **not** like-for-like: the clients now have 50,000 samples instead of 60,000 (MNIST) and 10,250 instead of 11,959 (BloodMNIST), and the evaluation set is different.

## 6. Checks before the grid (addendum 1)

1. **Sizes and disjointness:** validation ∩ test = ∅ and validation ∩ train_rest = ∅, with the expected sizes.
2. **The held-out evaluation does not touch the global RNG:** compare the torch, numpy and random states before and after the evaluation. Also, a 5-round CPU run with and without the held-out evaluation must give identical histories (|Δ| = 0). This is checked on CPU because the GPU is not bit-reproducible (B2.4r).
3. **Reward scale:** on round 1, the number of validation batches equals the number of test batches (156 for MNIST).
4. A 10-round GPU smoke test per dataset, td3 and fixed, completes and writes all the fields.

## 7. Rules and cost

- **Mandatory order:** this plan with its hash → code and checks (addendum 1 with the hashes) → grid.
- No held-out test result from seeds 160–179 is inspected before the full grid (86/86) is complete.
- **Cost:** the measured official throughput is ~1.5 GPU hours per run, even with 6 processes in parallel (Step 2: 30 runs in 44 h; B3.1: 40 runs in ~62 h). 86 runs ≈ **130 GPU hours**.
  - With the Mon–Fri 7:00–18:00 window (~125 usable hours per week), that is about **1 to 1.5 calendar weeks**.
  - The queue is interleaved by dataset, attack and condition, and launched with `setsid` + `scripts/janela_execucao.sh`.
- **A cheaper option if the author prefers:** EB only (LMP is fully filtered in every previous run), giving 46 runs ≈ 70 GPU hours. It is not the default, because the document asks for LMP and EB.
- No commit.
