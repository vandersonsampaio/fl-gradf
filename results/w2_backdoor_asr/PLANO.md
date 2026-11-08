# Plan — W2: a real backdoor with Attack Success Rate (ASR) in the skeleton

**Status:** **CONFIRMATORY**. Final plan, frozen on 2026-10-23 before any code for this experiment was written (hash in `PLANO.sha256`). Changes only through dated addenda.
**Date:** 2026-10-23
**Batch:** W1–W4 (validity threats raised in `docs/experimentos_prioritarios_adaaggrl.md`). W2 is item 3 of that document.
**Environment:** in-house framework (logistic MNIST, CPU), the AdaAggRL ablation learner (`scripts/frente1_ablacao_adaaggrl.build_learner_cls`), the same one used in B2.6/B2.7/B2.7c.

## 0. Why this experiment

### 0.1 The existing `low_mag_backdoor` is not a backdoor

`AttackSimulator._low_mag_backdoor` (`src/classification/attack_simulator.py:390`) subtracts a shift from `update[target_class]`, with `target_class = 0`. On the flat parameter vector `[W_flat (784 × 10), b (10)]`, index 0 is `W[pixel 0, class 0]`. Pixel 0 is exactly 0 in every MNIST image in every shard used (α ∈ {0.05; 0.1; 0.5}, checked on 2026-10-23). So this coordinate never contributes to any logit.

The "attacker" is therefore an honest client whose update carries a marker on a dead coordinate. It has no trigger, so an ASR cannot be defined. Any accuracy difference reported under `low_mag_backdoor` reflects how the defenses treat near-honest clients, not how they resist a backdoor. This is the same kind of defect that the paper documents for the official IPM.

W2 replaces it with a real data-level backdoor, a trigger plus a target label, and measures ASR.

### 0.2 Question

> When a backdoor is actually being implanted, is TD3 equivalent to the non-adaptive controls in **security** (ASR), not just in clean accuracy?

It is possible that Acc(TD3) ≈ Acc(constant) while ASR(TD3) ≠ ASR(constant). In that case, equivalence in clean accuracy would not imply equivalence in security.

## 1. The attack (`pixel_backdoor`)

- **Level:** data, as with `label_flipping`. Each active Byzantine client trains locally on a poisoned copy of its shard. The update is the honest result of training on that copy, with no update-level manipulation.
- **Trigger:** pixels in rows 24–27 × columns 24–27 (a 4 × 4 square in the bottom-right corner) set to 1.0, on the [0, 1] scale used by the shards. This region is non-zero in 0.012% of the (image, pixel) pairs across all MNIST shards, so the trigger is a feature that honest clients essentially never activate.
- **Target class:** t = 0.
- **Poison fraction:** each round, a fresh random subset of p = 0.5 of the attacker's local samples is stamped and relabelled to t. The other samples keep their original pixels and labels. The subset is drawn from the `AttackSimulator` RNG, as in `label_flipping`.
- **Attackers and schedule:** clients [0, 1] (`BYZ` from B2.7). They are active in the same rounds as every other attack (`_is_active`).
- **Isolation:** the new attack is an addition. The code paths of all existing attacks must stay bit-identical; this is checked in §5.

## 2. Systems, cells, horizon and seeds

| system | definition |
|---|---|
| `fedavg_clean` | FedAvg, attack `none`. Gives the ASR floor (the natural rate at which triggered images are classified as t) and the clean-accuracy ceiling |
| `fedavg` | FedAvg under `pixel_backdoor`. The no-defense baseline |
| `td3_ref` | ablation learner in `td3` mode, as in B2.7c |
| `fixed_center` | the same skeleton with a constant action a = [0.5; 0.5; 0.5; 0.5], b = 0.5, and no TD3 |
| `fixed_b025` | the same skeleton with a = [0.5]×4, b = 0.25. This is the constant fixed **a priori** from B2.6 (seeds 52–61) and confirmed in B2.7c (seeds 82–91), so it is an independently selected "best constant" |
| `cosserver_only` | the skeleton with the server signal (FLTrust-style cosine), b = 0.5. Secondary: it tests the paper's reading that the server signal is better under backdoor attacks |

- **Cells:** α ∈ {0.05; 0.1; 0.5} (logistic MNIST, 10 clients, root of 100 with `root_seed = seed`).
- **Horizon:** H = 150 is primary. H = 15 and H = 50 are descriptive.
- **Seeds:**
  - development 92–96 (§4), used only for the validity gate;
  - confirmation **180–189** (n = 10 per cell). These are free in the seed registry and disjoint from every range used so far (42–91, 100–114, 115–124 reserved, 130–149, 200–209 reserved).
- **Pairing:** by seed. `_reseed(seed)` is called before each system, as in B2.7.
- **Grid:** 6 systems × 3 cells × 10 seeds = **180 runs**.

## 3. Metrics

- **Clean accuracy:** `global_accuracy` as in every previous framework experiment (mean over clients of the accuracy on each client's test shard).
- **ASR:** over the pooled test shards of the 10 clients, take every sample with true label ≠ t, apply the trigger, and compute the fraction that the global model classifies as t.
- **Recorded every round:** ASR_t, clean accuracy, and the aggregation weight mass on the attackers.
- **Secondary ASR metric:** the mean of ASR_t over rounds 1–150.

## 4. Attack validity gate (development seeds 92–96, before any confirmation)

**Only `fedavg_clean` and `fedavg` are run on 92–96.** No defense is run on these seeds, so the gate cannot be tuned against the comparison being tested.

**The attack is valid** if, in every cell at H = 150, both conditions hold:

1. it is effective: mean ASR(`fedavg`) − mean ASR(`fedavg_clean`) ≥ 50 p.p.;
2. it is stealthy: mean clean accuracy(`fedavg_clean`) − mean clean accuracy(`fedavg`) ≤ 5 p.p.

The second condition keeps the attack a backdoor rather than a degradation attack.

**Ladder** (pre-declared; the first rung that passes is used, and it is recorded in an addendum before confirmation):

| rung | trigger | p |
|---|---|---|
| 1 | 4 × 4 | 0.5 |
| 2 | 6 × 6 (rows and columns 22–27) | 0.5 |
| 3 | 6 × 6 | 0.8 |

If no rung passes, W2 stops. It is then reported as "no valid backdoor of this family was found for logistic MNIST", and the confirmation is not run.

## 5. Code checks (before the confirmation, recorded in an addendum)

1. **Regression:** 15 rounds on seed 42, `label_flipping` α 0.05, `td3_ref` and `fixed_b025`, must reproduce B2.7/B2.7b at H = 15 with |Δ| = 0. This shows that adding the attack did not change any existing path.
2. **Unit tests** for the trigger:
   - only the 16 (or 36) trigger pixels change;
   - labels become t on exactly round(p·n) samples;
   - the original arrays are not modified.
3. **ASR sanity:** a model trained only on triggered, relabelled data has ASR ≈ 1, and the untrained model has ASR ≈ the base rate.

## 6. Hypotheses and tests (fixed now)

The unit is the seed, and differences are paired by seed. For each constant c ∈ {`fixed_center`, `fixed_b025`} and each cell, two differences are computed:

- D_ASR = ASR(`td3_ref`) − ASR(c);
- D_acc = acc(`td3_ref`) − acc(c).

Negative D_ASR favors TD3.

**H1 (primary, security):** for each (c, cell), classify D_ASR with the paper's convention:

| verdict | condition |
|---|---|
| **TD3 more secure** | CI95 entirely below 0 |
| **TD3 less secure** | CI95 entirely above 0 |
| **equivalent** | TOST with margin **±5 p.p. of ASR**, i.e. CI90 inside ±5 |
| **inconclusive** | none of the above |

- The ±5 p.p. margin is a practically negligible change in attack success, fixed now, before any ASR is seen.
- Holm–Bonferroni correction over the 6 (c, cell) tests, separately for the TOST family and for the superiority/inferiority family (family α = 0.05).

**H2 (secondary, clean accuracy):** the same classification for D_acc, with the paper's MNIST margin of ±1 p.p. and Holm over the same 6 tests.

**Joint reading (pre-declared):** a (c, cell) pair where D_acc is **equivalent** but D_ASR is **not equivalent** is flagged as "accuracy equivalence does not imply security equivalence".

**Descriptive (no verdict):**
- the ASR reduction of each defense vs. `fedavg`;
- `cosserver_only` vs. `fixed_center` on ASR and accuracy;
- ASR_t curves;
- the attackers' weight mass;
- results at H = 15 and H = 50;
- one-sided Wilcoxon p-values and Cohen's d_z.

**Power:** there is no previous estimate of the between-seed sd of ASR. The confirmatory TOST may therefore be under-powered. If so, the outcome is "inconclusive", not "equivalent".

## 7. Rules

- **Mandatory order:**
  1. this plan, with its hash;
  2. the attack code, the unit tests and the checks in §5, recorded in addendum 1 with the code hashes;
  3. the validity gate on seeds 92–96, with the chosen rung, recorded in addendum 2;
  4. the confirmation grid.
- No ASR or accuracy from seeds 180–189 is inspected before the full grid (180/180) is complete.
- A failed run is re-run with the same seed. If it cannot be re-run, the cell is analysed with the available n, and this is stated.
- **Cost (estimated from B2.7c, ~30 min per skeleton run with 12 CPU processes):**
  - 4 skeleton systems × 30 runs = 120 runs, ~5 h of wall-clock time;
  - FedAvg runs take minutes;
  - the gate takes minutes.
- The grid respects the Mon–Fri 7:00–18:00 window (`scripts/janela_execucao.sh`).
- No commit. Files stay untracked until the author decides.
