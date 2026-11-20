# Result — W1 rerun: held-out evaluation of the global model, fresh seeds

**Date:** 2026-11-08.

**Status: CONFIRMATORY, valid verdict.** With the official test set kept out of the reward, the resets and the EB attacker's data, TD3 does not beat the fixed action on either dataset.
- **MNIST:** the two are equivalent at the end of training, and the fixed action is better over the whole run.
- **BloodMNIST:** the final-accuracy equivalence is a joint collapse, as the plan's reading rule requires. The fixed action is again better over the whole run.

**Pre-registration:**
- **Plans:** `PLAN.md` (frozen 2026-11-01) on top of `../w1_protocolo_independente_oficial/PLANO.md` (frozen 2026-10-23). `PLAN.md` matches `PLAN.sha256`.
- **Addendum:** `ADDENDUM1.md` (2026-11-01, before the checks and the grid) matches `ADDENDUM1.sha256`. The code hashes listed in it match the code that ran.
- **External code:** `external/AdaAggRL` at commit `27b9c18`, unedited.

**Checks before the grid:** `verificar_global` passed on 2026-11-01 at 20:23 (`verification.txt`, `VERIFICATION_PASS`). On seed 150, 25 rounds, the separate copy pointed at validation reproduced the logged accuracy exactly (max |Δ| 0.00000). `inner.net` differed from the global weights in 100% of the steps. On MNIST, mean |held-out − validation| was 0.50 and 0.80 p.p.

**Grid:**
- **Size:** 86/86 runs, 0 failures.
- **Seeds:** MNIST 220–229 and BloodMNIST 230–239; `fedavg`/`none` on 220–222 and 230–232.
- **Timing:** from 2026-11-01 20:23 to 2026-11-08 13:04 on the GPU.
  - It ran 6 runs at a time, then 4 from 2026-11-02.
  - It was paused at the author's request several times between 2026-11-03 and 2026-11-06, with SIGSTOP and SIGCONT. Each pause and resume is logged in `chain.log`.
- **Guard:** every run repeated check 1 at round 1 and every 10 rounds without a mismatch. The analysis refuses any run without `eval_target` = `global` or with a failed comparison.
- **Blindness:** no held-out or validation result of seeds 220–239 was looked at before 86/86. Monitoring read counts, errors and process state only.
- **Analysis:** `analisar_w1.py` ran once, on 2026-11-08 at 13:06.

## 1. Confirmatory tests

Δ = fixed − td3, in p.p., paired by (attack, seed), with n = 20 per dataset (LMP and EB pooled). The verdict rule (PLANO §5) is:
1. superiority or inferiority by the two-sided t test with Holm;
2. otherwise, equivalence by TOST with margins ±1 p.p. on MNIST and ±3 p.p. on BloodMNIST;
3. otherwise, inconclusive.

**H1:** the primary metric is the median held-out accuracy over rounds 401–500. Holm runs over the 2 datasets for each test type.

| dataset | Δ | CI90 | CI95 | d | TOST p (Holm) | t p (Holm) | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|---|
| MNIST | +0.08 | (−0.01; +0.17) | (−0.03; +0.19) | +0.35 | <0.0001 | 0.129 | 0.452 | **EQUIVALENT** |
| BloodMNIST | +1.47 | (+0.25; +2.70) | (−0.00; +2.95) | +0.47 | 0.022 | 0.101 (raw 0.050) | 0.070 | **EQUIVALENT → joint collapse** (Sec. 2) |

**H2:** the secondary metric is the mean held-out accuracy over rounds 1–500, with no correction.

| dataset | Δ | CI90 | CI95 | d | TOST p | t p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|---|
| MNIST | +2.10 | (+0.71; +3.48) | (+0.42; +3.77) | +0.59 | 0.907 | 0.017 | 0.070 | **fixed BETTER** |
| BloodMNIST | +0.80 | (+0.40; +1.21) | (+0.31; +1.29) | +0.77 | <0.0001 | 0.003 | 0.004 | **fixed BETTER** |

- **BloodMNIST H2:** this Δ lies inside the ±3 margin and is also significant, so it counts as superiority by the rule's order. It is small next to the margin.
- **BloodMNIST H1:** this Δ is close to the t test's threshold before Holm (raw p 0.050, CI95 lower bound −0.003). The sign again favours the fixed action.

## 2. Reading rules fixed in advance

**Joint collapse on BloodMNIST (PLAN §4).** Both conditions stay far more than 20 p.p. below `fedavg`/`none`:

| BloodMNIST | median 401–500 | mean 1–500 | rounds 401–500 below 50% | resets per run |
|---|---|---|---|---|
| LMP fixed | 28.58 | 27.43 | 100% | 16–22 |
| LMP td3 | 26.33 | 26.25 | 100% | 13–19 |
| EB fixed | 30.80 | 29.03 | 100% | 20–26 |
| EB td3 | 30.11 | 28.60 | 100% | 21–27 |
| none fedavg (n = 3) | 74.75 | 60.30 | 0% | 0 |

Every attacked run has a reset inside rounds 401–500. **The H1 equivalence on BloodMNIST is therefore reported as both conditions failing alike under attack, not as both defending.** The same collapse showed in the first W1's validation series (seeds 170–179, about 30% against 75%).

**Val–test gap, read against `fedavg`/`none`.** Median validation accuracy minus median held-out accuracy over rounds 401–500, averaged over seeds, in p.p.:

| dataset | attacked conditions | `fedavg`/`none` |
|---|---|---|
| MNIST | −0.37 to −0.15 | −0.25 |
| BloodMNIST | +0.37 to +0.45 | −0.43 |

On both datasets the gap is under 0.5 p.p. and close to the gap without any policy. Keeping the test set out of learning changes the level by less than half a point. It does not change the ranking.

## 3. Descriptive results

**Per attack, Δ on the primary metric (n = 10, no test registered):**

| dataset | attack | Δ | CI95 | Wilcoxon p |
|---|---|---|---|---|
| MNIST | LMP | +0.03 | (−0.10; +0.16) | 0.625 |
| MNIST | EB | +0.14 | (−0.06; +0.33) | 0.193 |
| BloodMNIST | LMP | +2.25 | (−0.63; +5.14) | 0.160 |
| BloodMNIST | EB | +0.70 | (−0.63; +2.02) | 0.322 |

**Means per condition, held-out, %:**

| dataset | attack | condition | median 401–500 | mean 1–500 | validation, median 401–500 | resets | attackers' weight mass |
|---|---|---|---|---|---|---|---|
| MNIST | LMP | fixed | 96.32 | 91.67 | 95.95 | 0.3 | 0.0000 |
| | | td3 | 96.29 | 91.64 | 96.07 | 0.3 | 0.0000 |
| | EB | fixed | 96.32 | 91.36 | 96.06 | 0.5 | 0.0000 |
| | | td3 | 96.18 | 87.20 | 96.03 | 2.4 | 0.0003 |
| | none | fedavg | 96.54 | 92.82 | 96.29 | 0 | — |

- **MNIST:** both conditions end within about 0.3 p.p. of FedAvg without attack, which leaves little room for the final median to separate them.
- **Where the H2 difference on MNIST comes from:** it comes from EB (mean 1–500 of 91.36 against 87.20), where TD3 starts slower and resets more often.

**TD3 behaviour (`mecanismo.csv`; deterministic actor at steps 0 and 500 on the states of rounds 401–500; σ_a = 0.0475):**
- The median drift is 0.010–0.012, which is about a quarter of σ_a.
- sd_estados is above σ_a only on BloodMNIST EB (0.066). It is already 0.067 at initialization, so it does not come from learning.
- This repeats B2.2, B3.2 and the first W1: the policy barely moves and does not react to the state.

## 4. Conclusion for P2

- **On MNIST, the conclusion of the old protocol holds when the evaluation set takes no part in learning.** TD3 gives no advantage over the fixed action b\* = center and loses on the area under the curve.
- **On BloodMNIST, the held-out evaluation also reproduces the old picture, with both conditions collapsing.** This means the official setup does not defend there, and it says nothing in favour of TD3.
- **The threat that the policy "sees" the test set is closed for this question.** Removing that access does not create a TD3 advantage on either dataset.

## Outputs

- `analise.txt`: the frozen analysis, run once at 86/86.
- `resumo_runs.csv`: one row per run.
- `mecanismo.csv`: TD3 behaviour metrics.
- `verification.txt`, `verificacao_global/`: the checks of PLAN §2.
- `grid.log` and `chain.log` (run log and pauses) are kept locally, not in the repository, like the other logs.
- `raw/`: one JSON per run, with `steps`, `held_out`, `global_checks` and `eval_target`; the states in `obs/`; actors at steps 0 and 500 in `actors/`.
