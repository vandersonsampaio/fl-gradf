# Plan — W1 rerun: held-out evaluation of the global model, on fresh seeds

**Status:** CONFIRMATORY. Frozen on 2026-11-01 before the code changes below and before any run on these seeds (hash in `PLAN.sha256`). Changes only through dated addenda.

**Why:** the first W1 grid produced no valid verdict. Its held-out series measured the last sampled client's update instead of the global model (`results/w1_protocolo_independente_oficial/RESULTS.md`). The author ordered this rerun on 2026-11-01.

**Base plan:** `results/w1_protocolo_independente_oficial/PLANO.md` (frozen 2026-10-23) applies in full, including the split, the conditions, b\* = center on both datasets, the metrics, H1 and H2 with their margins and tests, and the descriptive list. Only the items below change.

## 1. What changes

| item | first W1 | this rerun |
|---|---|---|
| Held-out evaluation | `inner.net` after the step (last sampled client) | **The global model:** `inner.aggregate_weights` loaded into a separate copy of the network. `inner.net` is not touched |
| Seeds | MNIST 160–169, BloodMNIST 170–179 | **MNIST 220–229, BloodMNIST 230–239** |
| `fedavg`/`none` seeds | 160–162, 170–172 | **220–222, 230–232** |
| Output folder | `results/w1_protocolo_independente_oficial/` | `results/w1_rerun_global_eval/` |
| Pause window | Mon–Fri 8–18 | **None.** The author cancelled it on 2026-11-01 and will say when to pause and resume |
| Start | | As soon as the P1 parity chain ends, after the checks of Sec. 2 pass |

**Seeds:** 220–239 appear in no plan or result of `fl-gradf`, the P1 worktree or `fl-fedtriage` that a search found. 200–209 are reserved for TRIAGE's final test and are not used.

**What was already seen.** Validation accuracy of the global model and the client-model series on seeds 160–179 were inspected, and they suggest no TD3 advantage. Margins, tests and b\* are not changed in response. The new seeds were not seen by anyone.

## 2. Checks before the grid (`run_w1.py verificar_global`)

Development seed 150, 25 rounds, GPU, EB: MNIST `fixed`, MNIST `td3`, BloodMNIST `fixed`. Each run also records the old measure (`inner.net`) for comparison. **PASS** requires all of:

1. **The evaluation path measures the global model.** The separate copy, pointed at validation, reproduces the logged `history["acc"][-1]` within 0.001 at every checked step.
2. **The defect is gone for a stated reason.** `inner.net` differs from the global weights after at least 90% of the steps, which also confirms directly the diagnosis of the first W1.
3. **MNIST:** mean |held-out − validation| of the global model ≤ 3 p.p. over the 25 rounds.
4. `rng_intact` = true in every run.
5. All fields are written, with `eval_target` = `global`.

If a check fails, the grid does not start and the author is told. The statistical check of addendum 1 (evaluation does not perturb the run) is not repeated: the new path does not touch `inner.net`, and criterion 4 covers the RNG.

## 3. Guard during the grid

Every 10 rounds, and at round 1, each run repeats check 1. A mismatch aborts that run with an error, and the run is reported as FAIL. Each run stores the comparisons in `global_checks`.

## 4. Analysis

`analisar_w1.py`, run once at 86/86, with its paths and seeds updated. It refuses runs without `eval_target` = `global` or with a failed comparison. One descriptive line is added: the fraction of rounds 401–500 with held-out accuracy below 50%.

**Reading fixed in advance:**
- If both conditions stay more than 20 p.p. below `fedavg`/`none`, an EQUIVALENT verdict on that dataset is reported as a joint collapse, not as both conditions defending.
- The val–test gap is read against `fedavg`/`none`, which gives the gap without any policy.

## 5. Rules

- No held-out or validation result of seeds 220–239 is inspected before 86/86. Monitoring reads counts, errors and process state only.
- Cost: about 130 GPU hours, about 5.5 days without pauses.
- No commit without the author's request.
