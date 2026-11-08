# Plan — W3: a simple adaptive attacker against the skeleton's threshold

**Status:** **CONFIRMATORY**. Final plan, frozen on 2026-10-23 before any code for this experiment was written (hash in `PLANO.sha256`). Changes only through dated addenda.
**Date:** 2026-10-23
**Batch:** W1–W4 (`docs/experimentos_prioritarios_adaaggrl.md`, item 4). W3 is exploratory in scope. Whether it enters the paper is decided at writing time. It runs only because its cost is low (CPU).
**Environment:** in-house framework (logistic MNIST, CPU), the AdaAggRL ablation learner, as in B2.7c.

## 0. Why this experiment

The paper lists "adaptive attack against the agent" as out of scope. Every attack evaluated so far is blind to the defense.

The skeleton excludes a client when its normalized score is at or below δ = b · max k̃. A constant b gives a fixed target that an attacker can learn to stay above. TD3, in principle, can move b. So this is the scenario in which adaptation is most likely to pay off.

> When the adversary deliberately exploits the exclusion mechanism, does TD3 gain anything over a constant policy?

## 1. The attack (`adaptive_threshold`)

The attacker is "informed", like `trim_attack`/`krum_collusion`: it sees the honest updates of the round through `compute_param_updates_auto`.

For each active Byzantine client i at round t, the attacker submits a mix of its honest update and a malicious one:

  u_i = (1 − ε_i,t) · h_i + ε_i,t · m_t,   with m_t = −c · h̄_t,

where:
- h_i is the client's honest update;
- h̄_t is the mean honest update of the other clients;
- c = 2, as in the corrected IPM of B2.5.

At ε = 0 the client is honest; at ε = 1 it sends the pure IPM-2 update.

**Feedback (white-box acceptance oracle):** after aggregation, attacker i learns whether its final weight was > 0. This is a strong but simple assumption, stated as such. A real attacker would have to infer acceptance from the change in the global model.

**Adaptation (AIMD, per attacker):**
- ε_i,1 = 0.1;
- if accepted, ε ← min(1, ε + 0.1);
- if excluded, ε ← ε / 2.

**Non-adaptive reference attack `ipm2`:** ε = 1 in every round.

## 2. Systems, cells, horizon and seeds

| system | definition |
|---|---|
| `fedavg` | FedAvg (no defense; the attacker is always accepted, so ε → 1) |
| `td3_ref` | ablation learner in `td3` mode |
| `fixed_center` | a = [0.5]×4, b = 0.5, no TD3 |
| `fixed_b025` | a = [0.5]×4, b = 0.25 (fixed a priori in B2.6/B2.7c) |

- **Cells:** α ∈ {0.05; 0.1; 0.5}.
- **Horizon:** H = 150 is primary; H = 15 and H = 50 are descriptive.
- **Attackers:** clients [0, 1].
- **Seeds:**
  - development **125–129** (§4);
  - confirmation **190–199** (n = 10).
  - Both ranges are free in the seed registry. 115–119 is avoided because it is reserved for the TRIAGE adaptive-attack development.
- **Pairing:** by seed, with `_reseed(seed)` before each system.
- **Confirmation grid:** 4 systems × 3 cells × 10 seeds = **120 runs**, all under `adaptive_threshold`.

## 3. Metrics

- **Clean accuracy at H = 150:** primary.
- **Recorded every round:** each attacker's ε_i,t, its acceptance, its weight mass, and the threshold b actually used (for TD3, its action a₅).

## 4. Validity gate (development seeds 125–129)

Only `fixed_center` and `fedavg` are run on the development seeds, under both `adaptive_threshold` and `ipm2`. TD3 is **not** run on 125–129.

**The attack exploits the mechanism** if, in at least 2 of the 3 cells, both conditions hold:

1. it beats its blind version: mean acc(`fixed_center`, adaptive) ≤ mean acc(`fixed_center`, `ipm2`) − 2 p.p.;
2. it gets through the filter: the attackers' mean acceptance rate under `fixed_center` is ≥ 30%.

**Ladder** (pre-declared; the first rung that passes is used, and it is recorded in an addendum):

| rung | c | additive step |
|---|---|---|
| 1 | 2 | +0.1 |
| 2 | 4 | +0.1 |
| 3 | 4 | +0.05 |

If no rung passes, W3 stops and is reported, as exploratory, as "the simple adaptive attacker does not exploit the threshold rule". The confirmation is not run.

## 5. Code checks (addendum 1)

1. **Regression:** 15 rounds on seed 42, `label_flipping` α 0.05, `td3_ref`, must reproduce B2.7 with |Δ| = 0. This shows that the feedback hook does not change existing paths.
2. **Unit tests:**
   - ε = 0 gives exactly h_i, and ε = 1 gives exactly −c · h̄;
   - the AIMD update and the bounds on ε;
   - the hook reports acceptance = (final weight > 0).

## 6. Hypotheses and tests (fixed now)

D = acc(`td3_ref`) − acc(c), paired by seed, for c ∈ {`fixed_center`, `fixed_b025`} and each cell (6 tests).

**H1 (primary):** does TD3 beat the constant? Per (c, cell), the paper's convention is applied:

| verdict | condition |
|---|---|
| superiority | CI95 entirely above 0 |
| inferiority | CI95 entirely below 0 |
| equivalence | TOST ±1 p.p., CI90 inside the margin |
| inconclusive | none of the above |

Holm–Bonferroni over the 6 tests, separately for each family (family α = 0.05).

**Overall reading (pre-declared):**
- "adaptation pays off against this adversary" if TD3 is superior to **both** constants in ≥ 2 of 3 cells;
- "no advantage" if it is not superior to `fixed_b025` in any cell.

**Descriptive:**
- the ε trajectories;
- the acceptance rate per system;
- whether TD3's a₅ moves in response to the attacker (drift and sd_estados, as in the paper);
- `fedavg` vs. the defenses.

## 7. Rules

- **Mandatory order:** this plan with its hash → code, tests and checks (addendum 1) → gate on 125–129 (addendum 2) → confirmation.
- No result from seeds 190–199 is inspected before the full grid (120/120) is complete.
- **Cost:** ~30 min per skeleton run with 12 CPU processes.
  - gate: 2 systems × 2 attacks × 3 cells × 5 seeds = 60 runs, about half of them skeleton runs, ~1.5–2 h;
  - confirmation: 90 skeleton runs, ~4 h.
- The grid respects the Mon–Fri 7:00–18:00 window. No commit.
