# Addendum 1 — decisions on the CIFAR-10 metric and the compute budget

**Date:** 2026-11-02. Taken by the author, closing items 1 and 2 of `RESULTS.md`, Sec. 7. Before any confirmatory CIFAR-10 run.

## 1. Metric

**Primary metric of the CIFAR-10 campaign (E1–E3): the trailing mean of accuracy over rounds 106–110 (R = 110, 5 rounds), for every rule and system.** R is unchanged.

- **Reason:** FLTrust swings about 8 p.p. per round on CIFAR-10 under parity (`RESULTS.md`, Sec. 5), so a single final round measures where the swing happens to land.
- **Decided on** block C of the pilot (seeds 0–2, 2 cells), not on confirmatory seeds.
- **The final-round accuracy at R is reported alongside as sensitivity.**
- **MNIST is unchanged:** final round, as before. The 5-round mean is not available in the existing MNIST runs.
- **The campaign's runners must record accuracy for at least rounds 106–110.** The guard checks it.
- **It does not remove FLTrust's instability.** Gains after R remain as reported in `RESULTS.md`, Sec. 5.

### σ_E0 recomputed

Same definition as `../p1_cifar10/e0b/PLAN.md`, Sec. 4, with the metric above. Block C with parity, 2 cells, 3 seeds, p.p.:

| quantity | final round at R | trailing mean, rounds 106–110 |
|---|---|---|
| σ_E0^E1 | 1.69 | **0.56** (0.5σ = 0.28) |
| σ_E0^E3 | 2.93 | **0.58** (0.5σ = 0.29) |
| D_s, E1 (root 2,500 − 100) | | −5.58, −5.54, −6.54 |
| D_s, E3 (FLTrust − FedAvg) | | −7.19, −7.29, −8.24 |

- The sensitivity margin of E3 falls below the primary 1.0 p.p. margin again, so it is the stricter TOST.
- **Limits:** 3 seeds and 2 cells, as in E0b. σ_E0^E3 remains a proxy, since no adaptive system ran in the pilot. These values are used only for the sensitivity margin. The primary margin is unchanged.
- τ_CNN (7.86) does not depend on this metric.

## 2. Budget

**Level 0 with the E1 factorial, k = 18:** 235.9 h against 236 h. The author accepts a small overrun rather than reducing the grid. No cell or seed is dropped if the campaign runs slightly past 236 h.

- This acceptance covers the grid as costed. New arms arising from the GRADF ablation (`gradf_ablation/PLAN.md`, Sec. 7) need a new costing and the author's decision.

## 3. Still open before the pre-registration

The result of the GRADF ablation, which may change the E3 families. The guard and the pre-registration come after it.
