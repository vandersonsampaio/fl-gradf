# Plan — ablation of GRADF's hardening layers on MNIST under step parity

**Date:** 2026-11-02. **Status: frozen before the grid, after the checks of Sec. 4 passed (`checks.txt`).** Changes only through dated addenda. **Authorized by the author on 2026-11-02** to run alongside the W1 rerun with 5 processes, an exception to "P1 runs do not overlap W1".

**Why:** the GRADF repetition (`../gradf_repeat/RESULTS.md`) found GRADF − Random robust at about +3.3 p.p., but GRADF votes FLTrust in 87% of rounds and sits 5.81 p.p. above fixed FLTrust where it nearly always votes it. Random also beats the fixed rules (+4.41 p.p. against B₋ₛ). Fixed rules do not pass through Layers 1, 2 and 4, so the comparison against B₋ₛ mixes selection with hardening. FedStrategist and AdaAggRL do not pass through those layers either: each has its own `_run_round`, and `exp10` never uses `HardeningPipeline`.

**Status of the evidence:** re-analysis on seeds already seen, MNIST only. It informs the P1 pre-registration and is not confirmatory.

## 1. Questions

- **Q1 (primary): selection or hardening?** Does GRADF beat a fixed FLTrust that passes through the same layers?
- **Q2: like-for-like capture.** How do GRADF and Random compare with the cross-fitted best fixed rule when every fixed rule passes through the same layers?
- **Q3: which layer?** What are the effects of Layers 1, 2 and 4 on fixed FLTrust?
- **Q4: why does Random beat the fixed rules?** How much of Random's accuracy depends on the layers?
- **Q5: a reference for the unhardened systems.** How do FedStrategist and AdaAggRL compare with a Random without layers?

## 2. Arms

All arms use `GRADFFederatedLearner`, with the configuration of item 2 variant b: exp10 seeded, root 100, `server_steps = median`, 21 cells, 15 rounds, seeds 42–51, `GroundTruthClassifierStub`. No arm needs pretraining.

| arm | selector | Layer 1 | Layer 2 | Layer 4 | purpose |
|---|---|---|---|---|---|
| F000 | fixed FLTrust | off | off | off | baseline, learner check (Sec. 5) |
| F100 | fixed FLTrust | on | off | off | Q3 |
| F010 | fixed FLTrust | off | on | off | Q3 |
| F001 | fixed FLTrust | off | off | on | Q3 |
| F110 | fixed FLTrust | on | on | off | Q3 |
| F101 | fixed FLTrust | on | off | on | Q3 |
| F011 | fixed FLTrust | off | on | on | Q3 |
| **F111** | fixed FLTrust | on | on | on | **Q1, Q2, Q3** |
| M111 | fixed median | on | on | on | Q2 |
| T111 | fixed trimmed mean | on | on | on | Q2 |
| K111 | fixed Krum | on | on | on | Q2 |
| R000 | Random | off | off | off | Q4, Q5 |

**12 arms × 21 cells × 10 seeds = 2,520 runs.** Reused without rerunning: GRADF and Random from the 3 executions, fixed rules from `../mnist_fixed`, FedStrategist from `../mnist_rq3`, AdaAggRL from the published runs, as in `../RESULTS.md`.

**Not in this plan:**
- GRADF without layers, because its pretraining and online reward run through the pipeline, so removing them changes more than the layers.
- FedStrategist and AdaAggRL with layers, which would change their published definitions.
- Layer 5, a stub with no effect on the update.
- The Layer-3 rules outside D_R, as fixed arms.

## 3. Implementation

- **`scripts/p1/gradf_ablation.py`**, with no change to `src/`. A `HardeningPipeline` subclass takes the three switches. Off means: Layer 1 accepts every update, Layer 2 returns the update unchanged and draws no noise, Layer 4 accepts every round.
- **Selector:** `FixedActionSelector` (`exp1_baseline.py`), or `RandomSelector` as in exp10.
- **Seeds:** the run seed is `crc32(seed|alpha|attack|label)`, as in exp9 and exp10, with label = rule name (`fltrust`, `median`, `trimmed_mean`, `krum`) or `random`. All F arms share a seed. Arms that draw different amounts of noise diverge after the first draw, so pairing across layers is by seed, not by stream.
- **Audit trail:** each run writes to `/dev/null`. The shared `results/audit_trail.jsonl` already reached 167 MB, and nothing here reads it.
- **Logging:** per round, as in `gradf_repeat.py`: Layer-4 acceptance, clients rejected by Layer 1, old and new accuracy, global accuracy.
- **Launcher:** `scripts/p1/run_gradf_ablation.sh [N]`: one job per seed, resumable, a job starts only with at least 6 GB available, CPU, 1 thread per process. Outputs in `raw/`.

## 4. Checks before freezing (seed 42, α 0.05, `sign_flipping` and `gaussian_noise`)

1. **The switches change nothing when all are on.** Random with all layers on equals `../mnist_rq3` in both cells (0.5816, 0.7584), through the new subclass.
2. **Off means off.** Layer 4 on rejects rounds and off rejects none (`sign_flipping`). Layer 1 on rejects clients and off rejects none (`gaussian_noise`). Layer 2 off returns the update itself and draws no random numbers.
3. **Fixed selector.** Under F111, every round's voted rule is FLTrust, and two runs of F111 give the same accuracy.

If a check fails, the grid does not start and the author is told.

## 5. Learner check

**F000 against `../mnist_fixed` FLTrust at root 100**, 210 pairs. It tests whether the GRADF learner without layers behaves like the fixed-rule learner. Bit-level equality is not required, since the two learners may draw differently from the server's RNG. The mean difference with its CI95 is reported.

If the CI95 falls outside ±1.0 p.p., the learners differ by more than the layers. The cause is then reported, and the +5.81 p.p. of the repetition is not attributed to the layers.

## 6. Reading rule

All on the 18 confirmatory cells, per seed, using `inference.describe` (CI95 for superiority or inferiority, CI90 inside ±1.0 p.p. for equivalence). The 21-cell values are reported alongside.

**Q1.** Δ_e,s = mean over cells of GRADF(execution e) − F111, for each of the 3 executions.
- **Superiority in all 3:** the votes add accuracy beyond hardened FLTrust.
- **Equivalence in all 3:** GRADF's gain is explained by one good rule plus hardening, and adaptation is not shown.
- **Inferiority in all 3:** hardened FLTrust beats GRADF.
- **Anything else:** inconclusive. It is not settled by more runs without a new plan.

**Q2.** B₋ₛ^H is the cross-fitted best of F111, M111, T111, K111 (`inference.crossfit_references`). GRADF − B₋ₛ^H per execution, and Random − B₋ₛ^H. Same categories. The cell-wise captures use the same references.

**Q3 (descriptive).** Main effect of each layer in the 2³ on FLTrust: mean of the arms with the layer minus mean of the arms without it, per seed, with CI95 and Holm across the three. Two-way means are tabulated. Per attack and per α as well.

**Q4 (descriptive).** Random (`../mnist_rq3`) − R000 per seed, with CI95. Also Random − B₋ₛ^H and R000 − B₋ₛ.

**Q5 (descriptive).** FedStrategist − R000 and AdaAggRL − R000. This is like for like on the layers but not a full match, because those systems have other learners. AdaAggRL's rows come from the published, unseeded runs.

**Mechanism (descriptive).** Layer-4 rejection and Layer-1 rejection per arm and attack.

**Not attributed.** No comparison with published runs is attributed to the correction (`../PLAN.md`, notes).

## 7. What it decides

A proposal for the author, which does not change on its own any frozen plan:
- **The E3 families of the CIFAR-10 pre-registration.** If Q1 does not show superiority, E3 should compare GRADF with hardened fixed rules, and the capture claim should be restated.
- **Whether the campaign needs hardened or unhardened arms for each system.** Cost for CIFAR-10 is computed only after this result.
- **The manuscript's framing:** selection, hardening, or both.

## 8. Cost

Measured on 2026-11-02 with W1 running: one Random cell, 1 thread, no pretraining, 13.4 s including imports and data loading. The other arms have the same structure.

| | jobs | runs per job | assumption | wall time |
|---|---|---|---|---|
| alongside W1, 5 processes | 10 | 252 | 15–25 s per run | **2–3.5 h** |
| after W1, 10 processes | 10 | 252 | about 13 s per run | **about 1 h** |

- The range for 5 processes allows for the slowdown seen under load in the repetition (1.3 to 2.6 times the smoke-test time). That repetition ran alongside 4–6 W1 runs.
- Total CPU: about 9–18 h.
- Before and after the grid: code and checks about 1 h, analysis and write-up about 1–2 h.

## 9. Rules

- W1 is not touched. P1 runs do not otherwise overlap W1.
- Seeds 42–51 only. Seeds 62–71 and 200–209 stay reserved, and 220–239 belong to W1.
- No commit without the author's request.

## Code hashes (sha256)

```
764c8a8226fe28217ec05d68d5a3bea6d879607682f2352f89e0e52349f6cf23  scripts/p1/gradf_ablation.py
ad1c5b0e63c52129c3fed20a9e800a70083bf24bbaaf1e4b88b9d3a2c10921c7  scripts/p1/run_gradf_ablation.sh
```
