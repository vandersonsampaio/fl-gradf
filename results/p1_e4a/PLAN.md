# E4a: temporal headroom, metric check on MNIST

**Date:** 2026-11-04. **Status: FROZEN on 2026-11-04, approved by the author, before the grid.** The checks of Sec. 4 passed on seed 42 (`checks.txt`). No grid job has run. The code and `PLAN.sha256` hashes below cover this version.

**Source:** `results/p1_plan/PLAN.md`, Sec. E4. That plan leaves the phase skeleton to this document and allows phases chosen deliberately from the rule ranking.

**Purpose:** show that the temporal-headroom metric responds in a regime built so that the best rule changes within a run. **The headroom here is constructed to validate the metric, and the text must say so. This is not a confirmatory claim.** The result feeds gate P-2.

**Not blind.** The phases and their rules come from the MNIST rerun on the same seeds (Sec. 2), whose results are known.

## 1. Definitions

- **Phase-wise greedy reference (G):** runs each phase's winning rule only during that phase and switches at the change point. The model carries the state left by the previous phase, so G is a reference, not a ceiling or an oracle. Another sequence may beat it. It is not deployable, because it knows the attack and the change points.
- **Phase score A:** the equal-weight mean, over phases, of the mean test accuracy in each phase's last 5 rounds.
- **Temporal headroom:** H_T = A(G) − A(best fixed rule over the whole run).

**Why the metric differs from E1–E3.** The trailing mean at the end of a run sees only the last phase, so a fixed rule that wins that phase would hide what happened before. The end-of-run trailing mean is reported as secondary.

## 2. Scenarios

Same attack sequence on both α, each attack for 15 rounds, R = 45: `sign_flipping` → `gaussian_noise` → `trim_attack`.

Each phase's rule is the winner on the trailing mean at root 100, over the 4 rules of D_R, seeds 42–51, from `../p1_parity/mnist_trailing/mnist_fixed`:

| α | phase | rule in G | mean, % | runner-up, % | gap, p.p. (SD over seeds) | same winner leaving one seed out |
|---|---|---|---|---|---|---|
| 0.1 | `sign_flipping` | FLTrust | 74.78 | TM 42.77 | 32.01 (3.13) | 10 of 10 |
| 0.1 | `gaussian_noise` | TM | 79.08 | FLTrust 74.44 | 4.64 (2.60) | 10 of 10 |
| 0.1 | `trim_attack` | FLTrust | 74.36 | median 39.85 | 34.52 (2.62) | 10 of 10 |
| 0.5 | `sign_flipping` | Krum | 88.16 | TM 86.82 | 1.34 (0.05) | 10 of 10 |
| 0.5 | `gaussian_noise` | TM | 89.26 | median 89.01 | 0.25 (0.05) | 10 of 10 |
| 0.5 | `trim_attack` | TM | 86.59 | Krum 86.56 | 0.03 (0.59) | **8 of 10** |

- **α = 0.1 is the main scenario:** the winner alternates and the gaps are large.
- **α = 0.5 is secondary:** only two rules appear, the gaps are small, and the last phase is close to a tie. It shows how the metric behaves with little contrast. A small H_T there is an expected result, not a failure.
- **Choices against the example skeleton:**
  - No attack-free phase. The ranking has no such cell, so its rule could not be taken from it, and it adds no contrast.
  - Phases of 15 rounds, the horizon on which the ranking was measured.
  - The sequence is the example's and is used on both α, without tuning per α.
- **Rough expectation at α = 0.1, not a criterion:** if state carried no effect, G would gain about 4.64 / 3 ≈ 1.5 p.p. over fixed FLTrust. Carry-over can move this either way.

## 3. Arms and settings

**Arms:** G, and FLTrust, median, trimmed mean and Krum fixed for the whole run. Adaptive systems stay out: the master plan lists them under E4b, and they are not needed to check the metric.

| setting | value |
|---|---|
| Model and data | logistic model, MNIST, 10 clients, 20% Byzantine, existing partition of each α |
| Root | 100, `root_seed` = seed |
| FLTrust's server update | `server_steps = median` |
| Global rate | factor 1, as adopted for MNIST |
| Local training | 5 epochs, lr 0.01 |
| Seeds | 42–51, the same as E1–E3 on MNIST |

- **Seed use:** these are also E4b's development seeds, so E4a cannot serve as E4b's confirmation.
- **Partition:** fixed per α, as in the rest of P1. The seed changes the root and the random stream, not the partition.
- **Paired random stream:** all arms of a scenario and seed start from the same RNG state, from `crc32("seed|alpha|e4a")`.

## 4. Code and checks before the runs

**New code, nothing in `src/`,** so the hashes frozen for the campaign and for `ADDENDUM_2` are unaffected:

- `scripts/p1/e4a_phases.py`: a learner that sets the attack, rule and strategy object at each round, using the blind or informed round of the existing learners as the attack requires. It saves per-round accuracy, attack and rule.
- `scripts/p1/run_e4a.sh`, resumable.
- `scripts/p1/analyze_e4a.py`.

**Checks, all required before the grid (seed 42):**

1. **Regression:** with one attack and one rule for 15 rounds and exp9's seed string, FLTrust and TM under `sign_flipping` and `trim_attack` at α = 0.1 match `mnist_fixed` exactly.
2. **Switch:** G equals fixed FLTrust at every round of its first phase at α = 0.1, on a shortened run. From the first round after the switch, they differ.
3. **Repetition:** two runs of G give identical per-round accuracy.
4. **Schedule:** the saved attack and rule match Sec. 2 at every round.

## 5. Analysis

Once, on the complete grid. Per α, not pooled, with per-seed differences on n = 10 (`inference.describe`, CI95 and Wilcoxon). No Holm family and no verdict category, since there is no confirmatory test.

- **H_T against two references:**
  - the in-sample best fixed rule, which is chosen on these runs and is conservative for H_T;
  - the cross-fitted rule, chosen on the other 9 seeds.
- **Secondary, reported in full:**
  - the mean of each phase's last 5 rounds for every arm;
  - whether each phase's rule in G still wins its phase after the first phase, measuring carry-over;
  - the end-of-run trailing mean (rounds 41–45) and the final round;
  - per-round curves, as a figure for the text.

**Reading, fixed now:**

- **CI of H_T above zero at α = 0.1:** the metric detects the constructed headroom.
- **CI includes zero or lies below it:** the sequence of static winners does not beat the best fixed rule. This is reported as a result about carry-over and as a limit of G.

**The scenario is not retuned on these results.** Any other sequence gets a new name and its own addendum, and counts as E4b development.

## 6. Cost

10 jobs, one per seed, with 5 arms × 2 scenarios × 45 rounds = 450 rounds each, 4,500 rounds in all. The measured rate is about 1.2 to 2.1 s per round under load, from the root-100 fixed-rule jobs of the rerun (2,205 rounds in 2,600 to 4,700 s).

- About 9 to 16 min per job, so **about 45 min to 1 h 20 on 2 processes**, plus the checks.
- CPU only (`CUDA_VISIBLE_DEVICES=""`), with the launcher's gate of at least 6 GB free.
- It runs on this PC from `fl-fedmed-p1`, inside the limit of 6 processes per execution while W1 exists. It does not take slots from the campaign.

## 7. Limits

- Built by design, on seeds whose results are known, so it says nothing about how often such regimes occur.
- One attack order and one phase length.
- One partition per α.
- MNIST only. Repeating on CIFAR-10 stays conditional on cost, and is outside this plan.
- G is not an upper bound, so H_T does not measure the most an adaptive system could gain.

## Code hashes (sha256)

```
0f702e04992ddd86fc4d2199839333694a21a9520da73e997c613a6fad14848c  scripts/p1/e4a_phases.py
d982cc54b59515ab0bccc2b8cf6b3e529d57cd4afa959797908d73d9fa9998ef  scripts/p1/run_e4a.sh
47f3566859d5639a72981f06cc8bca2757165b3a25160c3f38ce347968bbe0b3  scripts/p1/analyze_e4a.py
4f4e8699c2caba0d3731e3b18d337e4efb5651fe0f7c18c40f6895ac80f1e0a6  scripts/p1/inference.py
c7f1e6a34026e223c3a48b92bb82a429be174ec715c2b2b30652080774004a8b  src/fl/attacked_learner.py
e6fc76e4c437e1b2569ad10830b97181ebadb89623f952721d0e4820a69a67ec  src/fl/federated_learner.py
cfff7d3beba4173bc86587ce153ea95a6fedc51370fc8f9bb48f56c1b5a7f367  src/defense/aggregation_methods.py
82c5715018a5877374361e29386125c089bd875c6b8cf2c682bc25df87ed3993  src/classification/attack_simulator.py
d411c4773d682e9d56980f9e20d5f9dafe1e415b316db9772d3d9beef2968f04  src/experiments/exp9_dominance_grid.py
28285584be823b45ec3ea81c9ac14560e284c29ec635bc2dfe45688ef8da4c93  src/utils/data_loader.py
```
