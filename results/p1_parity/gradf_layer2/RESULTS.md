# Result — splitting Layer 2 into clipping and noise, MNIST under step parity

**Date:** 2026-11-03. **Status: complete, not confirmatory.** Plan: `PLAN.md`, frozen before the grid. 10/10 jobs, 0 failures, 08:50–09:36, 10 processes, with the W1 rerun paused; minimum available memory about 6 GB. Analysis: `scripts/p1/analyze_gradf_layer2.py`, output in `analysis/`.

**In short:** the whole gain of Layer 2 on fixed FLTrust comes from the noise. By the frozen rule the reading is **noise**. Clipping alone changes nothing. The logged norms fit a smaller effective server step, so Layer 2 should not be presented as a defense by itself. That a smaller step causes the gain is not shown.

## 1. Regression

F000 and F010 equal the ablation in all 420 pairs, with max |d| = 0.

## 2. Q1: noise or clipping

18 confirmatory cells, p.p.; arm means: F000 74.92, C010 74.93, N010 83.65, F010 83.64.

| contrast | mean (CI95) | category |
|---|---|---|
| N010 − F000 | +8.72 (+7.12; +10.32) | superiority |
| C010 − F000 | +0.00 (−0.01; +0.01) | equivalence |
| F010 − N010 | −0.00 (−0.02; +0.01) | equivalence |
| F010 − C010 | +8.72 (+7.11; +10.32) | superiority |

**Q1 reading: noise.**

- The gain grows as α falls: +11.29 at α 0.05, +9.57 at 0.10, +2.13 at 0.50.
- It appears under every attack: +6.82 to +9.96, and +1.30 under `fltrust_aligned`.
- Clipping binds only under `gaussian_noise` (2.5% of coordinates) and nowhere else. It changes the accuracy of that attack by +0.01 p.p.

## 3. Q2: the mechanism (descriptive, no test)

| mean over all rounds | F000 | N010 |
|---|---|---|
| ‖g₀‖ | 1.55 | 1.72 |
| ‖update‖, honest clients | 1.05 | 1.09 |
| ‖Δ‖ | 0.97 | 0.78 |
| **Step along g₀** | **0.338** | **0.113** |
| Trust, honest clients | 0.156 | 0.065 |
| Trust, Byzantine clients | 0.167 | 0.058 |

- **The hypothesis fits.** The step along g₀ is about three times smaller under noise while ‖g₀‖ and the honest clients' norm stay similar. By the last 5 rounds it is about six times smaller (0.348 against 0.055).
- **It does not stay constant.** Under noise the step falls every round, from 0.37 to 0.05. Without noise it stays around 0.23–0.37 after the first round.
- **The noise does not separate attackers from honest clients.** It lowers the trust of both. Its Byzantine mean includes `fltrust_aligned`, where the attacker copies g₀.
- **‖Δ‖ falls only 20%.** So under noise much of the aggregate is not along g₀, and that noise enters the global model.

## 4. What is not established

- **Causation.** These runs do not show that a smaller step produces the gain. A noise-free arm that shrinks the step would test it (plan, Sec. 7). It has not been authorized or run.
- **The CIFAR-10 CNN.** `legacy_dp_epsilon_for` holds the noise's total L2 norm constant across models, so Sec. 7 of the plan is too broad in saying the norm grows with the dimension: that holds only at a fixed ε, and P1's CNN rescales ε. The effect relative to the updates on the CNN has not been measured. FLTrust's swing of about 8 p.p. per round on CIFAR-10 makes this worth testing, but no link has been measured.
- **Generality.** FLTrust without Layers 1 and 4, logistic model, 15 rounds, seeds already seen, one partition per α.

## 5. Proposals for the author (plan, Sec. 7; nothing changes on its own)

- Do not present Layer 2 or hardening as a defense. Report that noise improves FLTrust on MNIST and that the mechanism fits a smaller step.
- Keep the hardened fixed arms in E3, because they are what make the comparison like for like. Their purpose is the comparison, not a defense claim.
- Run the noise-free arm before the pre-registration, with a step factor taken from these norms under its own addendum.

## Outputs

`analysis/analysis.txt`, `q1_contrasts.csv`, `q1_by_alpha.csv`, `q1_by_attack_type.csv`, `q2_all_rounds.csv`, `q2_last_5_rounds.csv`, `q2_step_by_alpha.csv`, `q2_step_by_attack_type.csv`, `q2_step_by_round.csv`; `raw/`.
