# Result — ablation of GRADF's hardening layers on MNIST under step parity

**Date:** 2026-11-03. **Status: complete, not confirmatory.** Plan: `PLAN.md`, frozen before the grid. 10/10 jobs, 0 failures, 2026-11-02 21:04 to 2026-11-03 00:46, 5 processes, alongside the W1 rerun; 16.6 CPU hours. Analysis: `scripts/p1/analyze_gradf_ablation.py`, output in `analysis/`.

**In short:** the gain of GRADF over the fixed rules is hardening, not selection. Against a fixed FLTrust that passes through the same layers, GRADF is −0.72 to −1.16 p.p. and never above it. By the frozen rule Q1 is **inconclusive**, because the three executions do not fall in one category. Layer 2 does most of the work and Layer 1 does nothing. Why Layer 2 helps is not established (Sec. 4).

## 1. Learner check (plan Sec. 5)

F000 − fixed FLTrust at root 100, 210 pairs: mean +0.01 p.p., per-seed CI95 (−0.48; +0.49), inside ±1.0 p.p. **PASS.** The learners agree on average, so the difference below can be attributed to the layers. They do not agree pair by pair: 204 of 210 pairs differ, by up to 21.54 p.p., and per-cell means differ by up to 2.98 p.p.

Under F111 and the other fixed arms every round votes the arm's rule, there are no Layer-3 errors, and the final accuracy equals the last round's global accuracy.

## 2. Q1 and Q2: selection against hardened fixed rules

18 confirmatory cells, p.p.:

| execution | GRADF − F111 (CI95) | TOST p | category |
|---|---|---|---|
| first run | −0.72 (−1.30; −0.15) | 0.151 | inferiority |
| new 1 | −1.16 (−1.88; −0.44) | 0.688 | inferiority |
| new 2 | −0.84 (−2.12; +0.45) | 0.390 | no evidence |

**Q1 reading: inconclusive** (frozen rule: anything other than one category in all 3). Superiority is absent in all three executions, and so is equivalence. On the 21 cells the values are −0.43, −0.86 and −0.68.

**Q2.** FLTrust is the hardened best rule in every fold (mean over 18 cells: FLTrust 83.53, trimmed mean 73.23, median 72.02, Krum 46.59), so B₋ₛ^H equals F111 and GRADF − B₋ₛ^H repeats the table above. **Q2 reading: inconclusive.** Random − B₋ₛ^H is −4.20 (−5.15; −3.26), inferiority. Cell-wise, 5 cells have cross-fitted headroom; GRADF captures 1–2 of them and Random 1.

| comparison, 18 cells | before (unhardened fixed rules) | like for like |
|---|---|---|
| GRADF against the best fixed rule | +7.45 to +7.89 | **−0.72 to −1.16** |
| Random against the best fixed rule | +4.41 | **−4.20** |
| GRADF over fixed FLTrust, 8 cells with ≥ 90% FLTrust votes | +5.81 | **−0.06** (above F111 in 3 of 8) |

GRADF − Random (+3.3 p.p.) is unchanged and remains like for like. It fits GRADF being close to F111 while Random mixes in worse rules. The largest loss against F111 is at α 0.05 under `trim_attack` (−5.5 p.p.), the cell where GRADF votes FLTrust least (64%).

## 3. Q3 and Q4: the layers

Main effects on fixed FLTrust, 18 cells, p.p., with Holm over the three (Wilcoxon 0.0059 for L2 and L4; 0.48 for L1):

| layer | effect (CI95) |
|---|---|
| Layer 1 | −0.02 (−0.10; +0.06) |
| Layer 2 | **+5.18 (+4.40; +5.97)** |
| Layer 4 | +3.47 (+2.65; +4.28) |

The effects are not additive:

| arm | F000 | F001 (L4) | F010 (L2) | F011 (L2 + L4) | F111 |
|---|---|---|---|---|---|
| mean, 18 cells | 74.92 | 81.85 | 83.64 | 83.56 | 83.53 |
| Layer-4 rejected rounds, 21 cells | 0% | 59.1% | 0% | 6.6% | 7.3% |

- **With Layer 2 on, Layer 4 adds nothing** (83.64 against 83.56). Layer 4 alone recovers most of the gain by rejecting most rounds.
- **The effects concentrate at low α.** Layer 2: +6.23 at α 0.05, +5.43 at 0.10, +2.23 at 0.50. Layer 4: +5.03, +3.95, −0.07.
- **Layer 1 rejects no honest client.** It removes both attackers under `gaussian_noise`, with no net accuracy gain under FLTrust (−0.15).

**Q4.**

| comparison, 18 cells | p.p. (CI95) | category |
|---|---|---|
| Random − R000 | +3.81 (+1.01; +6.62) | superiority |
| R000 − B₋ₛ | +0.60 (−2.88; +4.07) | no evidence |
| B₋ₛ^H − B₋ₛ | +8.61 (+6.88; +10.34) | superiority |

Without the layers, Random is no longer shown above the fixed rules, but the interval is wide and equivalence is not shown either.

**Q5 (not a full match, other learners).** FedStrategist − R000 is −0.85 (−4.02; +2.32), no evidence. AdaAggRL − R000 is +4.23 (+0.18; +8.27), from published, unseeded runs.

## 4. What is not established

- **Why Layer 2 helps.** The plan has one switch for clipping and noise. By the formula, Laplace(1/50) per coordinate over about 7,850 parameters has an L2 norm of about 2.5, and it is added to client updates but not to g₀. A hypothesis, **not tested**, is that the noise lowers the effective server step under FLTrust's rescaling, which would also fit Layer 4 rejecting far fewer rounds. If so, this is a step-size effect rather than a defense, it depends on the dimension of the model, and it need not transfer to the CIFAR-10 CNN.
- **Generality.** MNIST, logistic model, root 100, 15 rounds, seeds already seen, one partition per α. Short horizons favour step-size effects.
- **Independence of the executions.** As in the repetition, the three GRADF executions are far from independent, so "all 3" is a weak requirement.
- **GRADF without layers,** which the plan excludes. Its votes are only compared with fixed rules under the same layers.

## 5. Proposals for the author (plan Sec. 7; nothing changes on its own)

- **E3 families:** Q1 does not show superiority, so E3 should compare GRADF with hardened fixed rules, and "capture" should be restated. A claim of selection gain is not supported on MNIST.
- **Campaign arms:** a hardened FLTrust arm at least, and a hardened arm for each rule of D_R if the budget allows. The cost needs computing, since the budget is already at level 0 with the factorial.
- **Framing:** hardening as the main result, with selection as a hypothesis under test on CIFAR-10 and the attribution of Layer 2 stated as open.
- **Before the pre-registration:** a small addendum separating clipping from noise (F010-like arms with each one alone). Not authorized and not run.

## Outputs

`analysis/analysis.txt`, `q1_gradf_vs_f111.csv`, `q2_vs_hardened_best.csv`, `q3_main_effects.csv`, `q3_effects_by_attack_type.csv`, `q3_effects_by_alpha.csv`, `q4_q5.csv`, `mechanism_per_arm.csv`, `l4_rejection_by_attack.csv`, `per_cell.csv`; `raw/`.
