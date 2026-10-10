# Result — B2.8: oracle ceiling of per-round selection (granularity axis)

**Date:** 2026-09-30
**Plan:** `PLANO.md` (English translation in `PLANO.en.md`; hash in `PLANO.sha256`; `PLANO.md` and `scripts/b28_oraculo_por_regra.py` checked against the hash before the analysis: OK).
**Grid:** 210/210 runs (seeds 42–51 × 3 alphas × 7 attacks), 09/30 18:54 → ~19:15, CPU, no errors.
**Analysis:** `scripts/b28_oraculo_por_regra.py analisar` → `analise.txt`, `oraculo_por_celula.csv`. **Exploratory.**

---

## 1. Verdict by the PLANO criterion

- **Target cells** (the skeleton beats the best static fixed rule, W with CI95 > 0): **14/19**.
- In them, the per-round oracle is **below** the skeleton (Q with CI95 < 0) in **8/14** and **above** in **2/14** (`label_flipping` α = 0.1; `sign_flipping` α = 0.1).
- **PRE-REGISTERED VERDICT: "GRANULARITY EXPLAINS".** In most target cells, not even an ideal choice of rule every round, made by looking at the test set, reaches per-client filtering.

## 2. Robustness of the verdict (exploratory, not pre-registered)

The verdict **depends on three α = 0.5 cells with minimal differences**: `gaussian_noise` −0.12 p.p., `sign_flipping` −0.27 and `krum_collusion` −0.47. All have CI95 < 0, but are irrelevant in practice.

| requirement on \|Q\| | below the skeleton | above | reading |
|---|---|---|---|
| > 0 (PLANO criterion) | 8/14 | 2/14 | "granularity explains" |
| > 1 p.p. (practical relevance) | 5/14 | 2/14 | **inconclusive** (< 7) |

**Calibrated reading:** the strong evidence is concentrated at **α ≤ 0.1**, with model attacks in which S_R separates the clients. There the oracle is **4–14 p.p. below** the skeleton:
- `gaussian_noise` α = 0.05: −13.7 p.p.;
- `gaussian_noise` α = 0.1: −11.3 p.p.;
- `krum_collusion` α = 0.05: −10.8 p.p.;
- `krum_collusion` α = 0.1: −8.9 p.p.;
- `trim_attack` α = 0.05: −5.9 p.p.

At α = 0.5, skeleton and per-round oracle are practically tied. P2 should report the criterion's verdict **together with** this caveat.

## 3. Patterns per attack type

- **Model attacks with strong heterogeneity** (`gaussian_noise`, `krum_collusion`, `trim_attack` at α ≤ 0.1): per-client weighting wins comfortably even against the oracle. No single rule per round matches excluding the right clients.
- **Label attacks** (`label_flipping`): the oracle **beats** the skeleton (α = 0.1: +8.9 p.p.; α = 0.5: +11.9 p.p., the latter outside the target cells because there the skeleton already loses to Trimmed-Mean). Consistent with C.0: where S_R does not separate the clients, the choice of rule matters more than the per-client filter.
- **`low_mag_backdoor`:** the oracle beats the skeleton (+5.4 to +5.7 p.p. at α ≤ 0.1). The cell is not a target at α = 0.1 because there the skeleton does not beat the static rule. The ASR was not measured.
- **Gain of per-round selection over the best fixed rule (G):**
  - large at α ≤ 0.1: up to +19.7 p.p. on `sign_flipping` α = 0.1 and +17.5 on `label_flipping` α = 0.1;
  - small at α = 0.5: ≤ 2.7 p.p.;
  - negative on `gaussian_noise` α ≤ 0.1 (−7.1 and −7.7): the per-round greedy loses to the best fixed rule, a typical limitation of greedy choice.

## 4. What the oracle chooses

- **α ≤ 0.1:** overwhelmingly **FLTrust** (89–140 of 150 rounds per cell), the only rule in the arsenal anchored on the root dataset. **Clustering** comes second on the sign and trim attacks.
- **α = 0.5:** the choice varies by attack: Trimmed-Mean (`gaussian_noise`), Clustering (`trim_attack`, `sign_flipping`, `label_flipping`), FedAvg (`low_mag_backdoor`).
- Implication: the ceiling of per-rule selection under high heterogeneity depends on the **server (root) signal**. It is the same privacy trade-off as `cos_server` (B2.6/H4; the root dataset is an open data-governance question for the clinical setting).

## 5. Consequences for P2 and P3

- **P2 (granularity axis):** the difference that P1 attributed to "discrete vs. continuous" is, **at α ≤ 0.1 with model attacks**, a difference of **granularity**: one rule for everyone vs. one weight per client. Not even the ideal rule choice reaches the per-client filter. This is **not** a general claim: on label attacks, (idealized) per-rule selection beats the skeleton, and at α = 0.5 the two tie.
- **P3 (two-level direction):** the per-client filter wins where the signal separates, and the robust rule wins where it does not. This reinforces the **per-client filter followed by a robust rule** architecture.

## 6. Limitations

- **Exploratory:** already-used seeds; ablation and exp9 references already known.
- **The oracle uses the test set to choose:** it is a ceiling, not a method. This only strengthens the cells where it loses.
- **Greedy:** optimal per round, not over the trajectory (seen in the negative Gs on `gaussian_noise` α ≤ 0.1).
- **15 rounds:** gap mixed with convergence; B2.7 addresses the horizon.
- **ASR** of `low_mag_backdoor` not measured.

## Note of 2026-10-05 (weighting × robustness)

- Two of the per-round oracle's 7 arms (FedAvg and FedProx) weight by sample size; the other 5 and the skeleton do not.
- Under per-client label skew (α ≤ 0.1) and a balanced global test set, **standard FedAvg's sample-size weighting alone, without any attack, costs up to ~4.5 p.p.** relative to uniform weighting (oracle-8, α 0.1, H = 150: 85.0% × 89.5%; FedAvg-10: 1.0–1.8 p.p.). Tables that compare defenses with standard FedAvg (and B2.8's and B2.7's FedAvg/FedProx arms) must discount this effect, so as not to credit the defense's robustness with what is only weighting.
- In B2.8, the oracle chooses the rule by its own accuracy, every round, so it can avoid FedAvg/FedProx when sample-size weighting hurts.
- The metric (accuracy on the IID global test set; the clients' test sets are equal-size splits) does **not** by itself favor uniform weighting: B2.8s's metric sensitivity is vacuous by construction and reproduced the verdict (`results/b28s_sensibilidade_metrica/RESULTADO.md`).
