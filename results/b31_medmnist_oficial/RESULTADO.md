# Result — B3.1 + B3.2: fixed vs. TD3 in the official AdaAggRL, BloodMNIST

**Date:** 2026-10-09
**Pre-registration:** `PREREGISTRO.md` (`315b880`; English translation in `PREREGISTRO.en.md`), with addenda 0, 1 and 2 (English translations in `*.en.md`). `analisar_b31.py` checked against Addendum 1's hash before running (`dff2220a…`, identical).
**Grid:** 40/40 runs (seeds 135–144 × LMP/EB × fixed/td3; BloodMNIST, q = 0.5, 500 rounds), from 2026-10-05 17:27 to 2026-10-09 01:46. Interleaved queue, 6 processes on the GPU.
- **Execution:** no failures and **no run repeated** (pre-registration §6).
- **Pauses:** the 7h–18h Mon–Fri window was applied by the controller. The author's exceptions (manual resumes on 10/05–10/08) were logged. Pausing with SIGSTOP does not change a run's trajectory, only the wall clock.

**Analysis:** `scripts/passo2_oficial/analisar_b31.py --margem 3.0 --ataques LMP EB`, run once with the complete grid. It produced `analise.txt`, `resumo_runs.csv` and `mecanismo.csv`.

---

## 1. B3.1 — confirmatory result

**Primary metric:** median accuracy over rounds 401–500. D = fixed − td3, n = 20 pairs (attack × seed), margin M = ±3.00 p.p. (Addendum 1; **weak equivalence**, by the §3 rule).

| metric | Δ (p.p.) | CI90 | CI95 | d | TOST ±3.0 | Wilcoxon | verdict |
|---|---|---|---|---|---|---|---|
| **primary (median 401–500)** | **+0.13** | (−0.64; +0.90) | (−0.80; +1.06) | +0.07 | **p < 0.0001** | p = 0.70 | **EQUIVALENT** |
| AUC 1–500 | +0.12 | (−0.30; +0.54) | (−0.39; +0.63) | +0.11 | p < 0.0001 | p = 0.50 | EQUIVALENT |
| AUC 251–500 | +0.09 | (−0.49; +0.68) | (−0.61; +0.80) | +0.06 | p < 0.0001 | p = 0.29 | EQUIVALENT |

The primary and both AUCs agree, so the result is **not** "reset-sensitive".

**Per attack (descriptive, Holm):**

| attack | metric | Δ (p.p.) | CI95 | d | Wilcoxon (Holm) |
|---|---|---|---|---|---|
| LMP | primary | −0.47 | (−1.94; +1.00) | −0.23 | 0.70 (0.70) |
| EB | primary | +0.73 | (−0.57; +2.03) | +0.40 | 0.28 (0.55) |
| LMP | AUC 251–500 | −0.27 | (−1.74; +1.20) | −0.13 | 0.85 (0.85) |
| EB | AUC 251–500 | +0.45 | (+0.09; +0.82) | +0.88 | 0.027 (0.055) |

→ **PRE-REGISTERED VERDICT: EQUIVALENCE CONFIRMED** (M = ±3.0 p.p.).
- The primary's CI90 lies **well within ±1 p.p.** That is, equivalence would also hold with the MNIST margin, although that is not the pre-registered test.
- No per-attack difference is significant after Holm. The closest case (EB, AUC 251–500, Holm 0.055) points towards the fixed action and is descriptive.

## 2. B3.2 — mechanistic hypotheses (Holm over H3–H5, σ_a = 0.0475)

| hypothesis | statistic | result |
|---|---|---|
| **H3** corr. of EB × LMP actions > 0.9 | median r = 0.598 (per seed: 0.657 · 0.558 · 0.555 · 0.454 · 0.551 · 0.643 · 0.621 · 0.575 · 0.629 · 0.665) | Holm p = 1.00 → **NOT confirmed** |
| **H4** drift \|π₅₀₀ − π₀\| < σ_a | median 0.0122; max. 0.0189 (≈ 1/4 of the noise) | Holm p < 0.001 → **CONFIRMED** |
| **H5** S_swap \|π₅₀₀(s) − π₅₀₀(s′)\| < σ_a | median 0.0210; max. 0.0290 | Holm p < 0.001 → **CONFIRMED** |

**Descriptive:**

| attack | drift | sd_estados | S_swap | S_shuffle |
|---|---|---|---|---|
| EB | 0.0124 | 0.0676 | 0.0208 | 0.0284 |
| LMP | 0.0120 | 0.0095 | 0.0210 | 0.0110 |

sd_estados is the standard deviation of π₅₀₀'s output across the states of rounds 401–500. On MNIST (B2.2): drift 0.0097, S_swap 0.0055, S_shuffle 0.0061.

**Reading:**
- **H4 repeats:** in 500 rounds the policy moves away from its initialization by about 1/4 of its own exploration noise, of a similar order to MNIST.
- **H5 repeats under the pre-registered rule:** replacing the observation with the other attack's changes the action by much less than σ_a. In absolute terms, however, S_swap is ~4× MNIST's (0.021 vs. 0.0055).
- **Under EB, π₅₀₀ varies more across states (sd 0.068 > σ_a)** than under LMP (0.0095). The most likely explanation is in §3: with resets every ~23 rounds, the observed states change a lot over the run and the same almost-static network produces more spread-out outputs. This is exploratory. The variation still does not turn into a performance difference (§1).
- **H3 fails:** the actions executed under EB and LMP with the same seed correlate ~0.6, vs. ~0.99 on MNIST. B2.1 already noted that this correlation comes from the **shared noise sequence** and drops when the runs have different reset histories (s110, r = 0.899). On BloodMNIST there are ~20 resets per run, vs. 0.2–2.2 on MNIST (§3), and that is what desynchronizes the runs. This interpretation is exploratory: the confirmatory result is only that H3 does not repeat.

### Post-hoc reference: the initial actor π₀ on the same states (descriptive, not pre-registered)

Added after the analysis, like B2.2's "initial actor" reference on MNIST. The same three sensitivity measures were computed for the **initial actor π₀** of each td3 run, on the same states of rounds 401–500 used for π₅₀₀. Script: `scripts/passo2_oficial/b31_pi0_referencia.py`; outputs `pi0_referencia.txt` and `pi0_referencia.csv` (per run).

| attack | sd_estados π₀ | sd_estados π₅₀₀ | S_swap π₀ | S_swap π₅₀₀ | S_shuffle π₀ | S_shuffle π₅₀₀ |
|---|---|---|---|---|---|---|
| EB | 0.0637 | 0.0676 | 0.0186 | 0.0208 | 0.0287 | 0.0284 |
| LMP | 0.0090 | 0.0095 | 0.0186 | 0.0210 | 0.0106 | 0.0110 |

Medians per attack (n = 10 each). Per-run ratio π₅₀₀/π₀ (median, n = 20): sd_estados 1.03 (0.84–1.18), S_swap 1.04 (0.95–1.16), S_shuffle 1.04 (0.83–1.21).

**Reading:**
- **The final policy is as sensitive to its input as a freshly initialized network.** All three ratios are ≈ 1. In 500 rounds, training did not make the actor respond more (or less) to the state. This is the same conclusion as B2.2 on MNIST, now on BloodMNIST.
- **The larger spread under EB is already present in π₀** (sd_estados 0.064 vs. 0.009 under LMP). It comes from the states that the EB runs visit, not from anything learned. This supports the explanation given above (periodic resets make the observed states vary more).
- **The ~4× higher S_swap than on MNIST is also already present in π₀** (0.0186). It reflects BloodMNIST's states, not a policy that came to depend on them.
- This reference is descriptive and does not change the pre-registered verdicts for H3–H5.

## 3. Exploratory findings (not confirmatory): periodic-reset regime

- **All 40 runs are in a periodic-reset regime.**
  - Mean resets per run: ~20 (fixed 19.95; td3 21.05).
  - The median interval between resets is **23–24 rounds**, the same in the four cells.
  - **100% of the runs** have a reset inside the 401–500 window.
  - On MNIST (B2.1), the mean was 0.2–2.2 resets per run.
- **Accuracy level** (mean of the primary):

  | condition | EB | LMP |
  |---|---|---|
  | fixed | 32.7% | 30.9% |
  | td3 | 31.9% | 31.3% |

  This is above FedAvg under attack in B3.0 (17–21%), but far below FedAvg without attack (77.8%). **The §1 equivalence is between two systems operating in the same degraded regime:** neither the fixed action nor TD3 holds BloodMNIST under LMP or EB in the official environment.
- **Relation to Addendum 2:** the reward scale is ~3× smaller on BloodMNIST, and the reset threshold (−80) was kept. What is observed is not an absence of resets, but regular resets in both conditions. The exact mechanism (why the cycle is ~23 rounds) was not investigated.
- **td3 vs. fixed in resets:** ~1 more reset per run for td3 (descriptive paired Wilcoxon p = 0.044). The difference is small compared with the common regime and does not show up in accuracy.
- **Weight mass on attackers:** median ~0.016 in the four cells, vs. ≤ 0.0001 on MNIST. The aggregation skeleton lets a little weight through to the attackers on BloodMNIST, equally with or without TD3.

## 4. Consequences (Gate B′, pre-registration §7)

- **Gate B′: the rule is met.** Equivalence (H1) and the mechanism (H4, H5) repeat.
  - P2 can state, on two datasets (one of them a health dataset), that the published AdaAggRL's TD3 is equivalent to a fixed action and that the policy neither moves away from its initialization nor comes to depend on its input.
- **Mandatory caveats when citing B3.1:**
  1. The ±3.0 p.p. margin is a **weak equivalence** (Addendum 1), although the observed CI90 lies within ±1 p.p.
  2. The equivalence happens in a **degraded, periodic-reset regime** (~31% accuracy, one reset every ~23 rounds). It says that TD3 adds nothing to the fixed action, **not** that either of them defends BloodMNIST well in the official environment.
  3. **H3 does not repeat.** The near-perfect identity of actions across attacks seen on MNIST is specific to the low-reset regime and should not be presented as a general property.
- It **does not change** the P2 conclusions on MNIST (B2.1–B2.8). It extends the scope of the main finding to a medical dataset.

## 5. Files

- `analise.txt`: full output of the pre-registered analysis.
- `resumo_runs.csv`: per run, the primary, the AUCs, the resets, the reset in 401–500 and the mass on attackers.
- `mecanismo.csv`: per td3 run, drift, sd_estados, S_swap and S_shuffle.
- `pi0_referencia.txt`, `pi0_referencia.csv`: the post-hoc π₀ reference (§2), not pre-registered.
- `raw/`: the 40 JSONs, the `obs` and the actor checkpoints. Intermediate checkpoints are not versioned (pre-registration §8).
