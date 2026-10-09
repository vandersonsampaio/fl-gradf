# Result — B2.6: decomposition of the skeleton (memory, weighting shape, signal)

**Date:** 2026-10-01
**Pre-registration:** `PREREGISTRO.md` (English translation in `PREREGISTRO.en.md`; hash in `PREREGISTRO.sha256`; `PREREGISTRO.md`, `scripts/b26_decomposicao.py` and `scripts/run_grid_b26.sh` checked against the hash before the analysis: OK).
**Grid:** 80/80 jobs = 1,680 runs (8 variants × seeds 52–61 × 21 cells), 09/30 19:15 → 10/01 03:46, CPU, 0 failures.
**Analysis:** `scripts/b26_decomposicao.py analisar` → `analise.txt`, `por_semente.csv`. Unit: seed (mean of the 19 valid cells), n = 10.

---

## 1. Confirmatory hypotheses (Holm over H1–H4, α = 0.05)

| hypothesis | Δ (p.p.) | CI95 | CI90 | test | Holm | result |
|---|---|---|---|---|---|---|
| **H1** memory contributes (`sr_only` − `sr_nomem` > 0) | −0.35 | (−0.82; +0.12) | — | one-sided Wilcoxon, p = 0.97 | 1.00 | **NOT confirmed** |
| **H2** memory strength matters (`sr_only` − `sr_memof` ≠ 0) | −0.63 | (−1.08; −0.18) | — | two-sided Wilcoxon, p = 0.010 | **0.039** | **CONFIRMED** |
| **H3** binary mask ≈ soft (`sr_bin` − `sr_only`, TOST ±1 p.p.) | +1.02 | (+0.47; +1.57) | (+0.58; +1.47) | TOST, p = 0.54 | 1.00 | **NOT confirmed** |
| **H4** `cos_server` ≈ S_R (`cosserver_only` − `sr_only`, TOST ±1 p.p.) | +0.42 | (−0.22; +1.06) | (−0.09; +0.94) | TOST, p = 0.035 | 0.105 | **NOT confirmed** (passes uncorrected, fails with Holm) |

## 2. Reading of each hypothesis

**H1, memory.** There is no evidence that the persistent penalty (λ = 2) helps **on average**. The point estimate favors switching it off (−0.35 p.p.). The effect is **heterogeneous across attacks** (descriptive, Holm over 7 attacks) and cancels out in the aggregate:
- it helps on `sign_flipping` (+3.4 p.p.);
- it hurts on `low_mag_backdoor` (−3.2), `fltrust_aligned` α = 0.5 (−1.2) and `label_flipping` (−1.9, not significant after Holm);
- it is neutral on the others.

**H2, memory strength.** **Confirmed:** the official code's weak memory (0.9^flag, λ ≈ 1.11) is **better** than our reproduction's λ = 2 (+0.63 p.p.). The per-attack profile mirrors H1's: strong memory hurts `low_mag_backdoor` and `fltrust_aligned` and helps `sign_flipping`. Consequence: λ = 2 **was not neutral**. Stronger memory does not explain headroom; it tends to hurt.

**H3, weighting shape.** Equivalence was **not** confirmed because the difference **favors the binary mask** (+1.02 p.p., the whole CI95 above 0, point estimate above the margin). *Outside the pre-registered test* (the criterion was equivalence): continuous (soft) weighting **is not an advantage**; a binary filter with the same threshold does **as well or better** in 5 of the 7 attacks (`label_flipping` +5.1; `low_mag_backdoor` +2.2; `fltrust_aligned` +1.1; `krum_collusion` +0.5) and worse on `sign_flipping` (−1.9) and `gaussian_noise` (−0.1).

**H4, signal.** Equivalence **not confirmed** after Holm. In the aggregate the difference is small (+0.42 p.p., CI90 within ±1 p.p.), but it **hides opposite per-attack profiles** (all with Holm < 0.05, except `sign_flipping`):
- **S_R wins** on the model attacks, especially at α = 0.05: `gaussian_noise` −4.6; `krum_collusion` −3.5; `trim_attack` −2.7. And it wins on `fltrust_aligned` α = 0.5 (−8.6), where the attacker aligns exactly with the server update, which is `cos_server`'s own signal;
- **`cos_server` wins** on the label and backdoor attacks: `label_flipping` **+11.5** (e.g. α = 0.1: 83.4 vs. 65.3) and `low_mag_backdoor` +3.8.

The signals are **complementary**, not equivalent. This confirms, now on new seeds, the exploratory profile seen in the ablation.

## 3. Exploratory (no criterion)

- **Combined signals** (`sr_cosserver` = 0.5 S_R + 0.5 `cos_server`): +1.21 p.p. over `sr_only` (CI95 −0.05 to +2.47) and +0.62 over the best single signal per seed (CI95 −0.51 to +1.74). The simple mean does **not** capture the complementarity, because it inherits each signal's losses (e.g. α = 0.05 `gaussian_noise`: 69.9 vs. 80.4 for S_R). A combination that selects or weights per attack is left for P3 (complementary-signals direction).
- **Threshold b:**
  - b = 0.25: **+1.57 p.p.** (CI95 +1.22 to +1.92);
  - b = 0.75: **−10.45 p.p.** (CI95 −11.4 to −9.5).

  **The threshold is the component with the largest effect** of all those tested, an order of magnitude above memory, shape and signal in the aggregate. This is consistent with B2.3, where the collapses came from a₅ ≈ 0, and justifies B2.4.

## 4. Consequences for P2 (revision of the design principles)

| design principle (before B2.6) | after B2.6 |
|---|---|
| "memory matters" | **Does not hold in the aggregate.** The effect depends on the attack and cancels out; strong memory (λ = 2) is worse than the official code's weak one (H2). Rephrase: *memory does not explain the headroom; its effect is attack-specific.* |
| "continuous vs. discrete" (P1) | **Continuous weighting is not an advantage:** the binary mask is as good or better. Together with B2.8, the P1 difference is one of **per-client granularity**, not of continuity. |
| "the per-client signal matters where it separates" | **Reinforced and refined:** S_R and `cos_server` separate **different** attacks. No single signal dominates. |
| "the threshold is the decisive component" | **Reinforced:** b has the largest measured effect (−10.5 to +1.6 p.p.). |

Implications for P3:
- **Complementary-signals direction:** strongly motivated, but the combination needs to be **selective**, not a simple mean.
- **R3:** `cos_server` dispenses with the inversion, but requires the root and is vulnerable to server-aligned attacks.
- **R6:** the threshold as the central interpretability parameter.

## 5. Limitations

- 15 rounds; MNIST; the in-house framework's regime (logistic, 10 clients).
- H3's test was an equivalence test; the "binary is better" direction is a **non-pre-registered** reading of the CI.
- The per-attack effects use Holm over 7 attacks, but are descriptive (the confirmatory family is H1–H4).
- `sr_cosserver` uses a naive combination rule (mean).
