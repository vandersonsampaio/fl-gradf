> English translation of `PREREGISTRO.md`. The Portuguese original is the frozen record (its hash is in `PREREGISTRO.sha256`); if the two ever disagree, the original prevails.

# Pre-registration — B2.6: decomposition of the skeleton (memory, weighting shape, signal)

**Status:** **FINAL**, approved by the author on 2026-09-30 and frozen before any run (hash in `PREREGISTRO.sha256`).
**Date:** 2026-09-30
**Scope:** confirmatory; addresses the gap "separate granularity and memory; no study has done it" from the systematic literature review.
**Record:** local, with a hash; committing is up to the author.

## 1. Question

P2 shows that TD3 does not contribute (B2.1–B2.3) and that the headroom comes from a **fixed skeleton**: a per-client signal, a threshold and memory. **Which pieces of the skeleton matter?**
- memory (persistent penalty);
- the weighting shape (soft vs. binary mask);
- the signal (S_R from gradient inversion vs. `cos_server`, which does not reconstruct data).

## 2. Regime and reference

- **Regime:** in-house framework, identical to the ablation and to C.0 — MNIST, 10 clients, Byzantine [0, 1], 15 rounds, root 100, 3 alphas × 7 attacks = 21 cells. **Valid cells: 19** (excludes `fltrust_aligned` α ≤ 0.1, an attack artifact; C.0).
- **Seeds: 52–61** (new, reserved for confirmation in the in-house framework).
- **Reference (base skeleton) = `sr_only`:** the score is S_R from gradient inversion; the soft weighting is min–max, with threshold δ = max(w̃)·b, b = 0.5; memory is λ^h with λ = 2. Same as the ablation. Chosen because the MMD cues added nothing (ablation, H2).

## 3. Variants

Each variant changes **one** piece relative to `sr_only`.

| id | changed piece | definition |
|---|---|---|
| `sr_only` | — (reference) | as above |
| `sr_nomem` | memory off | λ = 1 (no penalty) |
| `sr_memof` | the official code's memory | weight × 0.9^flag with the flag **before** the decrement (λ ≈ 1.11, as in the official code) |
| `sr_bin` | binary mask | clients above δ get uniform weight; same λ^h memory |
| `cosserver_only` | signal without inversion | score = cosine to the server update on the root; **confirmatory** replication of the ablation's exploratory finding |
| `sr_cosserver` | combined signals (**exploratory**) | score = mean of S_R and `cos_server`, with a fixed rule |
| `sr_b025`, `sr_b075` | threshold (**sensitivity**) | b = 0.25 and 0.75; reported as sensitivity, never used to choose a configuration |

## 4. Metric and unit

- **Metric:** global accuracy at round 15 (the same as the ablation and C.0: mean over the 10 clients' `X_test`). No resets in this framework.
- **Unit: seed.** For each seed, the mean over the 19 valid cells. D = variant − `sr_only`, n = 10.

## 5. Confirmatory hypotheses (family H1–H4, Holm, α = 0.05)

- **H1 (memory contributes):** `sr_only` − `sr_nomem` > 0. **One-sided** Wilcoxon.
- **H2 (memory strength matters):** `sr_only` − `sr_memof` ≠ 0. Two-sided Wilcoxon. If significant, our reproduction's λ = 2 is not neutral relative to the official code.
- **H3 ("continuous" does not matter):** `sr_bin` ≈ `sr_only`. **TOST ±1.0 p.p.** (CI90 reported along with CI95).
- **H4 (a signal without inversion is equivalent):** `cosserver_only` ≈ `sr_only`. **TOST ±1.0 p.p.**

**Per attack** (descriptive): Δ, CI95, d and Wilcoxon with Holm over the 7 attacks, for each hypothesis.

**Exploratory, no criterion:**
- `sr_cosserver` − `sr_only`, and `sr_cosserver` − best of (`sr_only`, `cosserver_only`), per attack (the complementarity hypothesis);
- sensitivity to b;
- per-attack profile of S_R vs. `cos_server`.

## 6. Expected reading for P2

- **H1 confirmed** → "memory matters" enters the design principles.
- **H3 confirmed** → the continuous shape of the weighting is not what matters. Together with B2.8, it closes the "discrete vs. continuous" axis.
- **H4 confirmed** → a signal **without data reconstruction** replaces S_R in this regime. This supports R3 of GRADF-v2, at the cost of requiring the root dataset (an open data-governance question for the clinical setting).

## 7. Cost

- **CPU only, in parallel with the GPU.** Variants with inversion ≈ 62 s per run; `cosserver_only` ≈ 10 s.
- **Grid:** 8 variants × 21 cells × 10 seeds = 1,680 runs ≈ 25 process-hours, **~3–4 h** with 8 processes (`nice 19`, 2 threads).

## 8. Rules

- Implementation in a **new** file (`scripts/b26_decomposicao.py`), reusing the ablation learner without changing it. Nothing in `src/` changes.
- The random extractor is always built, as in the ablation, to preserve the RNG and the pairing by seed across variants.
- Analysis pre-written and tested with synthetic fixtures before the freeze; run once, with the complete grid.
- No accuracy is looked at before the grid is complete.

## 9. Author's decisions (2026-09-30) and validation before the freeze

1. **Reference = `sr_only`.**
2. **TOST margin** in H3/H4: **±1.0 p.p.**
3. **Unit:** mean of the 19 valid cells per seed, **n = 10**.
4. **Exploratory variants included** (`sr_cosserver`, `sr_b025`, `sr_b075`).

Validation (already-used seeds, before the freeze):
- `weights_for` matches `compute_weights_and_penalty` from `src/` for `sr_only` and `sr_nomem` in 200 random cases;
- B2.6's `sr_only` **exactly reproduces** the ablation's `sr_only` (seed 42, α = 0.5, `sign_flipping`: 0.889400 = 0.889400);
- the analysis, tested with discarded synthetic fixtures and planted effects, reacted correctly (H1 detected at −3 p.p.; H3/H4 equivalent with Δ ≈ 0).

Execution: `scripts/run_grid_b26.sh` (80 jobs = 8 variants × 10 seeds, 8 processes, `nice 19`, 2 threads), started **after** B2.8 finished, to reduce CPU contention with B2.3b on the GPU.
