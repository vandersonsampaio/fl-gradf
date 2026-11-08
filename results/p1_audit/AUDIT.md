# P-1 audit of the P1 manuscript (MNIST)

**Date:** 2026-10-29.
**Manuscript audited:** FGCS submission FGCS-D-26-05588, LaTeX source `docs/P1/fgcs_src/elsarticle-template-num-names.tex` (line numbers below refer to it).
**Plan:** `docs/plano_p1_completo.md`, gate P-1 (Sec. 4.4).
**Code:** `scripts/p1/audit_mnist.py` (claim check), `scripts/p1/inference.py` (E1–E3 analyses), `scripts/p1/run_mnist_inference.py` (applies them to MNIST).
**Outputs:** `claims_check.csv`, `extra_numbers.json`, `table{1..5}_*.csv`, `fig3_mean_by_attack.csv`, `mnist_inference.json`, `mnist_e3_*.csv`.

## 1. Number check

**57/57 checked numbers reproduce from the raw per-seed CSVs.** This covers Tables 1, 2 and 4, the in-text counts in Secs. 6.1–6.3, d\*, the extractor comparison and the legacy capture counts. Table 3 and Fig. 3 were regenerated as CSV.

The defects are not wrong numbers. They are wrong labels and mixed definitions:

| # | Where | Finding | Fix |
|---|---|---|---|
| A1 | l. 236, 437, 480 vs Table 4 | **Not a duplication.** These are two different tests that happen to round to the same p and d. Extractor comparison (random − pretrained): Δ = 0.0186, p = 0.0068, d = 1.105. AdaAggRL − Random: Δ = 0.0302, p = 0.0065, d = 1.115. | Print three decimals for d and two significant digits for p. |
| A2 | l. 369 vs l. 437 | **Two capture criteria are mixed.** Sec. 6.2 uses R\_s(c) > 0 against B(c), which gives AdaAggRL 10, GRADF 7, FedStrategist 7 and pretrained 8. The "from nine to seven … none and two" in Sec. 6.3 uses the pre-revision criterion A\_s − O(c) > sd\_s + sd\_O, which gives 9, 0, 2 and 7. | Use one criterion. Under the current one, the pretrained count is 8/21, not 7. |
| A3 | l. 433 | **Wrong reference label.** The comparison is described against "the global best fixed rule". The numbers behind it (AdaAggRL +0.020, p = 0.052; GRADF −0.052; FedStrategist −0.054) are against the **cell-wise** rule O(c). Against B(c) = acc(trimmed mean, c), every system is *above* the reference: GRADF +0.010 (p = 0.36), FedStrategist +0.009 (p = 0.052), Random +0.052 (p < 10⁻⁶). | Rewrite. B(c) is a weaker reference than the Random-selector, not a "more demanding criterion". |
| A4 | l. 236 | **Circular justification.** The extractor choice cites "the aggregate comparison reported in Section 6". Section 6 only reports each variant against Random. | Cite the comparison in Sec. 4.4 itself, or move it to Sec. 6. |
| A5 | l. 369 | **The 18-cell counts keep d\* from the 21-cell grid.** Recomputing d\* on the 18 cells flips it to FLTrust and gives AdaAggRL 9/18 and discrete 9/18, instead of 10 and 7. | State the choice. Under the plan, E3 recomputes d\* on the confirmatory grid. |
| A6 | l. 378 | The "per-cell SD of about 2.9 p.p." matches Random and FedStrategist (2.97 and 2.98 p.p.). GRADF's per-cell SD is **11.2 p.p.** | Report it per system. |
| A7 | Fig. 3 | 4 of 28 bars differ from the data by 0.002 (FedStrategist krum collusion; Random gaussian noise, sign flipping, trim attack). The source still carries the comment "substitua pelos valores exatos". | Generate the bar values from `fig3_mean_by_attack.csv`. |
| A8 | Table 1, roots 3000 and 50 | **Seed counts are mixed.** FLTrust uses 10 seeds; median, TM and Krum use 3 (42–44). With 10 seeds, the root-3000 "Non-FLTrust (robust)" cell is **1/21, not 2/21**; everything else is unchanged. | Use 10 seeds for all rules. |

## 2. Protocol findings (affect the text and the CIFAR-10 code)

**B1. The partition does not change with the seed.**
- `load_dataset_participants` reads one fixed Dirichlet partition per α from `data/processed/mnist/<split>/`.
- The seed only changes three things: the root subsample (`root_seed = seed`), the attack simulator's RNG, and the selectors' RNGs.
- Sec. 5.1 (l. 253: "generated once per seed") and Sec. 5.4 ("the same seed determines … the data partition") are wrong. All inference is conditional on one partition per α.

**B2. NumPy's global RNG is never seeded.**
- `_LogisticModel.fit` shuffles mini-batches with `np.random.permutation`, so runs are not reproducible: the same configuration repeated gives differences of 1–4×10⁻⁴.
- As a result, median, TM and Krum vary almost nothing across seeds (per-cell SD ≈ 0.05 p.p.).
- FLTrust's across-seed spread at root 100 (2.7–5.4 p.p.) comes from the root subsample.
- **Consequence for E2:** winner switching between seeds is driven by FLTrust's root sample, not by data variation.

**B3. The trimmed-mean root invariant was assumed, never tested.**
- Median, TM and Krum were run once, with the full 3,000-sample server set, and reused in the root-100 and root-50 grids. The zero differences in `mnist_inference.json` are therefore vacuous.
- From the code, the invariant holds for the blind attacks. For informed attacks it holds only in distribution: `InformedAttackedFederatedLearner` fits the server model before the clients, so the root size shifts the unseeded RNG stream.
- **It fails for `fltrust_aligned`.** The attacker submits a copy of the server's root update, so TM, median and Krum at root 100 differ from the reused root-3000 values in those 3 cells.
- Two quick runs (seed 42, α = 0.5):

  | Attack | Root 3000 | Root 100 | Note |
  |---|---|---|---|
  | trim attack | 0.8684 | 0.8699 | single thread |
  | sign flipping | 0.8728 / 0.8731 | 0.8722 | two repeats at root 3000 |

  The differences are of the same size as the run-to-run noise.
- **Under plan Sec. 4.4 this is the documented-nondeterminism case: the cause is identified.** Before E1, the CIFAR-10 runner must seed the global RNG per run. Exact invariance then becomes testable.

**B5. Seeded rerun (`results/p1_invariance/`, `--seed_rng`, 2,520 runs, seeds 42-51, roots 3000/100/50).**
- Exact invariant: for median, TM and Krum, root 100 and root 50 equal root 3000 in 540/540 (seed, cell) pairs outside `fltrust_aligned`. Under `fltrust_aligned` all 90 pairs differ (mean -0.122 at root 100, -0.172 at root 50).
- Drift of the published (unseeded) cell means: small outside `fltrust_aligned` (FLTrust at most 0.009). The large drifts (up to 0.22 at root 100, 0.38 at root 50) are all the non-FLTrust rules under `fltrust_aligned`, i.e. the copied root-3000 values. The published root-100/50 numbers for those 9 (rule, cell) pairs are wrong.
- Seeded Table 1 (wins, 21 cells): root 3000 FLTrust 18; root 100 FLTrust 9, TM 9; root 50 FLTrust 6, TM 12. Table 2 at root 100: alpha 0.5 0/7, 0.1 4/7, 0.05 5/7.
- Seeded E1 (V = FLTrust - TM, root 3000 vs 100): +0.139 [0.129, 0.148] on 21 cells, +0.159 [0.149, 0.169] on 18; Wilcoxon p = 0.002, superiority in both.
- Seeded E2 (root 100): H_LOSO 0.046 [0.042, 0.050] on 21 cells, 0.049 [0.045, 0.053] on 18; d* = FLTrust in all folds. The published-data 21-cell value (0.068) was inflated by the wrong `fltrust_aligned` values.

**B6. Code check of the root invariant (plan Sec. 4, part 1), on the code that produced B5.**
Path traced: `exp9_dominance_grid.run_dominance_grid` → `AttackedFederatedLearner` / `InformedAttackedFederatedLearner._run_round` → `_STRATEGIES[rule].aggregate`.
- **No detector or hardening on this path.** `exp9_dominance_grid.py` and `attacked_learner.py` import neither `src.detection` nor `HardeningPipeline`. The accuracy-based detector and the Layer-4 validation only exist in `GRADFFederatedLearner`.
- **The rules never receive the root.** For any rule other than `fltrust`, both `_run_round` implementations pass only `sample_sizes` to `aggregate`. `MedianStrategy`, `TrimmedMeanStrategy` and `KrumStrategy` take `**kwargs` and use none of them (`src/defense/aggregation_methods.py`).
- **Partitions and client data do not depend on the root.** `load_dataset_participants` loads the client shards first, then subsamples `server_val` with its own `default_rng(root_seed)`.
- **Global accuracy does not use the root.** It is the mean over the clients' own test sets.
- **The only channel is the attack input.** `InformedAttackedFederatedLearner` fits the server model on the root every round, for every rule, and passes it as `reference` to `poison_update`. Of the four informed attacks, only `_fltrust_aligned` reads `reference` (it returns a copy of it). `trim_attack`, `krum_collusion` and `low_mag_backdoor` use only `peer_updates`/`update`. Blind attacks never see the root.
- **RNG coupling, removed with `--seed_rng`.** The server fit draws a permutation whose length is the root size. Without a separate stream, this fit consumed the global `np.random` stream and shifted every later client fit, so informed-attack runs could differ across roots by noise alone (B2/B3). `--seed_rng` gives the server fit its own RNG (`server_rng_seed`), so the client fits see the same stream at every root size.
- **Conclusion.** Outside `fltrust_aligned`, median, TM and Krum are exactly root-invariant by construction. This matches the 540/540 equal pairs in B5. The same check must be repeated on the CIFAR-10 runner before E1 (plan Sec. 4); the GRADF and adaptive-system paths do use the root and are not covered by this invariant.

**B7. Canonical source for the fixed rules: the seeded rerun.** From here on, the MNIST fixed-rule numbers come from `results/p1_invariance/` (B5); the published files are kept only to reproduce the manuscript (Sec. 1). `scripts/p1/run_mnist_inference.py --source seeded` (default) writes `mnist_inference_seeded.json` and `mnist_e3_*_seeded.csv`; `--source published` reproduces the earlier outputs bit for bit. The adaptive systems still come from the published exp10 runs (unseeded; same partitions and same root subsample per seed, so the pairing holds).
- **d\* on 21 cells changes.** Root-100 global means: published TM 0.7099 vs FLTrust 0.7095 (the manuscript's d\* = TM, line 329); seeded FLTrust 0.7098 vs TM 0.6920. The manuscript's d\* was an artifact of the copied TM rows under `fltrust_aligned`. On 18 cells d\* = FLTrust under both sources (0.7023 vs 0.6828 seeded).
- **E2, 21 cells:** H_LOSO 0.068 → 0.046 [0.042, 0.050]; d\* agreement across folds 0.7 → 1.0. 18 cells: 0.049 under both.
- **E3, 18 cells (confirmatory):** unchanged. All categories and Holm p-values equal; means move by at most 0.0004. Capture counts: GRADF 8 → 7.
- **E3, 21 cells (sensitivity):** FedStrategist vs B\_{-s} goes from superiority (+0.014, Holm p = 0.027) to no evidence (+0.009, p = 0.26). GRADF vs B\_{-s}: +0.016 → +0.010, no evidence in both. Cells with cross-fitted headroom: 21 → 12 (d\* is now FLTrust, which already wins 9 of them).
- **Manuscript consequence:** every statement that uses TM as the global best fixed rule on the 21-cell grid (lines 329, 433 and the headroom/capture counts at 331 and 369) rests on the copied rows and must be recomputed from the seeded source.

**B8. Tables and figures regenerated from the raw data (`scripts/p1/make_tables_figures.py`).** Replaces `scripts/gen_paper_tables_figures.py`, which read summary/verdict CSVs, hard-coded the main checkout and wrote PNGs although the manuscript's figures are TikZ with typed-in data. The new script goes raw per-seed CSVs → `audit_mnist.py` functions → `results/p1_tables/<source>/`: `tables.tex` (Tables 1–5), `figure_data.tex` (the data blocks of `Fig1.tex`, `Fig2.tex`, `Fig3.tex`, in their own syntax) and three PNG previews.
- **`--source published` reproduces the submission:** every number of Tables 1–5 (258 numbers) and the Table 3 rule column match the manuscript; Fig. 2 matches (13/10/7/10); Fig. 1 left panel matches.
- **A9 (new). Fig. 1, right panel (root 100), contradicts Table 3 in two cells.** At α = 0.5 the figure shows trimmed mean for krum collusion and Krum for label flipping; the 10-seed raw data, and Table 3 itself, give the reverse (Krum, margin 0.002; TM, margin 0.009). The figure was drawn from `exp9_dominance_grid_10seeds_root100_verdict.csv`, an older verdict file. Win counts are unaffected (a swap).
- **Fig. 3** differs from the raw means by at most 0.002 (A7).
- **Old Fig. 2 PNG** used the Table 3 bold criterion (beats O(c)), giving 11/14/4/13; the submitted TikZ figure uses the headroom criterion (beats B(c) where O(c) > B(c)). The new script uses the submitted criterion.
- **`--source seeded` (canonical):** Tables 1–2 as in B5; Table 3 changes only in the best-fixed column under `fltrust_aligned`; Tables 4–5 unchanged (systems vs Random, no fixed rule involved); Fig. 2 becomes 12/12/11/11 (d\* = FLTrust, B7).

**B4. Some inputs were not versioned.** `results/tables/*` is git-ignored. Of the seven raw inputs read by `audit_mnist.py`, five were already tracked; the other two (`exp9_nonfltrust_extraseeds_raw.csv`, `exp10_passo_zero_combined_raw.csv`) are now force-added. The 3-seed originals are not read by the pipeline. The seeded rerun (`results/p1_invariance/raw/`) is versioned except its run logs (`*.log` is git-ignored).

## 3. E1–E3 analyses on MNIST (plan Secs. 4.3–4.5)

Unit = seed (n = 10). Tests: Wilcoxon (primary), t (complementary), Holm per family, TOST at 1.0 p.p. Fixed rules from the published files; the seeded-source values are in B7 (E1, E2 and Tables 1–2 in B5).

**E1. D\_s = mean\_c [V(c, 3000) − V(c, 100)], where V = FLTrust − TM:**

| Grid | Mean | 95% CI | p (Wilcoxon) | d\_z | Category |
|---|---|---|---|---|---|
| 21 cells | +0.156 | [0.146, 0.166] | 0.002 | 11.0 | superiority |
| 18 cells (no FLTrust aligned) | +0.159 | [0.149, 0.169] | 0.002 | 11.4 | superiority |

TM was not rerun at root 100 (B3), so on MNIST D\_s equals the FLTrust difference.

**E2. Headroom over D\_R:**

| Grid | In-sample | Ĥ\_LOSO, 95% bootstrap CI | d\* in-sample | d\* agreement across LOSO folds | Cell-winner agreement |
|---|---|---|---|---|---|
| 21 cells | 0.062 | **0.068** [0.057, 0.069] | trimmed mean | 70% | 98% |
| 18 cells | 0.050 | **0.049** [0.045, 0.053] | FLTrust | 100% | 98% |

- The cross-fitted headroom is clearly positive on MNIST and does not shrink toward zero, because the relational rules are nearly deterministic (B2).
- **Methodological note for the plan:** the in-sample headroom is *not* guaranteed to be an upper bound.
  - O(c) is biased upward by selection, but so is B(c) = acc(d\*, c).
  - With 21 cells, d\* flips between TM and FLTrust across folds, and the cross-fitted estimate comes out *above* the in-sample one.
  - The label "optimistic upper bound" (plan Sec. 4.5 / E2) should become "in-sample estimate".

**E3. Confirmatory grid = 18 cells; references are cross-fitted:**

| System | vs Random | vs B\_{−s} |
|---|---|---|
| AdaAggRL | +0.042 [0.025, 0.059], superiority | +0.095 [0.077, 0.114], superiority |
| Attack-aware | +0.012 [0.004, 0.021], superiority | +0.066, superiority |
| GRADF | −0.046 [−0.073, −0.020], inferiority | +0.007 [−0.022, 0.037], no evidence |
| FedStrategist | −0.052 [−0.065, −0.039], inferiority | +0.002 [−0.014, 0.018], no evidence |
| Random | — | +0.054 [0.039, 0.069], superiority |

- All superiority and inferiority results survive Holm.
- No comparison reaches equivalence at δ = 1.0 p.p.
- Sensitivity: AdaAggRL with the pretrained extractor vs Random gives +0.020 [−0.000, 0.041], no evidence. FedStrategist with GRADF's detector vs Random gives −0.047, inferiority.

**What changes in the MNIST story:**
1. The asymmetry between the families holds against Random.
2. Against the global best fixed rule, discrete selection is **not** below the reference. It is indistinguishable from it, and even the Random-selector beats it by about 5 p.p.
3. Only AdaAggRL is clearly above both references.

## 4. P-1 status

| Item (plan Sec. 4.4) | Status |
|---|---|
| MNIST numbers reconciled | **done**: 57/57 reproduce; text defects A1–A8 listed for the manuscript, A9 (Fig. 1) in B8 |
| Tables regenerable | **done** (B8): Tables 1–5 and the data of Figs. 1–3 regenerate from the raw per-seed CSVs, all versioned (B4) |
| Same source for abstract, results and conclusion | pending the text revision (Secs. 3, 4.1 of the plan) |
| E1–E3 analyses implemented and run on MNIST | **done** (Sec. 3 above, published source); rerun on the canonical seeded source (B7) |
| TM root invariant | **done**: code check (B6) and exact check (B5) — holds exactly for median, TM and Krum outside `fltrust_aligned` (0/540 nonzero pairs at roots 100 and 50); fails under `fltrust_aligned`, as expected |
