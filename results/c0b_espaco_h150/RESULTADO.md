# Result — C0b: remaining headroom at H = 150 against the best existing method

**Date:** 2026-10-04.
- Plan: `PLANO.md` (English translation in `PLANO.en.md`), commit `d676ec3`.
- Addendum 1 (code, per-client logging, checks; `ADENDO1.en.md`): commit `183a585`.
- Grid: 190 cell jobs (19 cells × seeds 72–81 × 8 systems, nested horizons 15/50/150) + 30 ceilings, from 10/03 09:25 to 10/04 16:58. No failures.
- Outputs: `analise.txt`, `espaco_restante.csv`, `grade_raw.csv`, `tetos_raw.csv`, `concordancia_sinais.csv`, `raw/` (includes `scores_*.csv.gz`, with S_R, cos_server, mask and weight per client and round).

## (i) Remaining-headroom map: **0/19 cells with headroom against the best existing method, at every horizon**

C0 criterion: gap to the FedAvg-8 oracle > 2 p.p. with CI95 > 0.

| horizon | against the best existing (8 systems) | against the reference skeleton (`sr_only`) |
|---|---|---|
| H = 15 | **0/19** | 6/19 |
| H = 50 | **0/19** | 5/19 |
| H = 150 | **0/19** | 5/19 |

- **No cell has headroom against the best existing method.** The only one that comes close is `label_flipping` α 0.05: the best is FLTrust (85.1%), +1.3 p.p. from the oracle (CI95 +0.7 to +1.9), below the 2 p.p. threshold.
- **Against the reference skeleton, the headroom is large** in 5 cells, all label or backdoor attacks:
  - `label_flipping` α 0.05: +17.2 p.p.;
  - `label_flipping` α 0.1: +5.9;
  - `label_flipping` α 0.5: +20.5;
  - `low_mag_backdoor` α 0.05: +3.3;
  - `sign_flipping` α 0.05: +2.0.
- **The (pre-registered) FedAvg-8 oracle underestimates the reference ceiling at α ≤ 0.1** (corrected on 2026-10-04; see the sensitivity section and `VERIFICACOES.md`).
  - In 13 of the 19 cells (all 6 at α 0.1), the best existing method is above it; at α 0.1 the difference is 3.9 to 5.6 p.p.
  - **Cause:** the framework's `FedAvgStrategy` weights **by sample size** (from ~10² to ~10⁴ examples per client at α 0.1), while the metric is the accuracy on the **global MNIST test set, IID and class-balanced** (split into equal parts, 1000 examples per client; `data/download_datasets.py`; checked in B2.8s) *(mechanism corrected on 2026-10-05)*. With per-client label skew, sample-size weighting unbalances the classes of the aggregated model. The oracle with uniform weighting of the 8 honest clients reproduces the best method (e.g. `sr_bin` under `gaussian_noise` α 0.1: 89.5% vs. 89.5%).
  - **Correction of the earlier reading:** what leads to the ceiling is excluding the attackers and **weighting the honest clients uniformly**, not "weighting the honest clients better than the uniform mean", as was written. It is not exploitation of the attackers (mass ≈ 0 in 4 of the 6 α 0.1 cells; the exception is `low_mag_backdoor`, `massa_atacantes_a0.1.csv`).
  - There is no bug in the oracle code: with an all-ones mask, it exactly reproduces FedAvg-10 (`VERIFICACOES.md` §1).
- **Gate C0 at H = 150:**
  - by the pre-registered criterion (sample-weighted oracle), 0/19;
  - by the sensitivity analysis with the uniform oracle, **1/19: `label_flipping` α 0.05**, with headroom of +2.8 p.p. (CI95 +2.2 to +3.4) against FLTrust.
  - Apart from that cell, no headroom remains against the best existing method. P3's differentiator lies in R3/R4/R5 (privacy, cost, adaptive robustness), with a narrow R1 target on `label_flipping` α 0.05.

## (iii) Best skeleton variant vs. best static rule (H = 150)

- **The variant wins in 12 of the 19 cells, with CI95 > 0,** by +0.14 to +2.7 p.p. These are mostly the model attacks (`gaussian_noise`, `krum_collusion`, `trim_attack` and `low_mag_backdoor` α 0.1). The best variant is usually `sr_b025` or `sr_bin`.
- **The rule wins in 1:** `trim_attack` α 0.1, Clustering, by −0.12 p.p.
- **Tie (CI95 contains 0) in 6:** `label_flipping` α 0.05 and 0.1, `sign_flipping` α 0.05 and 0.5, `low_mag_backdoor` α 0.05 and `fltrust_aligned` α 0.5. On the label attacks, the best rule (FLTrust or Clustering) is ahead on average, without significance.
- **Reading:** per-client granularity, with the tuned variant (threshold b = 0.25 or binary mask), is the best existing method in most cells, but by small margins (≤ 2.7 p.p.). FLTrust is practically equal to the best signal on the overall mean: 87.35% vs. 87.55% for `cosserver_only`.

## (ii) Signal-selection ceiling at H = 150

Mean over the 19 cells, per seed (n = 10), in %:

| estimate | accuracy | CI95 |
|---|---|---|
| per-seed max. (biased) | 89.18 | 88.83–89.52 |
| per-cell signal choice, in-sample | 88.92 | 88.48–89.36 |
| **per-cell choice, LOSO (reference)** | **88.79** | 88.37–89.21 |
| `cosserver_only` | 87.55 | 87.02–88.09 |
| FLTrust | 87.35 | 87.16–87.55 |
| `sr_b025` | 87.10 | 87.01–87.19 |
| `sr_only` | 86.04 | 85.58–86.49 |
| `sr_bin` | 85.39 | 84.83–85.96 |
| Clustering / Trimmed-Mean / Median | 82.46 / 75.58 / 74.41 | — |

- **Gain of the per-cell choice (LOSO) over the best single signal per seed: +1.24 p.p. (CI95 +0.98 to +1.49).** At 15 rounds (A0, B2.6) it was +1.75 p.p. **P3 direction 1's headroom shrinks with the horizon and is modest.**
- cos_server is chosen in 10/19 cells (`label_flipping` and `sign_flipping` at every α, `low_mag_backdoor` at α ≤ 0.1, `krum_collusion` and `trim_attack` at α 0.05), and S_R in 9 (noise and model attacks, especially at α 0.1 and 0.5).

## Per-client agreement between signals (`sr_only` log, 10 seeds, rounds ≥ 2)

Confirms A0(c), which had 1 seed:
- **The signals are nearly orthogonal:** the mean per-round Spearman between S_R and cos_server ranges from −0.13 to +0.40.
- **The AUCs are complementary:**
  - **S_R** separates `gaussian_noise` (0.99–1.0), `krum_collusion` (0.86–0.94) and `trim_attack` (0.80–0.90);
  - **cos_server** separates `label_flipping` (0.81–0.98), `sign_flipping` (0.86–0.96), `trim_attack` (0.87–0.95) and `krum_collusion` (0.82–0.89);
  - **neither separates `low_mag_backdoor`** (AUC ≤ 0.60);
  - on `fltrust_aligned`, cos_server is **inverted** (AUC 0.001), as expected for an attack designed against the server signal (R5).
- **Reading:** a non-learned selector based on each signal's separation has empirical support. The item (ii) ceiling (~+1.2 p.p.) is that of **choosing one signal per cell**. It **does not bound a per-client combination**, which uses both signals with different clients in the same round; that combination was not measured here.

## Implications

**Terminology:** the uniform oracle-8 is a **reference ceiling** (exclude the attackers and weight the honest clients equally), **not an upper bound**: nothing guarantees that another weighting of the honest clients will not beat it (e.g. class balancing under label skew).

1. **P2:** with a sufficient horizon, the best existing method closes the remaining headroom in almost every cell, including `label_flipping` α 0.1. The exception is `label_flipping` α 0.05: +2.8 p.p. against the uniform oracle (sensitivity). The headroom stays large against AdaAggRL and the reference skeleton on the label and backdoor attacks (9/19 cells against the uniform oracle).
2. **P3 (R1, Gate C0b):** the accuracy target against the best global method is **narrow**: only `label_flipping` α 0.05 (~+2.8 p.p.). The main differentiator is privacy (both strong signals have a cost), computational cost and adaptive robustness (`fltrust_aligned` inverts cos_server; `low_mag_backdoor` escapes both signals).
3. **The C0/B2.7/C0b ceiling at α ≤ 0.1:** the sample-weighted FedAvg-8 oracle should not be used as the ceiling. Use the uniform oracle in the next experiments. (A "sample-weighted" metric is not an alternative: the clients' test sets have the same size and are IID, so it coincides with the current one; B2.8s.)
4. **Note for P2 (weighting × robustness):** under per-client label skew (α ≤ 0.1) and a balanced global test set, **standard FedAvg's sample-size weighting alone, without any attack, costs up to ~4.5 p.p.** relative to uniform weighting (oracle-8, α 0.1, H = 150: 85.0% × 89.5%; FedAvg-10: 1.0–1.8 p.p.). Tables that compare defenses with standard FedAvg (and B2.8's and B2.7's FedAvg/FedProx arms) must discount this effect, so as not to credit the defense's robustness with what is only weighting.
5. **Direction 1 (signal selection):** the ceiling of choosing **one signal per cell** is small (+1.2 p.p.) and decreases with the horizon. A **per-client** combination is not bounded by that ceiling and remains open for P3. For the per-cell choice, a non-learned selector should suffice (R7).

## Post hoc sensitivity: ceilings with uniform weighting (2026-10-04)

Declared as post hoc in `VERIFICACOES.md` §3. Script `scripts/c0b_sensibilidade_oraculo_uniforme.py`, outputs in `sensibilidade_oraculo_uniforme/`. It is the same code as `run_ceiling`, with FedAvg's weighting made uniform, on seeds 72–81.

| α | sample-weighted oracle-8 (pre-reg.) | **uniform oracle-8** | sample-weighted FedAvg-10 | uniform FedAvg-10 |
|---|---|---|---|---|
| 0.05 | 86.46% | **87.92%** | 89.47% | 90.44% |
| 0.1 | 84.99% | **89.46%** | 89.39% | 91.23% |
| 0.5 | 91.67% | **91.74%** | 91.93% | 92.07% |

(H = 150.)

| C0 criterion (gap > 2 p.p., CI95 > 0) | H = 15 | H = 50 | H = 150 |
|---|---|---|---|
| headroom against the **best existing** (uniform oracle) | 0/19 | 1/19 | **1/19** (`label_flipping` α 0.05: +2.79 p.p., CI95 +2.19 to +3.39) |
| headroom against the **reference skeleton** (uniform oracle) | 7/19 | 9/19 | 9/19 |

- **At α 0.1, the best existing method reaches the uniform oracle** in every cell (gaps from −1.1 to +0.6 p.p.): excluding the attackers and weighting the honest clients equally is exactly what the best methods do.
- **At α 0.05, the only open cell is `label_flipping`**, in which the best existing method (FLTrust, 85.1%) is 2.8 p.p. below the uniform oracle (87.9%).
- The uniform FedAvg-10 is 0.7–5.3 p.p. above the best existing method at α ≤ 0.1. That is the gain from also using the data of the 2 clients who are attackers, which no defense can recover without exploiting the attackers. That is why it is not the relevant ceiling for the criterion.

## Caveats

- Descriptive; the best existing method and the best signal per cell are chosen in hindsight (optimistic for the existing methods, i.e. conservative for "headroom").
- Logistic MNIST, 10 clients, 2 Byzantine, root 100; the static rules are applied as in B2.7.
- The pre-registered ceiling (sample-weighted oracle) underestimates the reference ceiling (uniform oracle) at α ≤ 0.1; the uniform oracle is not an upper bound either (see Terminology). The sensitivity analysis with the uniform oracle is post hoc, motivated by a check done after the results.
- Before the grid, accuracies of one cell for seed 72 at H = 15 were exposed (declared in ADENDO1).
