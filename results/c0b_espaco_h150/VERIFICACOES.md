# Post-grid checks of C0b (2026-10-04)

Requested by the author after the RESULTADO. Cheap analyses on the existing code and data.

**Terminology:** the uniform oracle-8 is a **reference ceiling** (exclude the attackers and weight the honest clients equally), **not an upper bound**: nothing guarantees that another weighting of the honest clients will not beat it (e.g. class balancing under label skew).

## 1. Does the oracle code with an all-ones mask reproduce FedAvg-10? **YES, exactly**

- **Test:** a literal copy of `_Oracle8` from `scripts/b27_horizonte.py::run_ceiling`, but aggregating the 10 clients (nobody excluded). α 0.1, seeds 72–74, same reseeding.
- **Result:** equal to C0b's `teto_fedavg10` at H = 15, 50 and 150 on the 3 seeds (|Δ| = 0).
- **Conclusion:** **there is no bug in the oracle code.** C0, B2.7 and C0b do not need to be recomputed for that reason.

## 2. Weight mass on attackers of the best system in each α 0.1 cell (from `scores_*.csv.gz`)

Per-client weights were only recorded for the 4 skeleton variants. In the cells where the best is a static rule (Clustering on `label_flipping` and `trim_attack`), the measurement is of the best variant. Full output in `massa_atacantes_a0.1.csv`.

| attack | best existing | system measured | mass on attackers (rounds 1–150) | rounds 101–150 | rounds with both attackers excluded | honest clients excluded per round (of 8) |
|---|---|---|---|---|---|---|
| `gaussian_noise` | sr_bin | sr_bin | 0.0000 | 0.0000 | 99.9% | 0.01 |
| `krum_collusion` | sr_b025 | sr_b025 | 0.0000 | 0.0000 | 94.2% | 0.20 |
| `label_flipping` | Clustering | cosserver_only | 0.0033 | 0.0021 | 58.9% | 3.38 |
| `low_mag_backdoor` | sr_b025 | sr_b025 | **0.22** | **0.26** | 4.1% | 1.65 |
| `sign_flipping` | sr_b025 | sr_b025 | 0.038 | 0.054 | 24.5% | 0.85 |
| `trim_attack` | Clustering | sr_b025 | 0.0001 | 0.0000 | 63.6% | 0.22 |

- In 4 of the 6 cells, the mass on attackers is ≈ 0. Even so, the best method is 3.9 to 5.6 p.p. **above** the FedAvg-8 oracle.
- The exception is `low_mag_backdoor`: the mass (0.22–0.26) is ≈ that of a uniform weight (0.2), i.e. the attackers **get in**. It is the only cell in which "exploiting the attackers" can contribute: their update is small, and they train on almost real data.

## 3. Why methods under attack beat the "oracle" (diagnosis done to explain item 2)

- On `gaussian_noise` α 0.1, `sr_bin` excludes exactly the 2 attackers every round and almost never an honest client. With a binary mask, the 8 honest clients get **uniform** weight.
- The FedAvg-8 oracle also aggregates only the 8 honest clients, but the framework's `FedAvgStrategy` weights **by sample size**. At α 0.1, the sizes are very unequal: on seed 74, from 110 to 18,561 examples per client.
- **Test:** FedAvg-8 oracle with uniform weighting of the 8 honest clients, α 0.1, seeds 72–74, H = 150:

| seed | sample-size-weighted oracle-8 (C0b's) | **uniform oracle-8** | sr_bin under `gaussian_noise` | FedAvg-10 without attack |
|---|---|---|---|---|
| 72 | 84.97% | **89.49%** | 89.47% | 89.35% |
| 73 | 84.97% | **89.46%** | 89.50% | 89.34% |
| 74 | 84.99% | **89.42%** | 89.45% | 89.41% |

- **The "negative gaps" at α ≤ 0.1 are an artifact of the oracle's weighting, not of the oracle itself nor of exploiting the attackers.**
  - *(Mechanism corrected on 2026-10-05.)* The "mean of the accuracies on the clients' test sets" is, in practice, the accuracy on the **global MNIST test set, IID and class-balanced** (split into equal parts, 1000 examples per client; `data/download_datasets.py`; checked in B2.8s). With per-client label skew (Dirichlet α ≤ 0.1), sample-size weighting gives too much weight to the classes of the large clients, and the aggregated model becomes unbalanced on a balanced test set. Uniform weighting across clients dilutes that bias.
  - With uniform weighting, the oracle-8 reproduces `sr_bin`, which excludes the attackers and weights the honest clients equally, and lands ≈ at FedAvg-10.
- **Consequences** (to be decided; nothing was recomputed):
  - The (sample-size-weighted) FedAvg-8 oracle, used as the ceiling in C0, B2.7 and C0b, **underestimates the reference ceiling at α ≤ 0.1** by ~4.5 p.p. (α 0.1, H = 150).
  - The C0b "headroom" criterion against a **uniform** oracle-8 may change the verdict in cells such as `label_flipping` α 0.05 (best existing 85.1%). This calls for a sensitivity analysis: 30 ceiling jobs with the uniform oracle, minutes of CPU, declared as post hoc.
  - The reading of A0(a) and C0b ("weighting the honest clients better than the uniform mean") must be corrected to: **weighting the honest clients uniformly, rather than by sample size**, is what leads to the ceiling, because, under label skew, sample-size weighting unbalances the classes on the balanced global test set.
