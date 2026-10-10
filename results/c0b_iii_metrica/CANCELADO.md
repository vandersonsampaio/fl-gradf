# C0b-iii — CANCELLED (2026-10-05)

**Author's decision**, on 10/05, after the B2.8s result. No grid run was executed; there was only the reproduction smoke test of 10/04, outside the repository, which exactly reproduced C0b and the sensitivity analysis.

## Reason

The plan (`PLANO.md`, commit `7c0311b`; English translation in `PLANO.en.md`) compared two pairs of metric and ceiling: **U** (uniform metric over the clients' test sets + uniform oracle) and **P** (metric weighted by test-set size + sample-size-weighted oracle).

B2.8s showed that **the two metrics are identical by construction**. The clients' test sets are equal parts (1000 examples), split at random, of the global MNIST test set, which is IID and balanced (`data/download_datasets.py`). Therefore:
- the metric part of the sensitivity analysis would be vacuous (pairs U and P with the same metric);
- the ceiling part (uniform × sample-weighted oracle) was already measured in the C0b sensitivity analysis (`results/c0b_espaco_h150/sensibilidade_oraculo_uniforme/`).

Running ~18 CPU hours would add no information.

## What remains open (not planned)

The underlying concern ("part of the per-client filter's advantage comes from weighting the honest clients uniformly") can only be tested with a metric **on local test sets with each client's label distribution**, which the current pipeline does not have. If it is taken up again, it is a new experiment, with its own plan.

The code (`scripts/c0b_iii_metrica.py`, `scripts/run_grid_c0b_iii.sh`) remains versioned, unused.
