> English translation of `ADENDO0_metrica_agregacao.md`. The Portuguese original is the frozen record (its hash is in `ADENDO1.sha256`); if the two ever disagree, the original prevails.

# Addendum 0 — B3.1: metric and aggregation in the official code (before B3.0 finished)

**Date:** 2026-10-05, before seeing any B3.0 result and before any B3.1 run. `PREREGISTRO.md` (commit `315b880`) does not change; this addendum only **declares** how the official code measures and aggregates, to close in Part I the same question opened in Part II (C0b, `VERIFICACOES.md` §3; B2.8s).

## Metric (official code, `external/AdaAggRL/exp_environments.py`)

- The recorded accuracy (`history['acc']`, used for the primary and the AUCs) is `test(self.net, self.testloader)`, evaluated **on the whole global test set**, in a single `DataLoader` (batch 64, `shuffle=False`, `drop_last=True`). It is not a mean of per-client accuracies.
- **MNIST:** official test set (10,000), approximately class-balanced.
- **BloodMNIST:** official test set (3,421; 3,392 evaluated because of `drop_last`), **class-imbalanced**: 243 to 666 examples per class. The metric is the accuracy on the test set's natural distribution, without balancing.
- The TD3 reward is the difference of the **summed loss** over the same `testloader`.

## Aggregation (official code)

- **AdaAggRL (`fixed` and `td3`):** aggregation weights the clients' models by `k`, computed from the action and from each client's scores (state), with min-max, the a₅ threshold, the memory penalty and normalization. **It does not use the clients' sample size.**
- **B3.0 FedAvg** (the runner's `fedavg` condition): `average`, i.e. a **uniform mean** of the clients' models, without sample-size weighting.
- **Client sizes:** the `_build_groups_by_q` partition builds groups by dominant class, of different sizes, and splits each group into equal parts. Clients have different sizes across groups, but none of the aggregations above uses that size.

## Consequence for reading B3.1

- The fixed × td3 comparison is between two weightings **without sample size**, evaluated with the same metric (accuracy on the global test set). The "sample-size × uniform weighting" confusion found in the in-house framework **does not apply** to it.
- The B3.0 ceiling (FedAvg without attack) also uses a **uniform mean**, consistent with the systems being compared. On BloodMNIST the test set is imbalanced, so the margin M inherits the accuracy scale on that natural distribution; this is declared.
