> English translation of `ADENDO1.md`. The Portuguese original is the frozen record (its hash is in `ADENDO1.sha256`); if the two ever disagree, the original prevails.

# Addendum 1 — B3.1 + B3.2: attacks, margin, BloodMNIST deviations and code

**Date:** 2026-10-05, after B3.0 and before any B3.1 run. Order from `PREREGISTRO.md` §8: pre-registration (`315b880`) → B3.0 → this addendum (commit) → launch.

## 1. B3.0 result (§3 rules, applied mechanically)

Details in `results/b30_medmnist_sanity/RESULTADO.md`.
- **Convergence:** met, borderline. T_Blood = 77.83%; final drift of +1.79 p.p. (< 2).
- **Attacks included:** **LMP and EB** (drops of 56.9 and 60.8 p.p. for FedAvg).
- **Margin:** **M = ±3.00 p.p.** (formula: 7.19 p.p.; capped at 3.0). **§3 declaration:** equivalence at this margin is **weak**. BloodMNIST has a baseline error ~7× MNIST's in the same official regime.
- **Grid:** 2 conditions × 2 attacks × 10 seeds = **40 runs**, interleaved.

## 2. BloodMNIST adaptation deviations (PREREGISTRO §2: "listed in the B3.0 addendum")

1. **Clients:** **96** instead of 100. The official code creates `int(num_clients / num_class)` clients per class group, which gives 12 × 8. `subsample_rate` = 10/96, keeping **10 clients per round**; 20 attackers.
2. **Local steps:** each client has ~124 training images, i.e. **1 batch of 64 per local epoch** (official `drop_last`), vs. ~9 on MNIST.
3. **Reward scale:** the loss is summed over ~53 test batches (vs. ~156 on MNIST), so the official reset threshold (reward < −80) becomes relatively harder to reach. Kept as in the official code.
4. **Extractor:** `MNISTClassifier` with 3 channels and 8 outputs, trained centrally on the BloodMNIST training split (Adam 1e-3, 15 epochs, seed 0; 90.5% on the test set), frozen. Hash below.
5. **Data:** the official MedMNIST v2 `.npz` (Zenodo 10519652), without the `medmnist` package, so as not to change the versions in the official venv. Per-channel normalization; augmentations as in the official MNIST branch.
6. **MNIST path intact:** the shim only acts when `dataset == "BloodMNIST"`.

## 3. Metric and aggregation

See `ADENDO0_metrica_agregacao.md`: accuracy is measured on the global test set (imbalanced on BloodMNIST), and AdaAggRL's aggregation does not use the sample size.

## 4. Execution

- **Launcher:** `scripts/passo2_oficial/run_grid_b31.sh "LMP EB" 6`. The queue is interleaved (td3/fixed alternating by seed and attack) and the launcher records the PGID.
- **Execution window** (author's rule): `scripts/janela_execucao.sh` suspends at 7h and resumes at 18h, Monday to Friday. **Exception for 10/05:** the author asked for the run to continue that day; therefore the B3.1 controller only starts at 18h on 10/05.
- **Estimated cost:** ~8–11 h per run with 6 processes (B3.0), i.e. 40 runs in ~7 batches, ~60–75 GPU hours. With the window, the expected end falls around the weekend of 10/10–10/11.

## 5. Code and artifacts at freeze time
91451db16a765b51197f8ec066912712cfbbb097df496b95069881ac9d0beb3b  scripts/passo2_oficial/bloodmnist_shim.py
89a86ebef9ead31341426bc2a18cde05e6412e0eeeaf4ab094b3e9cc6e0655b8  scripts/passo2_oficial/run_b3.py
1602c9486ce11abf9c4391c3f01e008f55c56ab3443b6d1acfe821d4410d73f0  scripts/passo2_oficial/run_grid_b31.sh
dff2220a1ee52664e7d9149f57df7a089536f8c82040ae974798616d4fbda81d  scripts/passo2_oficial/analisar_b31.py
2253b777eff9b192964a80c6ff2b75b7055c148e720a8083458eccb4cae68451  scripts/passo2_oficial/treinar_extrator_bloodmnist.py
c1546628896d91af7c5c77bb0fb58c95878dab846e0ee4ae8c99bb81c303a203  scripts/janela_execucao.sh
3e154c823ed1a75ac125de4bc4ebc509934f783c5d2e3c42daa35403600b33a1  data/models/extract_feature_bloodmnist.pt
062023e186f537e26b3c21ea3b2614ddfc475e8a14825dfd20663bbb7e37bddc  data/raw/medmnist/bloodmnist.npz
