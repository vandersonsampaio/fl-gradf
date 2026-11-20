# Addendum 1 — W2: code and checks (PLANO §5)

**Date:** 2026-10-24, before any run on seeds 92–96 or 180–189. Code at commit `396d571` (branch `exp/review`), unchanged since.

## Checks (`verificacao.txt`)

1. Regression (15 rounds, seed 42, `label_flipping` α 0.05, through this experiment's code path, with the weight tap and the ASR recorder): `td3_ref` 0.719400 vs B2.7 0.719400 and `fixed_b025` 0.725400 vs B2.7b 0.725400, |Δ| = 0 in both. **OK.**
2. Unit tests (`src/classification/tests/test_pixel_backdoor.py`): pass; full suite 106 passed. **OK.**
3. ASR sanity: a model trained only on triggered data relabelled to t has ASR = 1.0000 (4×4 and 6×6). **OK.**

**Note on the untrained model (no change to the plan):** the untrained model also gives ASR = 1.0000. With all-zero parameters every logit ties and argmax returns class 0, which is the target t. So "the untrained model's ASR" is not a base rate. The floor that the analysis uses is `fedavg_clean` (PLANO §2), which is unaffected. Round-1 ASR values can be inflated by the same tie only if the global model is still zero, which it is not after the first aggregation.

## Code hashes (sha256)

```
b9f11d327149e76edd8124dd89eb58c48c08d8e287017003b415f538c0427741  src/classification/attack_simulator.py
1b3155e9a114543f492a7e9363a0cdd782f49b989e01ec7b31b72422ba83bac8  src/fl/attacked_learner.py
79a13bfc30d32c375292c119ccff41424629e9837e59480b12b0ff26b0b6b777  scripts/w_common.py
0bbc877bd38f1de72069caa99f59be855ad2523d1d1e346b8de2ac4798b826e7  scripts/w2_backdoor_asr.py
159641cdf8ee32f44621addb61ced6fdcc86dadf0ecf0baab9828bebe061c8cd  scripts/run_grid_w2.sh
679b08060fc56244d81042699ccdbd2a8ef85fe0221c6029156feccb1dd9d529  src/classification/tests/test_pixel_backdoor.py
4525cd80e2d0b9ec70be759f4323e7285cbf4feac6107e89e93ec20a20de2e41  scripts/frente1_ablacao_adaaggrl.py
```
