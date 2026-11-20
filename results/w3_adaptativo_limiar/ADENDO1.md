# Addendum 1 — W3: code and checks (PLANO §5)

**Date:** 2026-10-24, before any run on seeds 125–129 or 190–199. Code at commit `396d571` (branch `exp/review`), unchanged since.

## Checks (`verificacao.txt`)

1. Regression (15 rounds, seed 42, `label_flipping` α 0.05, `td3_ref`, through this experiment's learner with the feedback hook): 0.719400 vs B2.7 0.719400, |Δ| = 0. **OK.**
2. Unit tests (`tests/test_w3_adaptive.py`: mixing endpoints, AIMD bounds, tap normalization, Holm/verdicts): pass; full suite 106 passed. **OK.**

## Code hashes (sha256)

```
79a13bfc30d32c375292c119ccff41424629e9837e59480b12b0ff26b0b6b777  scripts/w_common.py
d26ccb3c6fbbfb85a359d52520936c850cc157599699c04d54ab6f2f245152e0  scripts/w3_adaptativo_limiar.py
a322dc593e2d7df8faa4185c05a931df89e26385107b5a94c9578841aee1b588  scripts/run_grid_w3.sh
792f01e23b3793b9ba2f56e2b1347883b562038491a2abe826c63148fa4bfd78  tests/test_w3_adaptive.py
4525cd80e2d0b9ec70be759f4323e7285cbf4feac6107e89e93ec20a20de2e41  scripts/frente1_ablacao_adaaggrl.py
```
