# Checks before the first confirmatory run

**Date:** 2026-11-04. All runs on development seeds (0, 1), none on seeds 42–51. Outputs in this directory.

| check (PREREGISTRATION, Sec. 8) | result | evidence |
|---|---|---|
| 1. `global_lr = 1` reproduces earlier code bit for bit, logistic and CNN | PASS. 18 of 19 learners in `logistic_pretrained` match; the one that differs is the final GRADF, whose DQN is not deterministic. The check asks for a match "where deterministic". | `probe_old.json`, `probe_new.json` |
| 2. `global_lr = 1/3` reproduces `cifar_step_pilot` for FLTrust | PASS, seeds 0 and 1, 3 rounds | `pilot.txt` |
| 3. `global_lr = 1/3` gives 1/3 of the step in each learner | PASS, except that GRADF after pretraining gives 0.487, with the same DQN | `probe_third.json` |
| 4. Pretraining receives the factor | PASS, 19 learners in the probe; also 17 trains per job in the dry runs | `probe_third.json`, `dry/` |
| 5. A job with a wrong setting is refused | PASS, 9 of 9 | `guard.txt` |

**Dry runs (seed 0, 2 rounds).** Every arm kind ran to the end: fixed, server-steps sensitivity, hardened, Random, GRADF, attack-aware, FedStrategist with detector, AdaAggRL with both extractors. Hardened arms over 20 rounds do not get stuck: Krum accepted 15 of 20 rounds and Random 20 of 20. `dry/`, `dry20/`.

**Across machines (factor 1/3, 2 rounds, SHA-256 of final parameters).** This PC and both VMs match on `cnn_fedavg`, `cnn_fltrust`, `cnn_krum_informed`, `cnn_random`, `cnn_fedstrategist` (`probe_third_vm2.json`, `probe_third_vm3.json`) and on AdaAggRL with the random extractor (`3bb2c558…`) and the pretrained extractor (`564a995b…`). The VMs run with `ONEDNN_MAX_CPU_ISA=AVX2` and `OPENBLAS_CORETYPE=Haswell`; on this PC those two settings change nothing (`inv/`).

**Root invariance on a development seed.** The trimmed mean at roots 100 and 2,500, seed 0, 2 rounds, `trim_attack`: identical parameters (`inv/`). The confirmatory check runs first in the campaign.

**Fixes found by these checks, included in the frozen code:**
- `exp10` passed `input_shape` twice to AdaAggRL on the CNN, so that arm could not run at all before.
- τ_CNN did not reach `exp10`, which would have used 5.0. It is now passed as `magnitude_threshold`.

**AdaAggRL extractors (SHA-256).** CIFAR-10: `5b042983efbeb61739c87c40b840d72569f12c62c1bf556141e3d6617fbef4c5`, trained with `scripts/pretrain_adaaggrl_extractor.py --dataset cifar10 --seed 0` (validation accuracy 0.4720). MNIST: `7d726b5c9eee6c1db7cc311f502d626610bc2d4ead0be24d6843cdd72074c19d`. As for MNIST, the script trains on the raw test split.

**Data.** The 126 files of CIFAR-10's `non_iid`, `non_iid_a0.1` and `non_iid_a0.05` have the same SHA-256 on this PC and both VMs. `iid` exists only on this PC, and the campaign does not use it.
