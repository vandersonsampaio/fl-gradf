# P1 CIFAR-10 campaign: implementation prerequisites and reproduction deviations

Plan: `results/p1_plan/PLAN.md`, Sec. 5.0 ("Implementation prerequisites before E0 runs").
Code state: branch `p1/segunda-submissao`. The checks below are unit tests on synthetic data; on CIFAR-10 only the E0 cost probe has run so far (`e0_bench/SUMMARY.md`, not yet versioned).

## 1. Prerequisites

| Prerequisite (plan Sec. 5.0) | Status | Where |
|---|---|---|
| CIFAR-10 runner: 7 rules × 7 attacks × 3 α, seeded, roots 100 and 2,500 | **done** | `exp9_dominance_grid --dataset cifar10 --model cnn --seed_rng [--root_size 100] --strategies fltrust median trimmed_mean krum fedavg fedprox clustering` |
| AdaAggRL gradient inversion on the CNN | **done** | `adaaggrl_agent.reconstruct_client_distribution_cnn`, used by `AdaAggRLGridLearner` when `model_type='cnn'` |
| GRADF Layer-2 noise recalibrated to the CNN's size | **done** (ε); Layer-1 threshold open (D5) | `hardening.legacy_dp_epsilon_for`, applied by `exp10_selector_comparison --model cnn` |
| Adaptive systems on the CNN (Random, attack-aware, GRADF incl. pretraining, FedStrategist, AdaAggRL), seeded | **done** | `exp10_selector_comparison --dataset cifar10 --model cnn --seed_rng` |
| Each item recorded as a deviation | this file | |

Root sizes: the CIFAR-10 `server_val` has 2,500 images (5% of the pool), so the enlarged root is the full set (no `--root_size`) and the canonical root is `--root_size 100`.

Tests (synthetic 4×4×1 images): `src/fl/tests/test_seeded_cnn.py`, `src/fl/tests/test_systems_cnn.py`, `src/defense/tests/test_adaaggrl_cnn.py`. Full suite: 82 passed. Regression: seeded MNIST runs are bit-identical to `results/p1_invariance/` (seed 42, root 100, `trim_attack` and `fltrust_aligned`, FLTrust and TM).

## 2. Deviations

Columns: specification · reproduction · reason · expected impact · classification (faithful / necessary / sensitivity).

**D1. Task model.**
- Specification: the plan's small CNN (2–3 conv layers), not ResNet-20.
- Reproduction: the existing `_CNNModel`: Conv(8)–Pool–Conv(16)–Pool–Dense(32)–Dense(10), SGD, 34,522 parameters on CIFAR-10. Unchanged from `exp1_cnn_scale`.
- Reason: keeps the flat-vector interface of the seven rules and the cost feasible on CPU.
- Expected impact: lower absolute accuracy than the original papers' models; the comparisons are within this model.
- Classification: necessary.

**D2. Reproducibility of CNN runs.**
- Specification: plan Sec. 3 (seeded runs, exact invariance check).
- Reproduction: with `--seed_rng`, each run seeds Python/NumPy/Keras from crc32(seed, α, attack, rule or system) via `keras.utils.set_random_seed`, enables TensorFlow deterministic ops, and takes the mini-batch order from NumPy (`_CNNModel.numpy_shuffle`) instead of Keras' own shuffling. The server's root fit uses its own RNG (`server_rng_seed`), as on MNIST. Keras 3 keeps its own seed state, so `tf.random.set_seed` alone was not enough (found by the test). Since the memory fix (`e0_bench/SUMMARY.md`, Memory), one compiled Keras model is shared per process and the initial weights (Glorot-uniform, zero biases) are drawn from NumPy, so no TensorFlow random draw remains in seeded runs.
- Reason: bit-for-bit reproducibility and the exact TM root-invariance check (first output of E1).
- Expected impact: none on the method; deterministic ops may slow training (measured in E0).
- Classification: faithful.

**D3. AdaAggRL gradient inversion on the CNN.**
- Specification: AdaAggRL (Wang et al., AAAI 2025) reconstructs client data by gradient inversion on the FL model.
- Reproduction: inversion on the CNN with the client's parameters θ_k; the dummy gradient (all trainable variables, flattened in `_CNNModel.params` order) is matched to `update / local_lr` by cosine similarity, Adam on the dummy images, 30 iterations, 16 images, TV weight 1e-4 (as in the linear version). Dummy images start from seeded uniform noise in [0, 1) instead of zeros, and the TV prior runs over both image axes.
- Reason: the existing implementation only supported the linear model; with ReLU layers an all-zero input gives degenerate gradients.
- Expected impact: closer to the paper than the linear version. `update / local_lr` is an approximation of the gradient when local training has several mini-batches or epochs, as on MNIST. Per-round cost is higher (second-order gradients through the CNN), measured in E0.
- Classification: faithful (inversion on the model) / necessary (initialization).

**D4. AdaAggRL feature extractor.**
- Specification: an unspecified pretrained CNN.
- Reproduction: the main configuration keeps the frozen random CNN (`RandomCNNFeatureExtractor`, the documented floor), as on MNIST. The pretrained-extractor sensitivity needs `results/models/adaaggrl_pretrained_extractor_cifar10.weights.h5`, which does not exist yet (`scripts/pretrain_adaaggrl_extractor.py --dataset cifar10`, trained on the CIFAR-10 test split).
- Reason: same as on MNIST.
- Expected impact: AdaAggRL's main result is a floor.
- Classification: sensitivity (pending the extractor weights).

**D5. GRADF hardening calibration on the CNN.**
- Specification: Layer 2 noise calibrated for the MNIST logistic model (ε = 50, 7,850 parameters); its per-coordinate Laplace noise has L2 norm ≈ √(2d)/ε.
- Reproduction: ε = 50·√(d/7,850) = 104.9 for d = 34,522, which keeps the noise L2 norm of the MNIST calibration (`legacy_dp_epsilon_for`). The per-coordinate clip (±2.0) is unchanged.
- Open item: Layer 1 rejects updates whose L2 norm exceeds an absolute 5.0, also calibrated on MNIST. It is not rescaled yet: the CNN's honest-update norms on CIFAR-10 are unknown. To be measured in E0 (honest update norms per round) and set so that the honest acceptance rate matches MNIST, before the E3 pre-registration.
- Reason: both quantities depend on the model's size and scale.
- Expected impact: without recalibration, noise or Layer-1 rejections could dominate GRADF's result.
- Classification: necessary.

**D6. GRADF pretraining and the attack-aware reference on the CNN.**
- Specification: classifier and selector pretrained on the same task model as the experiment.
- Reproduction: `train_gradf_models(..., model_kwargs=...)` runs the detection and selector episodes (8 rounds each, 7 actions per attack type) on the CNN. The attack-aware reference (`Oracle (decoupled)`) uses the oracle map derived from these episodes.
- Reason: rewards measured on the logistic model would not transfer to the CNN.
- Expected impact: pretraining is the largest fixed cost of GRADF and the attack-aware reference on CIFAR-10; measured in E0.
- Classification: faithful.

**D7. FedStrategist arsenal.** Unchanged from MNIST: the paper's three-rule arsenal extended to the seven rules. Classification: necessary (same as MNIST).

## 3. Still to check in E0 (before the E0 pre-registration)

- Honest update norms on the CNN, to set GRADF's Layer-1 threshold (D5).
- Cost per round: fixed rules, GRADF (with pretraining), FedStrategist, AdaAggRL (inversion), with and without deterministic ops.
- The project's TensorFlow does not see the GPU; E0 decides whether to fix that or run the campaign on CPU (plan Sec. 5.0, device per experiment).
