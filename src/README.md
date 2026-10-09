# GRADF — architecture and implementation notes

Back to the [main README](../README.md).

- [Pipeline](#pipeline)
- [Modules](#modules)
- [Experiments](#experiments)
- [Known simplifications](#known-simplifications)
- [Design notes](#design-notes)
- [Data layout](#data-layout)
- [Tests](#tests)

---

## Pipeline

```
GRADFFederatedLearner._run_round(), per client:
    local training (+ attack injection if configured)
        ↓
    GradientAnalyzer + AccuracyDetector + TemporalDetector + ClusteringDetector
        ↓
    RLAttackClassifier  →  attack type + confidence
        ↓
    RLDefenseSelector   →  aggregation-rule vote
        ↓
    HardeningPipeline.full_pipeline()  →  validate, sanitize, aggregate (majority-voted rule), check accuracy
        ↓
    XAIExplainer.generate_explanation()  →  AuditTrail entry (JSON Lines)
```

Exercised end to end by `tests/test_integration.py` and `notebooks/05_attack_simulation.ipynb`.

---

## Modules

### FL core (`fl/`)

- `federated_learner.py` — `FederatedLearner`, the training loop. `_LogisticModel` (default) keeps all parameters as one flat vector `[W_flat, b]`, so every aggregation rule works on a 1-D array. `_CNNModel` (`model_type='cnn'`, `input_shape=(H, W, C)`) is a small Keras CNN with the **same** flat-vector interface, so nothing downstream changes; `_make_model` is the only dispatch point. Built-in rules: `FedAvgStrategy`, `FLTrustStrategy`.
- `attacked_learner.py` — `AttackedFederatedLearner` injects attacks every round (gradient level via `poison_update`, data level via `poison_data`). Also hosts the informed-attacker path and `RotatingAttackedFederatedLearner` (the attack type cycles every round).
- `gradf_learner.py` — `GRADFFederatedLearner`, which runs the full pipeline above.

### Attacks (`classification/`)

- `attack_simulator.py` — `AttackSimulator`: 11 GRADF attack types + TARS variants (`sign_flipping`, `gaussian_noise`), `label_flipping`, informed attacks (`fltrust_aligned`, `trim_attack`, `krum_collusion`, `low_mag_backdoor`), and `PretenseAttacker` (honest for N rounds, then attacks).
- `rl_classifier.py` — `RLAttackClassifier`: supervised softmax over the 4 detection scores (see [Known simplifications](#known-simplifications)).

### Detection (`detection/`)

| file | modality |
|---|---|
| `gradient_analyzer.py` | 1 — magnitude, windowed z-score, Jensen–Shannon divergence against the previous gradient |
| `accuracy_detector.py` | 2 — accuracy degradation against a validation set (the caller supplies old/new accuracy) |
| `clustering_detector.py` | 3 — cosine distance to the peer updates of the same round |
| `temporal_detector.py` | 4 — consistency with the client's own previous update |

`combiner_nn.py` merges the 4 scores into an anomaly probability (`DetectionCombiner`). `modality_recorder.py` bundles the 4 detectors; it lives here, not in `utils/`, to avoid a circular import with `fl/`.

### Defense (`defense/`)

- `aggregation_methods.py` — Median, Trimmed-Mean, FedProx, Krum and Clustering, registered with `register_strategy()`; with FedAvg and FLTrust, 7 rules in total.
- `rl_selector.py` — `RLDefenseSelector`: a DQN over a 5-dimensional state `[attack_type_idx, confidence, prev_accuracy, prev_fairness_std, prev_latency]`, with a target network and a replay buffer. It is pretrained offline (`generate_selector_experiences`, `generate_rotating_selector_experiences`) and keeps training online every FL round (`GRADFFederatedLearner._update_selector_online`).
- `hardening.py` — `HardeningPipeline`: L1 magnitude/skewness validation → L2 DP clipping + noise → L3 classifier + selector choose the rule (majority vote) → L4 accuracy-regression check → L5 HE stub.
- `dp_accountant.py` — `GaussianDPMechanism`, an opt-in Gaussian mechanism with RDP (ε, δ) accounting (see [Known simplifications](#known-simplifications)).
- Competing defenses, reproduced from their papers' specifications (deviations documented in each module):
  - `tars_selector.py` — TARS: trust score + tabular ε-greedy Q-learning over the aggregation rules;
  - `adabfl_aggregator.py` — AdaBFL: client filter → trimmed mean → a "derivative model", fused by self-tuning weights;
  - `fedstrategist_selector.py` — FedStrategist: LinUCB contextual bandit over the rules, with the paper's per-rule costs;
  - `adaaggrl_agent.py` — AdaAggRL: gradient inversion + MMD cues + TD3 over continuous per-client weights, with a persistent penalty for flagged clients.

### XAI (`xai/`)

- `explanation_generator.py` — `XAIExplainer`: narrative + a counterfactual computed from the real old/new accuracy, for every ACCEPT/REJECT decision.
- `audit_trail.py` — `AuditTrail`: JSON Lines persistence (`add`, `load`, `query`, `clear`).

### Utilities (`utils/`)

- `data_loader.py` — `load_dataset_participants()` (with `root_size`/`root_seed` to subsample the FLTrust root) and generators of real training data for the classifier, combiner and selector (`generate_detection_dataset`, `generate_selector_experiences`, `generate_rotating_selector_experiences`).
- `stats.py` (`run_over_seeds`, `summarize`, `paired_significance`), `metrics.py`, `config_loader.py`, `logger.py`, `visualization.py`, `hospital_splitter.py` (MIMIC-III partitions, blocked on data access).

---

## Experiments

All in `experiments/`, run as modules (`python -m src.experiments.<name>`).

| experiment | purpose | main options |
|---|---|---|
| `exp1_baseline` | system comparison + per-component ablation (FedAvg, FLTrust, Median, Trimmed-Mean, RL-only, Hardening-only, Random/Oracle selector, GRADF); reports wall-clock cost | `--seeds`, `--attack_types`, `--root_size`, `--ground_truth_oracle` |
| `exp1_cnn_scale` | classical defenses on a small CNN (MNIST, CIFAR-10) | `--seeds` |
| `exp2_robustness` | robustness per attack and Byzantine fraction; retention vs. FedAvg and vs. the best fixed rule | `--seeds` |
| `exp3_scalability` | scaling with the number of clients | — |
| `exp4_adaptive` | adaptive attacker (rotating attacks): GRADF vs. static defenses vs. TARS, AdaBFL, FedStrategist | `--seeds`, `--root_size` |
| `exp5_clinical` | export/analysis tooling for a clinical survey (no data is fabricated) | — |
| `exp6_fairness` | fairness across clients | — |
| `exp7_modality_ablation` | multi-modal vs. single-modality detection | `--seeds` |
| `exp8_xai_examples` | qualitative XAI examples from `exp2`'s audit trails | — |
| `exp9_dominance_grid` | fixed-rule dominance across attack × α; root-size sensitivity | `--seeds`, `--root_size`, `--strategies` |
| `exp10_selector_comparison` | GRADF vs. FedStrategist vs. AdaAggRL vs. Random/Oracle (root = 100) | `--variant a\|b`, `--adaaggrl_feature_extractor` |

`--root_size N` adds `_root{N}` to the output names, so a sensitivity run never overwrites the canonical results.

---

## Known simplifications

- **`RLAttackClassifier` is not literal RL.** It is a supervised classifier: the true attack label is known during training. `RLDefenseSelector` is genuine RL, trained on rewards from real FL rounds.
- **`FedProxStrategy` is an approximation.** Real FedProx changes the *local* objective, which an aggregation-only interface cannot see; here it is a damped weighted mean at the server.
- **DP noise depends on dimensionality.** Layer 2 adds Laplace noise per coordinate, so its L2 norm grows with `sqrt(n_parameters)`. The default `dp_epsilon=50.0` is calibrated for MNIST-scale models (~7,850 parameters); other model sizes need a rescaled epsilon.
- **The Gaussian DP mechanism is opt-in.** `GaussianDPMechanism` (L2 clip, Gaussian noise added once to the aggregate, RDP accounting, tight sensitivity for mean-type rules and a conservative bound for the robust ones) only applies when passed as `HardeningPipeline(dp_mechanism=...)`; the default keeps the Laplace mechanism, so reported numbers do not change. It protects whoever sees the released global model, not against the server itself.
- **Modality ablation masks inputs.** "Single-modality" classifiers zero the other 3 inputs instead of retraining a smaller network, which keeps the architecture fixed.
- **CNN scale is clean on MNIST, ambiguous on CIFAR-10.** On MNIST all three defenses beat FedAvg by a wide margin (Cohen's d ≈ 9); on CIFAR-10 none is distinguishable from FedAvg at the reduced scale used (15 rounds, 3 seeds). Not resolved; needs a longer run before citing CIFAR-10 numbers.
- **The FLTrust root size decides the headline result.** The default root is 3,000 MNIST samples, 30× the ~100 of the original FLTrust paper. With `--root_size 100`, FLTrust's win rate falls from 18/21 to 8/21 cells. A large root also means centralizing raw client data, an open privacy question.
- **The `exp1` Oracle selector is gated by the real classifier,** so it is not an upper bound on weakly detected attacks (`sign_flipping`, `label_flipping`). The decoupled oracle (`--ground_truth_oracle`, also used by `exp10`) bypasses the classifier with the true attack index.

---

## Design notes

- **Two frameworks.** TensorFlow/Keras for the neural models (classifier, selector, combiner, CNN); NumPy for the logistic FL model.
- **Safe imports.** Every module can be imported without side effects; training and demo code sits behind `if __name__ == '__main__':`.
- **Import cycles.** `fl/` needs `detection/`, and `utils/data_loader.py` needs `fl/`; check both directions before moving a cross-cutting helper.
- **Isolate the audit trail per run.** `XAIExplainer()`/`AuditTrail()` default to one shared file; scripts that loop over runs and read the trail must pass a per-run `AuditTrail(path=...)`, cleared before training (as `exp2_robustness.py` does).
- **`HardeningPipeline` always needs `server_update`.** The selector can vote for FLTrust in any round, regardless of the learner's own `aggregation`, so `GRADFFederatedLearner` computes the server update every round.
- **Seeds and reproducibility.** A learner's `seed` does not seed the global `np.random`, so results can depend on execution order; GRADF's DQN is also non-deterministic across runs (see `results/d2_variancia_p1/`).

---

## Data layout

```
data/
  raw/        mnist/, cifar10/   (X_train.npy, y_train.npy, X_test.npy, y_test.npy)
  processed/  mnist/, cifar10/   (iid/, non_iid/, non_iid_a{α}/  →  server_val/, client_0/ … client_9/)
```

`server_val/` is the FLTrust root (5% of the training pool). Non-IID heterogeneity (Dirichlet α = 0.5): MNIST ≈ 0.65, CIFAR-10 ≈ 0.67. MIMIC-III (`data/download_mimic.py`) requires credentialed PhysioNet access.

---

## Tests

```bash
pytest                              # all 95 tests
pytest src/detection/tests/         # detection modalities + combiner
pytest src/classification/tests/    # attack classifier
pytest src/defense/tests/           # aggregation rules + defense selector
pytest tests/test_integration.py    # full GRADF pipeline and audit trail
pytest tests/test_utils.py          # config, metrics, logging, plotting
```

`tests/fixtures/mock_data.py` provides synthetic `ParticipantData`, so tests do not need the real datasets.
