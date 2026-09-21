# FedMD — Federated Learning Security Framework

## Overview

**FedMD** is a research-grade Federated Learning security framework targeting multi-hospital medical scenarios (MIMIC-III). The core research contribution is **GRADF** — a framework combining:

1. Multi-modal attack detection (gradient, accuracy, temporal, clustering signals)
2. RL-based attack classification (11+ attack types)
3. RL-based defense selection (7 aggregation strategies)
4. 5-layer hardening pipeline
5. XAI audit trail for every aggregation decision

The framework also reproduces the **TARS** experimental setup (Ahmed et al., 2025): N=10 clients, f=2 Byzantine (20%), MNIST and CIFAR-10 with non-IID Dirichlet(α=0.5) partitioning. `exp4_adaptive.py`'s Gate 1 comparison additionally reproduces two more competing adaptive baselines from the security-FL literature: **AdaBFL** (Tang, Liu & Huang, 2026 — adaptive multi-layer aggregation, no linked code repo) and **FedStrategist** (Haque, Kamal & Hossain, 2025 — LinUCB contextual bandit over the aggregation rule; its public repo targets a different model representation than this codebase, so it's a from-spec reproduction, not a port — see `src/defense/adabfl_aggregator.py`/`fedstrategist_selector.py`).

The full detection → classification → defense-selection → hardening → XAI pipeline is implemented and orchestrated by `GRADFFederatedLearner`, and runs end-to-end on MNIST/CIFAR-10 (already downloaded and partitioned). MIMIC-III itself is not downloaded (requires PhysioNet credentials — see below).

### Paper framing — read this before "Publication scope" below

The paper's central claim has been rewritten twice as evaluation rigor increased; **the framing below (and the "adaptive adversary — the security centerpiece" language a few paragraphs down) is the ORIGINAL, now-superseded framing** — kept here for implementation-scope purposes (which experiments exist, which are in/out), not as the paper's actual thesis. The current thesis lives in `docs/paper_A_esboco_v2.md`:

- **v1** (`docs/paper_A_esboco.md`): reframed around a negative result — under the root dataset size in use at the time (3,000 MNIST samples, ~5% of the training pool), FLTrust dominates every fixed-vs-adaptive comparison and no headroom for adaptive selection exists. Conclusion: adaptive selection doesn't pay off; the framework's real value is multi-modal detection fusion + XAI.
- **v2** (`docs/paper_A_esboco_v2.md`, current): supersedes v1. The 3,000-sample root turned out to be the confound — it's 30× the canonical value used in the original FLTrust paper (Cao et al., NDSS 2022, ~100 samples), and a size that large centralizes exactly the data FL exists to avoid centralizing. Re-run under the canonical root size (100 samples, 10 seeds, paired significance): FLTrust's dominance collapses, headroom for adaptive selection reappears, and a ground-truth-decoupled oracle proves that headroom is capturable in principle. But the *deployed* system — the DQN selector, a from-scratch TARS reproduction, even an oracle gated by the real classifier — still fails to capture it, specifically on the two attacks with weak detection (`sign_flipping`, `label_flipping`, both magnitude-preserving). Current thesis: **the ceiling on adaptive defense in Byzantine-robust FL is detection, not selection.**
- Full experimental trail (code changes, every table, seed counts, caveats): `references/sensibilidade_root_fltrust.md`. Follow-on architecture question this raises for the MIMIC-III path (a FLTrust root large enough to matter means centralizing raw patient data — three candidate designs, none implemented yet): `references/decisao_root_dataset_lgpd.md`. LaTeX draft in the ICLR 2027 template (not yet updated to v2's framing): `docs/paper/gradf_iclr2027.tex`.
- `exp9_dominance_grid.py` (not listed below in the original implementation table) is the experiment behind this: a dominance grid over heterogeneity × attack × fixed defense, parameterized by root size via `--root_size`/`--strategies`. `exp1_baseline.py` and `exp4_adaptive.py` also take `--root_size` now, for the same reason.

### Publication scope (conference paper vs. journal extension) — original framing, see above for current status

For the current conference submission, the paper is scoped as a **general FL security framework validated on standard non-IID benchmarks** (MNIST/CIFAR-10), not a medical-specific system. XAI is presented as part of the framework plus a qualitative illustration (`exp8_xai_examples.py`) — **without** the human-clinician validation study. The following are explicitly deferred to a future, expanded journal version:

- **MIMIC-III** (real multi-hospital data — see "Blocked on external dependencies" below).
- **XAI clinical validation** — the 15-clinician Likert survey (`exp5_clinical.py` tooling exists; no responses collected).
- **`exp3_scalability`** and **`exp6_fairness`** — kept implemented and runnable, just not part of the paper's core result set.

The paper's central experiments are `exp1_baseline` (system comparison **and** per-component ablation), `exp2_robustness`, `exp4_adaptive` (adaptive adversary), and — as of v2 — `exp9_dominance_grid` (root-size sensitivity / headroom), all run over multiple seeds with mean/std/95% CI and paired significance testing (`src/utils/stats.py`). `exp7_modality_ablation` backs the "multi-modal detection beats single-modality" claim with numbers instead of narrative.

---

## Implementation Status

### Complete — runnable today

| Component | File | Description |
|-----------|------|-------------|
| FL training loop | `src/fl/federated_learner.py` | FedAvg + FLTrust; `ParticipantData`, `RoundResult`; softmax logistic regression with mini-batch SGD |
| Attack injection | `src/fl/attacked_learner.py` | `AttackedFederatedLearner` — per-round attack injection (extracted from notebook 08, reused across the codebase) |
| GRADF orchestrator | `src/fl/gradf_learner.py` | `GRADFFederatedLearner` — wires detection → combiner → classifier → selector → hardening → XAI into one FL round |
| Attack simulator | `src/classification/attack_simulator.py` | 11 GRADF attack types + TARS paper variants (`sign_flipping`, `gaussian_noise`, `label_flipping`, `PretenseAttacker`) |
| Multi-modal detection | `src/detection/{gradient,accuracy,temporal,clustering}_detector.py` | All 4 modalities implemented, sharing the `GradientAnalyzer`-style windowed z-score pattern |
| Detection combiner | `src/detection/combiner_nn.py` | `DetectionCombiner` wraps `AnomalyCombinerNN` with `.train/.predict/.save/.load` |
| Attack classifier | `src/classification/rl_classifier.py` | `RLAttackClassifier` — supervised softmax classifier trained on real detection signals (see Design Notes on why it's not literal RL) |
| Defense selector | `src/defense/rl_selector.py` | `RLDefenseSelector` — genuine DQN trained on real FL-round rewards |
| Aggregation strategies | `src/defense/aggregation_methods.py` + `federated_learner.py` | All 7: FedAvg, FLTrust, Median, Trimmed-Mean, FedProx (simplified, see notes), Krum, Clustering |
| Hardening pipeline | `src/defense/hardening.py` | 5 layers, actually dispatches to classifier → selector → chosen aggregation strategy |
| XAI + audit trail | `src/xai/explanation_generator.py`, `src/xai/audit_trail.py` | Structured explanations with computed (not hardcoded) counterfactuals, persisted as JSON Lines |
| Data generation for training | `src/utils/data_loader.py` | `generate_detection_dataset`, `generate_selector_experiences`, `load_dataset_participants` — real FL rounds, not `np.random` |
| Dataset pipeline | `data/download_datasets.py` | MNIST + CIFAR-10 download; IID + Dirichlet non-IID partitioning into 10 client shards |
| Hospital splitter | `src/utils/hospital_splitter.py` | Non-IID hospital partitions from MIMIC-III (blocked on data — see below) |
| Config | `config/config.yaml`, `config/hyperparameters.yaml` + `src/utils/config_loader.py` | FL/GRADF parameters and network hyperparameters |
| Shared utils | `src/utils/{metrics,logger,visualization}.py` | Detection rate, FPR, fairness std; central logging; shared plotting |
| Experiments 1–4, 6, 7, 9 | `src/experiments/exp{1,2,3,4,6,7,9}_*.py` + `runner.py` | Real `FederatedLearner`/`GRADFFederatedLearner` calls, CLI-driven, save to `results/`. `exp1`/`exp2`/`exp4`/`exp7`/`exp9` support `--seeds` (multiple values) and report mean/std/95% CI via `src/utils/stats.py`; `exp1` also reports a paired t-test of each system vs. FedAvg. `exp1`/`exp2` also report wall-clock time (`wall_clock_seconds`, and `pretrain_seconds` for the classifier/selector pretraining shared by RL-only/Hardening-only/GRADF) — the "cost of defense" numbers, kept separate from the `n_clients`-scaling analysis in `exp3`. `exp1`/`exp4`/`exp9` also take `--root_size N` (subsamples the FLTrust root dataset for the current paper's central finding — see "Paper framing" above); `exp9` additionally takes `--strategies` to re-run only a subset of the 4 fixed defenses (only `fltrust` actually depends on `--root_size`) |
| Experiment 5 tooling | `src/experiments/exp5_clinical.py` | Export/analyze tooling for clinical validation — does **not** synthesize clinician responses (see below) |
| XAI qualitative examples | `src/experiments/exp8_xai_examples.py` | Pulls 1 real ACCEPT + 1 REJECT example (computed narrative/counterfactual) and a decision-rate table from `exp2`'s saved audit trails. Not a clinical study — run `exp2` first |
| Notebooks 02–08 | `notebooks/02_*.ipynb` → `notebooks/08_*.ipynb` | All executed, real outputs saved (MNIST-based) |
| Test suite | `src/**/tests/`, `tests/` | 95 tests: unit + integration (pytest, see below) |

### Blocked on external dependencies

| Component | File | Blocker |
|-----------|------|---------|
| MIMIC-III data | `data/download_mimic.py`, `data/merge_mimic_tables.py` | Requires a PhysioNet credentialed account (CITI training). Scripts are ready and fail with a clear error until then. |
| Notebook 01 (MIMIC EDA) | `notebooks/01_eda_mimic.ipynb` | Code is written and correct but **not executed** — depends on MIMIC-III data existing. |
| Clinical validation (exp5) | `src/experiments/exp5_clinical.py` | Requires 15 real clinicians to fill in a Likert survey. The export/analyze tooling is complete; the human data collection is out of code's scope. |

### Known simplifications (documented in code)

- **`RLAttackClassifier` is not literal RL** — it's a supervised softmax classifier trained on real detection signals (ground-truth attack labels come from the simulator during training). `RLDefenseSelector` **is** genuine RL: it trains on rewards observed from real FL rounds. See the design note in `src/classification/rl_classifier.py`.
- **`FedProxStrategy`** approximates the original FedProx (Li et al., MLSys 2020) as a damped weighted mean at the server, since the real algorithm modifies the *local* training objective — not observable from already-computed client deltas at the aggregation-only interface used here. See `src/defense/aggregation_methods.py`.
- **DP noise calibration is dimensionality-sensitive**: Layer 2's Laplace noise is added per-coordinate, so its total L2 norm scales with `sqrt(n_parameters)`. The default `dp_epsilon=50.0` is calibrated for MNIST-scale models (~7850 parameters); a much smaller model needs a smaller epsilon, and a much larger model needs a larger one, or the noise will swamp the signal (this was found and fixed during Fase 6 — GRADF was underperforming plain FedAvg with `dp_epsilon=1.0` on MNIST before recalibration). See the docstring in `src/defense/hardening.py`.
- **Audit trail must be isolated per run**: `XAIExplainer()`/`AuditTrail()` default to the same shared path (`results/audit_trail.jsonl`) across every instance. `exp2_robustness.py` originally created a fresh `GRADFFederatedLearner()` per `(attack_type, byzantine_fraction)` combination and read `.audit_trail.load()` right after — which silently accumulated records from every earlier combination in the same loop and inflated `detection_rate`. Fixed by giving each run its own `AuditTrail(path=f"results/audit_trail/exp2_{attack_type}_{frac}_{seed}.jsonl")`, cleared before training. If you add a new multi-run script that reads the audit trail mid-loop, apply the same pattern.
- **Modality ablation via masking, not architecture change** (`exp7_modality_ablation.py`): `RLAttackClassifier`'s network has a fixed 4-feature input. "Single-modality" classifiers are simulated by zeroing the other 3 columns rather than retraining a smaller network — keeps the comparison architecture-controlled, at the cost of not testing whether a smaller network would do better with fewer inputs (not the question this ablation asks).
- **Gradient Jensen-Shannon divergence bug (found and fixed during the production run)**: `GradientAnalyzer.js_divergence` compared the current gradient's histogram against `np.histogram()` of a *scalar* — `gradient_history[hospital_id][-2]`, which stores past **magnitudes** for the z-score, not past gradient vectors. A scalar histogram is degenerate (all mass in one bin), so the divergence against it was large and nearly constant regardless of actual behavior — verified by feeding the *same* gradient every round and observing `anomaly_score` stuck at ≈0.6 from round 2 onward instead of ≈0. This added a spurious anomaly floor to every client, honest or not, and diluted the real signal from attacks (like `sign_flipping`/`label_flipping`) that don't move the gradient's magnitude. Fixed by tracking the previous raw gradient vector separately (`GradientAnalyzer.last_gradient`) instead of reusing the magnitude history. See `src/detection/gradient_analyzer.py`.
- **`fltrust` selectable by the RL selector without a `server_update` (found and fixed during the production run)**: `RLDefenseSelector.actions` includes `'fltrust'`, and the selector can vote for it in any round of `GRADFFederatedLearner`, independent of the learner's own top-level `aggregation=` setting. `HardeningPipeline.layer3_secure_aggregation` never supplied the `server_update` that `FLTrustStrategy.aggregate` requires, so whenever the selector voted `fltrust`, aggregation raised `ValueError`, the round was silently discarded (`round_accepted` stays `False`), and — if this repeated every round — the global model never moved past random initialization (~10% accuracy on 10-class MNIST). Reproduced with `exp1_baseline.py --seeds 45` on `gaussian_noise` (`GRADF`/`RL-only` collapsed to 0.098). Fixed by computing a `server_update` every round in `GRADFFederatedLearner._run_round` (fit on `root_data`, or a carved root sample if unavailable) and threading it through `HardeningPipeline.full_pipeline` → `layer3_secure_aggregation` → `strategy.aggregate(...)`. See `src/fl/gradf_learner.py` and `src/defense/hardening.py`.
- **`RLDefenseSelector.train_on_experiences` was undertrained by default (found and fixed during Gate 1 investigation)**: it did a single gradient step per experience with no `epochs` parameter at all. With the tiny buffers produced by `generate_selector_experiences` (14–28 attack-type × strategy pairs), a diagnostic isolating this showed the DQN converging to the right action for 3 of 4 attack types purely by luck of the first gradient step, but to the WRONG action for `label_flipping` (picked `median`; only `clustering` neutralizes it in the reward buffer). `epochs=200` (added as a parameter, default kept at `epochs=1` for backward compatibility with `tests/test_defense.py`) converges correctly on all 4. Production callers (`exp1_baseline.py`, `exp4_adaptive.py`) now pass `epochs=200` explicitly. **Fixing this did not change GRADF's Gate 1 outcome** — a real, separate bug, not the root cause of that result. See `src/defense/rl_selector.py`.
- **The RL selector's implementation diverged from the proposal's own formalism, and never learned online (fixed post-Gate-1)**: `FORMALISMO_MATEMATICO_E_INEDITISMO.md` specifies a 5-dimensional selector state (`[â, confiança, acc^{t-1}, fairness_std^{t-1}, latência^{t-1}]`), but the code only ever used `[attack_type_idx, confidence]` (2 dims) — a real gap between what the proposal defines and what was implemented, not evidence against the proposal itself. Separately, `RLDefenseSelector` was trained exactly once, offline, before the first FL round, and then frozen for the entire run — "adaptive" described the architecture (choosing an action per round), not the learning behavior. Comparing this to the TARS reproduction (`src/defense/tars_selector.py`) made this stand out: TARS updates its policy every real round. Fixed: (1) `RLDefenseSelector` now uses the full 5-dim state, plus a target network and replay buffer (previously absent — both standard DQN stabilizers); (2) `HardeningPipeline.full_pipeline`/`GRADFFederatedLearner` now call `train_step` every real FL round with the actually-observed reward and next-state (see `GRADFFederatedLearner._update_selector_online`), not just at pretraining time; (3) `generate_selector_experiences`'s `next_state` used to be a literal copy of `state` (a degenerate, no-op Bellman target) — now uses the episode's real post-outcome accuracy/fairness/latency; (4) a new `generate_rotating_selector_experiences` (`src/utils/data_loader.py`, using `RotatingAttackedFederatedLearner`/`attack_for_round` moved to `src/fl/attacked_learner.py`) trains the selector under the same attack-rotation condition Gate 1 actually tests, closing a train/deploy mismatch (pretraining previously only used static single-attack episodes). Whether this changes the Gate 1 outcome is being re-measured at production scale — see the plan doc for the result once available.
- **`accuracy_retention` alone overstates GRADF's competitiveness (Decision A5)**: originally measured only against FedAvg (no defense) — `retention > 1.0` just means "beats doing nothing." `exp2_robustness.py` now also runs FLTrust/Median/Trimmed-Mean per `(attack_type, byzantine_fraction, seed)` combination and reports a second column, `accuracy_retention_vs_best_fixed` (vs. whichever of the 3 classical defenses scored highest for that specific combination) — the actually-relevant comparison. Seeds stayed at 5 (42–46) rather than the ≥10 the plan recommends, for session time-budget reasons — noted as not fully done, not silently skipped.
- **CNN scale (`exp1_cnn_scale.py`) reproduces MNIST cleanly but is ambiguous on CIFAR-10**: comparing FedAvg/FLTrust/Median/Trimmed-Mean under `sign_flipping` (20% Byzantine, 10 clients, 3 seeds) with `model_type='cnn'`: on MNIST, all 3 defenses beat FedAvg with huge, tight effect sizes (0.869→0.946-0.955, `p≈0.004`, `Cohen's d≈9`) — a much cleaner separation than `_LogisticModel` ever showed. On CIFAR-10, none of the 3 defenses is statistically distinguishable from FedAvg (`p` between 0.26 and 0.97), and two trend slightly *worse*. This could be a genuine finding (classical rules may not transfer as cleanly to CIFAR-10-scale CNN gradients) or an artifact of the reduced scale used for this pilot (15 rounds, 3 seeds, 2 local epochs — smaller than the logistic-regression tables' 20 rounds/5 seeds, chosen because a CNN's per-round cost via Keras is ~10-20x `_LogisticModel`'s hand-rolled numpy SGD). Not resolved — flagged for a longer run (more rounds/seeds) before citing the CIFAR-10 numbers as a stable result in the paper.
- **`DetectionCombiner`/`AnomalyCombinerNN` (`f_combine`) is instantiated but never called in the live pipeline — a text/implementation mismatch predating this session's root-size work, found via a direct audit of `docs/paper_A_esboco_v2.md` against the code**: `GRADFFederatedLearner.__init__` builds `self.combiner` (`src/fl/gradf_learner.py:61`) but no other line in the file references it; `RLAttackClassifier.classify()` receives the 4 raw z-scores directly (`input_shape=(4,)`, `src/classification/rl_classifier.py:27`), not `f_combine`'s fused scalar. `exp7_modality_ablation.py` (the "multi-modal fusion beats single signal" ablation) also never instantiates `DetectionCombiner` — it masks `RLAttackClassifier`'s own input columns. The fusion that ablation actually measures is the classifier's own hidden layers, not `f_combine`. Fixed in the paper text (`docs/paper_A_esboco_v2.md` §3.2/§3.3/§4.7/Apêndice A now describe the classifier receiving the raw 4-vector) — not fixed in code, since no reported number depended on `f_combine` being in the execution path in the first place.
- **FLTrust's root dataset size is not an inert configuration choice — it decides the headline result (found this session, drives the v2 paper framing)**: the root (`server_val`, used by `FLTrustStrategy.aggregate` and, per `GRADFFederatedLearner`, by `AccuracyDetector`/Layer 4 too) defaults to 3,000 MNIST samples (5% of the pool), 30× the ~100-sample root the original FLTrust paper (Cao et al.) uses. Re-running the dominance grid (`exp9_dominance_grid.py --root_size 100`, 10 seeds) collapses FLTrust's win rate from 18/21 to 8/21 cells, and from 5/7 to 0/7 under mild heterogeneity — where the anchored-vs-relational mechanism doesn't even predict an advantage, implying part of the original dominance was root size, not mechanism. `load_dataset_participants(..., root_size=N, root_seed=...)` subsamples the on-disk root without touching client partitions. Also raises an unresolved privacy question for the MIMIC-III path: a root big enough to matter means centralizing raw patient records, which the FL threat model exists to avoid — see `references/decisao_root_dataset_lgpd.md`. Full trail: `references/sensibilidade_root_fltrust.md`.
- **The Oracle-selector control (`exp1_baseline.py`) is gated by the same imperfect classifier as production, so it is not a valid upper bound for weakly-detected attacks**: `OracleSelector` picks, per classified attack index, whichever aggregation strategy scored highest reward during a short offline pretraining episode — but that choice is only ever *applied* when the real-time classifier agrees on the attack index. For `sign_flipping`/`label_flipping` (weak detection, 0.5–20%), the classifier predicts "no attack" most rounds, so the oracle silently applies `fedavg` instead of its own chosen `clustering` — and ends up scoring *below* `RandomSelector` and even plain FedAvg. Confirmed by instrumenting `HardeningPipeline.full_pipeline`'s `selected_strategy` with a ground-truth-attack-index stub bypassing the classifier: accuracy jumps from ≈0.79–0.81 to ≈0.89–0.90 (matching the strategy run standalone) once decoupled from detection. Not a bug in `_derive_oracle_map`/`OracleSelector.select_action` (both do exactly what they say) — a design gap in the ablation's oracle definition. This stub (`GroundTruthClassifierStub`/`GroundTruthOracleLearner`) now lives in `src/experiments/exp1_baseline.py` (CLI: `--ground_truth_oracle`), reproducing the "oráculo desacoplado" column of `docs/paper_A_esboco_v2.md` §5.3's Table 5 from versioned code rather than an ad-hoc script. See `references/sensibilidade_root_fltrust.md` §5, §12.
- **DP mechanism is opt-in, not a silent replacement**: `src/defense/dp_accountant.py::GaussianDPMechanism` (Gaussian noise, L2-clip-consistent sensitivity, real RDP-based (ε,δ) accounting over rounds) only takes effect when passed explicitly as `HardeningPipeline(dp_mechanism=...)`; the default (`dp_mechanism=None`) keeps the legacy per-coordinate Laplace mechanism described above, so this does not retroactively change any already-reported experiment numbers. See the module's docstring for the explicit trust model (protects whoever observes the released global model, not the server itself, since the server inspects updates in clear for detection before noise is added) and the per-aggregation-rule sensitivity table (tight for the 3 mean-type rules, a documented conservative bound for the 4 non-linear/robust rules, which don't have a simple closed-form tight sensitivity).

---

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requiriments.txt
```

Download and partition MNIST + CIFAR-10:

```bash
python data/download_datasets.py          # ~3 GB, ~5 min
python data/download_datasets.py --n_clients 10 --alpha 0.5 --seed 42
```

Verify setup:

```bash
python test_setup.py
pytest      # 95 tests — unit (detection/classification/defense) + integration (full GRADF pipeline)
```

Run the notebooks (all except 01 are executable today):

```bash
jupyter notebook notebooks/02_baseline_fedavg.ipynb        # FedAvg/FLTrust baseline
jupyter notebook notebooks/03_detection_analysis.ipynb     # detection modality analysis
jupyter notebook notebooks/04_rl_training.ipynb            # classifier + selector training
jupyter notebook notebooks/05_attack_simulation.ipynb      # GRADF vs FedAvg under attack
jupyter notebook notebooks/06_results_analysis.ipynb       # consolidates exp1-exp6 tables
jupyter notebook notebooks/07_fl_parameter_sweep.ipynb
jupyter notebook notebooks/08_fl_attack_simulation.ipynb
```

Run experiments (note: invoke as a module, not a script directly, so `src.*` imports resolve):

```bash
python -m src.experiments.exp1_baseline --seeds 42 43 44 --attack_types sign_flipping gaussian_noise
python -m src.experiments.exp2_robustness --seeds 42 43 44
python -m src.experiments.exp4_adaptive --seeds 42 43 44
python -m src.experiments.exp7_modality_ablation --seeds 42 43 44
python -m src.experiments.exp8_xai_examples   # run after exp2 — reads its saved audit trails
python -m src.experiments.runner   # runs exp1, exp2, exp3, exp4, exp6, exp7 in sequence (single seed unless --seeds is forwarded)
python -m src.experiments.exp5_clinical export   # produces a survey CSV for real clinicians

# Root-size sensitivity (current paper's central finding — see "Paper framing" above)
python -m src.experiments.exp9_dominance_grid --seeds 42 43 44 45 46 47 48 49 50 51        # full grid, canonical root (100)
python -m src.experiments.exp9_dominance_grid --root_size 100 --strategies fltrust --seeds 42 43 44 45 46 47 48 49 50 51   # only re-run the root-dependent strategy
python -m src.experiments.exp1_baseline --root_size 100 --seeds 42 43 44 45 46 47 48 49 50 51 --attack_types sign_flipping gaussian_noise label_flipping poisoning
python -m src.experiments.exp4_adaptive --root_size 100 --seeds 42 43 44 45 46 47 48 49 50 51
```

`exp1`/`exp2`/`exp4`/`exp7`/`exp9` each save both a `*_raw.csv` (one row per seed) and a `*_summary.csv` (mean/std/95% CI across seeds, plus a paired significance test where applicable) to `results/tables/`. `--root_size` on `exp1`/`exp4`/`exp9` suffixes output filenames with `_root{N}` so a sensitivity run never overwrites the canonical-root results.

---

## Architecture

### Core FL (`src/fl/`)

- `federated_learner.py` — `FederatedLearner` runs the FL training loop. `ParticipantData.from_npy()` loads per-client `.npy` shards and auto-flattens images. `_LogisticModel` (default, `model_type='logistic'`) stores all parameters as a flat vector `[W_flat, b]` so aggregation (cosine similarity, weighted mean) is consistent for both binary and K-class tasks. `_CNNModel` (`model_type='cnn', input_shape=(H,W,C)`) is a small Keras CNN implementing the SAME flat-vector interface (fit/set_params/params/accuracy) — Decision A4, see "Known simplifications" — so every downstream file (`attacked_learner.py`, `gradf_learner.py`, `hardening.py`, all aggregation strategies) works with either model unmodified; `_make_model` is the only dispatch point. Two built-in strategies: `FedAvgStrategy`, `FLTrustStrategy`.
- `attacked_learner.py` — `AttackedFederatedLearner(FederatedLearner)` injects attacks per round (gradient-level via `poison_update`, data-level via `poison_data`). Exposes `_compute_param_updates()` as a reusable hook.
- `gradf_learner.py` — `GRADFFederatedLearner(AttackedFederatedLearner)` overrides `_run_round` to run the full GRADF pipeline: 4 detectors → `RLAttackClassifier` → `RLDefenseSelector` → `HardeningPipeline.full_pipeline()` → apply aggregate → log XAI explanation, for every client every round.

### Attack Simulator (`src/classification/attack_simulator.py`)

`AttackSimulator` provides two attack surfaces (`poison_update` for gradient-level, `poison_data` for data-level) across 11 GRADF types + TARS aliases (`sign_flipping`, `gaussian_noise`) + `label_flipping`. `PretenseAttacker` wraps any attack to stay honest for N rounds before attacking.

### Multi-Modal Detection (`src/detection/`)

- `gradient_analyzer.py` — Modality 1: magnitude, windowed z-score, Jensen-Shannon divergence.
- `accuracy_detector.py` — Modality 2: accuracy degradation vs. a validation set, windowed z-score. Decoupled from `src.fl` — the caller supplies `old_accuracy`/`new_accuracy`.
- `temporal_detector.py` — Modality 4: consistency vs. the client's own previous update.
- `clustering_detector.py` — Modality 3: cosine distance to peer updates in the same round.
- `combiner_nn.py` — `AnomalyCombinerNN` (Keras) + `DetectionCombiner` wrapper combining the 4 scores into a binary anomaly probability.
- `modality_recorder.py` — `ModalityRecorder` bundles all 4 detectors + `ATTACK_LABELS`. Lives in `src.detection` (not `src.utils`) specifically to avoid a circular import with `src.fl` — see the module docstring.

### Defense (`src/defense/`)

- `aggregation_methods.py` — `MedianStrategy`, `TrimmedMeanStrategy`, `FedProxStrategy`, `KrumStrategy`, `ClusteringStrategy`, registered into `federated_learner._STRATEGIES` via `register_strategy()`.
- `rl_selector.py` — `RLDefenseSelector`, a DQN over a **5-dim state** `[attack_type_idx, confidence, prev_accuracy, prev_fairness_std, prev_latency]` (the richer state the formalism always specified, but that the code didn't implement until the post-Gate-1 RL rework — see "Known simplifications") → Q-values for 7 strategies. Has a target network (synced every `target_update_every` steps) and a replay buffer (`train_step` samples a mini-batch, not just the single passed-in experience) — both previously absent. Trains via `train_step`/`train_on_experiences` on reward signal from `src.utils.data_loader.generate_selector_experiences`/`generate_rotating_selector_experiences` (offline pretraining) **and** online, every real FL round, via `HardeningPipeline.full_pipeline` → `GRADFFederatedLearner._update_selector_online` — previously the network was trained once offline and frozen for the rest of the run. `train_on_experiences(experiences, epochs=...)` defaults to `epochs=1` (kept for backward compatibility) but production callers (`exp1_baseline.py`, `exp4_adaptive.py`) pass `epochs=200` — a single pass was found to leave the DQN undertrained (converges to the wrong action for `label_flipping` specifically) — see "Known simplifications".
- `tars_selector.py` — `TARSSelector`/`TrustScorer`: a from-scratch reproduction of TARS (Ahmed et al., 2025) used as a competing baseline in `exp4_adaptive.py`'s Gate 1 comparison, not part of the GRADF pipeline itself. Trust score (loss divergence + cosine similarity + magnitude deviation) → tabular ε-greedy Q-learning over the same aggregation strategies. Two pieces the original paper doesn't specify in closed form (the trust-scoring combination function, and the state discretization into Q-table bins) are our own documented instantiation, not the authors' code.
- `adabfl_aggregator.py` — `AdaBFLAggregator`: a from-scratch reproduction of AdaBFL (Tang, Liu & Huang, 2026), a second competing baseline in `exp4_adaptive.py`'s Gate 1 comparison. Unlike TARS/FedStrategist (which select among existing aggregation rules), AdaBFL fuses its own outputs every round: filter malicious clients (a magnitude/direction consistency check) → trimmed-mean → a "derivative model" synthesized from the most centrally-located benign client, combined via three self-tuning weights β1/β2/β3 updated from two discrepancy signals p1/p2. The paper's PDF (checked in full, including references/appendix) links no code repo. Three genuine ambiguities/inconsistencies in the paper (the filter's free parameters, the p2 discrepancy formula's likely typo, and a contradiction between Algorithm 2's pseudocode and its own body text on the β3 update condition) are resolved and documented in the module docstring.
- `fedstrategist_selector.py` — `DiagnosticStateVector`/`LinUCBAgent`: a from-scratch reproduction of FedStrategist (Haque, Kamal & Hossain, 2025), a third competing baseline in `exp4_adaptive.py`'s Gate 1 comparison. A 3-dim diagnostic state (variance of update norms, average pairwise cosine similarity, mean update norm) feeds a LinUCB contextual bandit over `{fedavg, median, krum}`, rewarded by accuracy gain minus a per-rule cost (`C_fedavg=0.1, C_median=0.4, C_krum=0.8`, from the paper's own Appendix). The paper links a public repo, but it's a standalone PyTorch harness built around a different model representation than this codebase's flat-numpy-vector models — reproduced here from the paper's closed-form spec instead of ported.
- `dp_accountant.py` — `GaussianDPMechanism`: opt-in replacement for Layer 2's legacy per-coordinate Laplace noise. Real (ε,δ) accounting via the `dp_accounting` library's RDP accountant (not hand-rolled composition), L2 clip + Gaussian noise added ONCE to the aggregated output (not per-client before aggregation), with per-aggregation-rule sensitivity (tight for `fedavg`/`fedprox`/`fltrust`, a documented conservative bound for `median`/`trimmed_mean`/`krum`/`clustering`). Passed via `HardeningPipeline(dp_mechanism=...)`; default (`None`) keeps the legacy Laplace mechanism so existing experiment numbers are unaffected. See its docstring for the explicit trust model (model-release DP, not a defense against the server itself).
- `hardening.py` — `HardeningPipeline`: Layer 1 (magnitude/skewness validation) → Layer 2 (DP clipping + Laplace noise by default, or clip-only + `dp_mechanism.privatize_aggregate` after Layer 3 if a `GaussianDPMechanism` is passed) → Layer 3 (classifier + selector pick the aggregation strategy, majority vote across accepted hospitals) → Layer 4 (accuracy-regression check via a caller-supplied `evaluate_fn`) → Layer 5 (HE stub, not evaluated in the paper — see "Publication scope").

### XAI (`src/xai/`)

- `explanation_generator.py` — `XAIExplainer` generates a narrative + a *computed* counterfactual (from real old/new accuracy, not hardcoded text) for every ACCEPT/REJECT decision.
- `audit_trail.py` — `AuditTrail` persists explanations as JSON Lines (`.add/.load/.query/.clear`).

### Utilities (`src/utils/`)

- `data_loader.py` — `generate_detection_dataset()` and `generate_selector_experiences()` run real (small-scale) FL rounds with injected attacks to produce labeled training data for the classifier/combiner/selector, replacing the original `np.random` placeholders. `load_dataset_participants()` loads MNIST/CIFAR-10 shards.
- `hospital_splitter.py` — MIMIC-III non-IID hospital partitions (blocked on data).
- `config_loader.py`, `metrics.py`, `logger.py`, `visualization.py` — shared config loading, evaluation metrics, logging, and plotting helpers.

---

## GRADF End-to-End Pipeline

```
GRADFFederatedLearner._run_round(), per client:
    local training (+ attack injection if configured)
        ↓
    GradientAnalyzer + AccuracyDetector + TemporalDetector + ClusteringDetector
        ↓
    RLAttackClassifier  →  attack type + confidence
        ↓
    RLDefenseSelector   →  aggregation strategy vote
        ↓
    HardeningPipeline.full_pipeline()  →  validate, sanitize, aggregate (majority-voted strategy), validate accuracy
        ↓
    XAIExplainer.generate_explanation()  →  AuditTrail entry (JSON Lines)
```

This is exercised end-to-end by `tests/test_integration.py` and by `notebooks/05_attack_simulation.ipynb`.

---

## What's Left

For the paper as currently framed (`docs/paper_A_esboco_v2.md` — see "Paper framing" above), in priority order:

1. ~~**Regenerate Table 1**~~ Done: recomputed under `--root_size 100`, 10 seeds (42–51) — `results/tables/exp1_baseline_table1_root100_raw.csv`/`_summary.csv`. FLTrust collapses (0.899→0.820, matching §5.2's grid finding); GRADF now beats FLTrust with significance (p≈2.1e-4) but that's the FLTrust collapse, not GRADF improving — it remains statistically tied with Median (p=0.36) and Trimmed-Mean (p=0.85), so the v1 conclusion ("GRADF doesn't beat the strong fixed rules") survives, just with a different strong rule. `docs/paper_A_esboco_v2.md` §5.1 updated with the real table (previously just a placeholder note). The stale `results/tables/exp1_baseline_raw.csv` (1-seed smoke test) is superseded by this file for Table 1's purposes but left on disk unchanged.
2. **Verify Table 9's detection-rate column under root=100**: the 0.5–4%/14–20%/100%/100% detection rates cited for `sign_flipping`/`label_flipping`/`gaussian_noise`/`poisoning` come from `exp2_robustness` under the OLD root=3,000 setup, never re-measured under root=100. This matters because `AccuracyDetector` uses `root_data` as its validation set — the same variable this whole investigation shows is not safe to assume is inert.
3. **MIMIC-III root-dataset architecture** (blocks re-using this finding on real clinical data, not just the conference submission): see `references/decisao_root_dataset_lgpd.md` for the three candidate designs (public/synthetic root, delegated computation without centralizing records, DP on the root update) — none implemented.
4. **Hyperparameter tuning**: `dp_epsilon`/`dp_clipping`/`magnitude_threshold`/`accuracy_tolerance` in `config/config.yaml` were calibrated for MNIST-scale models (~7,850 parameters) — revisit if adding CIFAR-10 results to the paper (larger model).
5. **Extend the root-size finding beyond `exp9`/`exp1`/`exp4`**: `exp2_robustness` still runs under the old root=3,000 default and hasn't been re-checked.

Deferred to the journal extension (not blocking the conference submission):

4. **MIMIC-III**: obtain PhysioNet credentials, run `data/download_mimic.py`, then `data/merge_mimic_tables.py`, then validate `src/utils/hospital_splitter.py` and execute `notebooks/01_eda_mimic.ipynb`.
5. **Clinical validation**: recruit 15 clinicians, have them fill in the CSV from `python -m src.experiments.exp5_clinical export`, then run `analyze` on the responses.
6. **`exp3_scalability`/`exp6_fairness`**: already implemented and runnable, just not part of the paper's core result set.

---

## Dataset Setup

### MNIST + CIFAR-10 (available now)

```
data/
  raw/
    mnist/        X_train.npy  y_train.npy  X_test.npy  y_test.npy
    cifar10/      (same)
  processed/
    mnist/
      iid/        server_val/  client_0/ … client_9/
      non_iid/    (same)
    cifar10/
      iid/
      non_iid/
```

Non-IID heterogeneity coefficient: MNIST ≈ 0.65, CIFAR-10 ≈ 0.67 (Dirichlet α=0.5).

### MIMIC-III (requires credentials)

1. Edit `data/download_mimic.py` with your PhysioNet email, run it (prompts for password).
2. Run `python data/merge_mimic_tables.py` to produce `data/raw/mimic/merged_data.csv`.
3. Run `python -m src.utils.hospital_splitter` to produce `data/processed/hospital_{a-e}/`.

Note: MIMIC-III has no native `specialty` column; `merge_mimic_tables.py` derives it heuristically from `ADMISSION_LOCATION` — see its docstring before using for anything beyond this framework's illustrative hospital split.

---

## Testing

```bash
pytest                          # all 95 tests
pytest src/detection/tests/     # detection modalities + combiner
pytest src/classification/tests/  # attack classifier (unit + real-data integration)
pytest src/defense/tests/       # aggregation strategies + defense selector
pytest tests/test_integration.py  # full GRADFFederatedLearner pipeline, audit trail
pytest tests/test_utils.py      # config, metrics, logging, plotting
```

`tests/fixtures/mock_data.py` provides synthetic `ParticipantData` (no real MNIST files needed) shared across the classification/defense/integration test suites.

---

## Key Design Notes

- **Flat parameter vector**: `_LogisticModel` stores all parameters as `[W_flat (n_features × K), b (K,)]` so FLTrust cosine similarity and all aggregation methods work on a consistent 1-D array.
- **Dual framework**: TensorFlow/Keras for the DQN/NN models (`RLAttackClassifier`, `RLDefenseSelector`, `DetectionCombiner`); NumPy for the FL task model (`_LogisticModel`). Both must be installed.
- **No module-level execution**: every `src/` module is safe to import — training/demo code lives behind `if __name__ == '__main__':` guards.
- **Import cycle avoidance**: `ModalityRecorder`/`ATTACK_LABELS` live in `src.detection.modality_recorder` (not `src.utils.data_loader`, where they originated) specifically because `src.fl.gradf_learner` needs them and `src.utils.data_loader` needs `src.fl` — putting them in `src.utils` would create a cycle. If you add new cross-cutting helpers, check both import directions before picking a home module.
- **Run experiments as modules**: `python -m src.experiments.exp1_baseline`, not `python src/experiments/exp1_baseline.py` — the latter doesn't put the repo root on `sys.path` and `src.*` imports fail.
- **Results directory**: `results/models/`, `results/tables/`, `results/figures/`, `results/audit_trail.jsonl` are created automatically by whichever script/notebook needs them.
