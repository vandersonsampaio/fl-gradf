# GRADF — Generic RL-based Adaptive Detection & Defense Framework

## Overview

**GRADF** (**G**eneric **R**L-based **A**daptive **D**etection & **D**e**F**ense Framework) is a research-grade framework for studying adaptive defense against Byzantine clients in Federated Learning (FL). It is domain-agnostic: it operates on flat model-update vectors and is validated on standard non-IID benchmarks (MNIST/CIFAR-10). Per FL round, GRADF combines:

1. Multi-modal attack detection (gradient, accuracy, temporal, clustering signals)
2. RL-based attack classification (11+ attack types)
3. RL-based defense selection (7 aggregation strategies)
4. 5-layer hardening pipeline
5. XAI audit trail for every aggregation decision

The framework also reproduces the **TARS** experimental setup (Ahmed et al., 2025): N=10 clients, f=2 Byzantine (20%), MNIST and CIFAR-10 with non-IID Dirichlet(α=0.5) partitioning. Besides TARS, it contains from-spec reproductions of three more competing adaptive defenses from the FL security literature: **AdaBFL** (Tang, Liu & Huang, 2026 — adaptive multi-layer aggregation, no linked code repo), **FedStrategist** (Haque, Kamal & Hossain, 2025 — LinUCB contextual bandit over the aggregation rule; its public repo targets a different model representation than this codebase, so it's a from-spec reproduction, not a port) and **AdaAggRL** (Wang et al., AAAI 2025 — gradient inversion + MMD cues + TD3 over continuous per-client weights). See `src/defense/{tars_selector,adabfl_aggregator,fedstrategist_selector,adaaggrl_agent}.py`.

The full detection → classification → defense-selection → hardening → XAI pipeline is implemented and orchestrated by `GRADFFederatedLearner`, and runs end-to-end on MNIST/CIFAR-10 (already downloaded and partitioned).

---

## Versions

| Version | Description |
|---------|-------------|
| **p1.0.0** | Code state that produced the paper **"Rule Selection Fails Where Client Weighting Succeeds: A Characterization of Adaptive Defense in Byzantine-Robust Federated Learning"**. Use this version to reproduce the paper's results. |

### Paper p1.0.0 — summary

The paper is a **characterization study**, not a new defense method. In it, GRADF is used as a probe for discrete rule selection. Its findings:

1. **Root dataset size governs fixed-rule dominance.** FLTrust's apparent dominance over the other robust rules is an artifact of an oversized root dataset (3,000 MNIST samples). At the canonical size used by the original FLTrust paper (~100 samples; Cao et al., NDSS 2022), its win rate collapses from 18/21 to 8/21 cells of the dominance grid, and from 5/7 to 0/7 under mild heterogeneity (α=0.5). A root that large also centralizes exactly the data FL exists to keep local.
2. **Under a canonical root, headroom for adaptive defense exists and is capturable in principle.** No fixed rule dominates across attack × heterogeneity conditions, and an oracle selector decoupled from detection captures the margin.
3. **Capturing it depends on the class of adaptive mechanism, not on adaptation itself.** Across 10 seeds with paired significance testing, two independent implementations of discrete rule selection (GRADF's DQN selector and FedStrategist) fail to beat random selection, while continuous per-client weighting (AdaAggRL) captures most of the available headroom, with large effect sizes.
4. **Explanatory and methodological framework.** The distinction between reference-anchored (FLTrust) and purely relational (Krum, Median, …) defenses explains why heterogeneity degrades some rules and not others. Evaluating adaptive defense needs, at minimum, a canonical root, strong baselines, multiple seeds, and floor (random) and ceiling (oracle) controls.

Scope: MNIST, the benchmark FLTrust was originally proposed on. The claims about discrete selection are empirical and limited to the instances and regime evaluated.

Experiments behind the paper: `exp9_dominance_grid` (root-size sensitivity / headroom), `exp10_selector_comparison` (GRADF vs. FedStrategist vs. AdaAggRL vs. Random/Oracle, root=100), plus `exp1_baseline` and `exp4_adaptive` under `--root_size 100`. See "Quick Start" for the commands.

### Evolution of the framing

The central claim was rewritten as evaluation rigor increased. **v1**: under the 3,000-sample root, FLTrust dominated every fixed-vs-adaptive comparison, so the conclusion was that adaptive selection doesn't pay off. **v2**: the root size turned out to be the confound. With the canonical root (100 samples, 10 seeds), headroom reappears, but the deployed selectors still fail to capture it on weakly-detected attacks (`sign_flipping`, `label_flipping`). **p1.0.0** adds the continuous-weighting family (AdaAggRL, `exp10`) and shows that the mechanism class, discrete selection vs. continuous weighting, decides who captures the headroom.

### Core vs. auxiliary experiments

The central experiments are `exp9_dominance_grid`, `exp10_selector_comparison`, `exp1_baseline` (system comparison **and** per-component ablation), `exp2_robustness` and `exp4_adaptive` (adaptive adversary). All run over multiple seeds with mean/std/95% CI and paired significance testing (`src/utils/stats.py`). `exp7_modality_ablation` measures multi-modal detection against single-modality detection, and `exp8_xai_examples` gives a qualitative XAI illustration. `exp3_scalability` and `exp6_fairness` stay implemented and runnable but are not part of the core result set.

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
| Competing adaptive defenses | `src/defense/{tars_selector,adabfl_aggregator,fedstrategist_selector,adaaggrl_agent}.py` | From-spec reproductions of TARS, AdaBFL, FedStrategist and AdaAggRL, with documented deviations |
| Aggregation strategies | `src/defense/aggregation_methods.py` + `federated_learner.py` | All 7: FedAvg, FLTrust, Median, Trimmed-Mean, FedProx (simplified, see notes), Krum, Clustering |
| Hardening pipeline | `src/defense/hardening.py` | 5 layers, actually dispatches to classifier → selector → chosen aggregation strategy |
| XAI + audit trail | `src/xai/explanation_generator.py`, `src/xai/audit_trail.py` | Structured explanations with computed (not hardcoded) counterfactuals, persisted as JSON Lines |
| Data generation for training | `src/utils/data_loader.py` | `generate_detection_dataset`, `generate_selector_experiences`, `load_dataset_participants` — real FL rounds, not `np.random` |
| Dataset pipeline | `data/download_datasets.py` | MNIST + CIFAR-10 download; IID + Dirichlet non-IID partitioning into 10 client shards |
| Config | `config/config.yaml`, `config/hyperparameters.yaml` + `src/utils/config_loader.py` | FL/GRADF parameters and network hyperparameters |
| Shared utils | `src/utils/{metrics,logger,visualization}.py` | Detection rate, FPR, fairness std; central logging; shared plotting |
| Experiments 1–4, 6, 7, 9, 10 | `src/experiments/exp{1,2,3,4,6,7,9,10}_*.py` + `runner.py` | Real `FederatedLearner`/`GRADFFederatedLearner` calls, CLI-driven, save to `results/`. `exp1`/`exp2`/`exp4`/`exp7`/`exp9`/`exp10` support `--seeds` (multiple values) and report mean/std/95% CI via `src/utils/stats.py`; `exp1` also reports a paired t-test of each system vs. FedAvg. `exp1`/`exp2` also report wall-clock time (`wall_clock_seconds`, and `pretrain_seconds` for the classifier/selector pretraining shared by RL-only/Hardening-only/GRADF) — the "cost of defense" numbers, kept separate from the `n_clients`-scaling analysis in `exp3`. `exp1`/`exp4`/`exp9`/`exp10` also take `--root_size N` (subsamples the FLTrust root dataset — see "Paper p1.0.0" above); `exp9` additionally takes `--strategies` to re-run only a subset of the 4 fixed defenses (only `fltrust` actually depends on `--root_size`); `exp10` takes `--variant a|b` (FedStrategist with its own detection vs. GRADF's shared detection) |
| XAI qualitative examples | `src/experiments/exp8_xai_examples.py` | Pulls 1 real ACCEPT + 1 REJECT example (computed narrative/counterfactual) and a decision-rate table from `exp2`'s saved audit trails — run `exp2` first |
| Notebooks 02–08 | `notebooks/02_*.ipynb` → `notebooks/08_*.ipynb` | All executed, real outputs saved (MNIST-based) |
| Test suite | `src/**/tests/`, `tests/` | 95 tests: unit + integration (pytest, see below) |

### Known simplifications (documented in code)

- **`RLAttackClassifier` is not literal RL** — it's a supervised softmax classifier trained on real detection signals (ground-truth attack labels come from the simulator during training). `RLDefenseSelector` **is** genuine RL: it trains on rewards observed from real FL rounds. See the design note in `src/classification/rl_classifier.py`.
- **`FedProxStrategy`** approximates the original FedProx (Li et al., MLSys 2020) as a damped weighted mean at the server, since the real algorithm modifies the *local* training objective — not observable from already-computed client deltas at the aggregation-only interface used here. See `src/defense/aggregation_methods.py`.
- **DP noise calibration is dimensionality-sensitive**: Layer 2's Laplace noise is added per-coordinate, so its total L2 norm scales with `sqrt(n_parameters)`. The default `dp_epsilon=50.0` is calibrated for MNIST-scale models (~7850 parameters); a much smaller model needs a smaller epsilon, and a much larger model needs a larger one, or the noise will swamp the signal (this was found and fixed during Fase 6 — GRADF was underperforming plain FedAvg with `dp_epsilon=1.0` on MNIST before recalibration). See the docstring in `src/defense/hardening.py`.
- **Audit trail must be isolated per run**: `XAIExplainer()`/`AuditTrail()` default to the same shared path (`results/audit_trail.jsonl`) across every instance. `exp2_robustness.py` originally created a fresh `GRADFFederatedLearner()` per `(attack_type, byzantine_fraction)` combination and read `.audit_trail.load()` right after — which silently accumulated records from every earlier combination in the same loop and inflated `detection_rate`. Fixed by giving each run its own `AuditTrail(path=f"results/audit_trail/exp2_{attack_type}_{frac}_{seed}.jsonl")`, cleared before training. If you add a new multi-run script that reads the audit trail mid-loop, apply the same pattern.
- **Modality ablation via masking, not architecture change** (`exp7_modality_ablation.py`): `RLAttackClassifier`'s network has a fixed 4-feature input. "Single-modality" classifiers are simulated by zeroing the other 3 columns rather than retraining a smaller network — keeps the comparison architecture-controlled, at the cost of not testing whether a smaller network would do better with fewer inputs (not the question this ablation asks).
- **CNN scale (`exp1_cnn_scale.py`) reproduces MNIST cleanly but is ambiguous on CIFAR-10**: comparing FedAvg/FLTrust/Median/Trimmed-Mean under `sign_flipping` (20% Byzantine, 10 clients, 3 seeds) with `model_type='cnn'`: on MNIST, all 3 defenses beat FedAvg with huge, tight effect sizes (0.869→0.946-0.955, `p≈0.004`, `Cohen's d≈9`) — a much cleaner separation than `_LogisticModel` ever showed. On CIFAR-10, none of the 3 defenses is statistically distinguishable from FedAvg (`p` between 0.26 and 0.97), and two trend slightly *worse*. This could be a genuine finding (classical rules may not transfer as cleanly to CIFAR-10-scale CNN gradients) or an artifact of the reduced scale used for this pilot (15 rounds, 3 seeds, 2 local epochs — smaller than the logistic-regression tables' 20 rounds/5 seeds, chosen because a CNN's per-round cost via Keras is ~10-20x `_LogisticModel`'s hand-rolled numpy SGD). Not resolved — flagged for a longer run (more rounds/seeds) before citing the CIFAR-10 numbers as a stable result.
- **FLTrust's root dataset size is not an inert configuration choice — it decides the headline result**: the root (`server_val`, used by `FLTrustStrategy.aggregate` and, per `GRADFFederatedLearner`, by `AccuracyDetector`/Layer 4 too) defaults to 3,000 MNIST samples (5% of the pool), 30× the ~100-sample root the original FLTrust paper (Cao et al.) uses. Re-running the dominance grid (`exp9_dominance_grid.py --root_size 100`, 10 seeds) collapses FLTrust's win rate from 18/21 to 8/21 cells, and from 5/7 to 0/7 under mild heterogeneity — where the anchored-vs-relational mechanism doesn't even predict an advantage, implying part of the original dominance was root size, not mechanism. `load_dataset_participants(..., root_size=N, root_seed=...)` subsamples the on-disk root without touching client partitions. Also raises an unresolved privacy question: a root big enough to matter means centralizing raw client data on the server, which the FL threat model exists to avoid.
- **The Oracle-selector control (`exp1_baseline.py`) is gated by the same imperfect classifier as production, so it is not a valid upper bound for weakly-detected attacks**: `OracleSelector` picks, per classified attack index, whichever aggregation strategy scored highest reward during a short offline pretraining episode — but that choice is only ever *applied* when the real-time classifier agrees on the attack index. For `sign_flipping`/`label_flipping` (weak detection, 0.5–20%), the classifier predicts "no attack" most rounds, so the oracle silently applies `fedavg` instead of its own chosen `clustering` — and ends up scoring *below* `RandomSelector` and even plain FedAvg. Confirmed by instrumenting `HardeningPipeline.full_pipeline`'s `selected_strategy` with a ground-truth-attack-index stub bypassing the classifier: accuracy jumps from ≈0.79–0.81 to ≈0.89–0.90 (matching the strategy run standalone) once decoupled from detection. Not a bug in `_derive_oracle_map`/`OracleSelector.select_action` (both do exactly what they say) — a design gap in the ablation's oracle definition. This stub (`GroundTruthClassifierStub`/`GroundTruthOracleLearner`) now lives in `src/experiments/exp1_baseline.py` (CLI: `--ground_truth_oracle`) and is reused by `exp10` as the decoupled-oracle ceiling.
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

Run the notebooks:

```bash
jupyter notebook notebooks/02_baseline_fedavg.ipynb        # FedAvg/FLTrust baseline
jupyter notebook notebooks/03_detection_analysis.ipynb     # detection modality analysis
jupyter notebook notebooks/04_rl_training.ipynb            # classifier + selector training
jupyter notebook notebooks/05_attack_simulation.ipynb      # GRADF vs FedAvg under attack
jupyter notebook notebooks/06_results_analysis.ipynb       # consolidates experiment result tables
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

# Paper p1.0.0 — root-size sensitivity and adaptive-mechanism comparison (canonical root = 100)
python -m src.experiments.exp9_dominance_grid --seeds 42 43 44 45 46 47 48 49 50 51        # full grid
python -m src.experiments.exp9_dominance_grid --root_size 100 --strategies fltrust --seeds 42 43 44 45 46 47 48 49 50 51   # only re-run the root-dependent strategy
python -m src.experiments.exp10_selector_comparison --variant b   # GRADF vs FedStrategist (shared detection) vs AdaAggRL vs Random/Oracle, seeds 42–51
python -m src.experiments.exp10_selector_comparison --variant a   # FedStrategist with its own detection
python -m src.experiments.exp1_baseline --root_size 100 --seeds 42 43 44 45 46 47 48 49 50 51 --attack_types sign_flipping gaussian_noise label_flipping poisoning
python -m src.experiments.exp4_adaptive --root_size 100 --seeds 42 43 44 45 46 47 48 49 50 51
```

`exp1`/`exp2`/`exp4`/`exp7`/`exp9`/`exp10` each save both a `*_raw.csv` (one row per seed) and a `*_summary.csv` (mean/std/95% CI across seeds, plus a paired significance test where applicable) to `results/tables/`. `--root_size` on `exp1`/`exp4`/`exp9` suffixes output filenames with `_root{N}` so a sensitivity run never overwrites the canonical-root results.

---

## Post-p1.0.0: AdaAggRL audit experiments

Follow-up work toward a second paper: does AdaAggRL's RL component (TD3) actually contribute, or is the headroom captured by its fixed skeleton (per-client signal + threshold + memory)? These experiments are **pre-registered** (a `PREREGISTRO.md` or `PLANO.md` plus a `.sha256` file with the hashes of the plan and of the code, frozen before any run) and live outside `src/` so the p1.0.0 code stays untouched.

### Directory layout

| Path | Contents | Versioned |
|------|----------|-----------|
| `scripts/frente1_ablacao_adaaggrl.py` | Ablation of our own AdaAggRL reproduction (`fixed`, `sr_only`, `cosmed_only`, `cosserver_only` vs. `td3_ref`); its learner is reused by B2.6, A0 and B2.7 | yes |
| `scripts/frente1_v0_diagnostico.py` | V0 cue sanity check | not yet (untracked) |
| `results/frente1_ablacao_adaaggrl/` | Ablation grid in our framework (seeds 42–51, 21 cells): `PREREGISTRO.md`, `RESULTADO.md`, raw/aggregated CSVs | not yet (untracked) |
| `external/AdaAggRL/` | Clone of the **official** AdaAggRL code (`github.com/yjEugenia/AdaAggRL`, commit `27b9c18`), never edited | no (see `external/.gitignore`) |
| `external/.venv_adaaggrl/` | Isolated venv with the official pinned versions (torch 2.3.0, SB3 2.3.2, numpy 1.26.4, gym 0.26.2, gymnasium 0.29.1) | no |
| `scripts/passo2_oficial/` | Runners, grid launchers and pre-written analyses for the official code (below) | yes |
| `scripts/passo2_oficial/shim/inversefed/` | Constants from Geiping's `invertinggradients` that the official code imports but doesn't ship | yes |
| `results/frente1_passo2_oficial/` | **Passo 2**: official code, 500 rounds, TD3 vs. fixed vs. random action, seeds 100–104 | yes (no logs) |
| `results/b21_replicacao_oficial/` | **B2.1 + B2.2**: confirmatory replication (fixed vs. TD3, seeds 105–114) + mechanistic hypotheses | yes (no logs; actor checkpoints only at steps 0 and 500) |
| `results/b23_steelman_oficial/` | **B2.3**: TD3 "steelman" (lr 1e-3, `learning_starts` 10), seeds 100–104, paired with Passo 2's runs; `EXECUCAO.md` logs a scheduling change | yes (no logs; actor checkpoints only at steps 0 and 500) |
| `scripts/c0_espaco_restante.py`, `scripts/c0_teto_corrigido.py`, `scripts/c0_teto_oraculo.py` | **C.0**: remaining headroom per cell in our framework (CPU only, project venv), against three ceilings: FedAvg without attack, best of 5 rules without attack, FedAvg over the 8 honest clients only (oracle) | yes |
| `results/c0_espaco_restante/` | C.0 outputs (seeds 42–51, 21 cells): `NOTAS.md` (interpretation, criterion frozen before the oracle run in §5, gate decision in §6) + `NOTAS.sha256`, one `analise*.txt` / `espaco_restante*.csv` pair per ceiling, raw ceiling runs | yes |
| `scripts/b28_oraculo_por_regra.py` | **B2.8**: greedy per-round oracle over the 7-rule arsenal (our framework, CPU only; `run` and `analisar` subcommands) | yes |
| `results/b28_oraculo_por_regra/` | B2.8 outputs (seeds 42–51, 21 cells, 210 runs): `PLANO.md` + `PLANO.sha256`, `raw/seed*.csv` (final accuracy and the rule chosen at every round), `oraculo_por_celula.csv`, `analise.txt`, `RESULTADO.md` | yes (no logs) |
| `results/b23b_steelman_normalizado/` | **B2.3b**: last TD3 steelman on the official code (reward normalized with SB3's `VecNormalize`, lr 1e-4, `learning_starts` 10), seeds 100–104, paired with Passo 2 and B2.3 | yes (no logs; actor checkpoints only at steps 0 and 500) |
| `scripts/b26_decomposicao.py`, `scripts/run_grid_b26.sh` | **B2.6**: confirmatory decomposition of the skeleton in our framework (CPU only): memory on/off and official-strength memory, binary vs. soft weighting, `cos_server` vs. S_R, combined signal and threshold sensitivity (`run` and `analisar` subcommands) | yes |
| `results/b26_decomposicao/` | B2.6 outputs (8 variants × seeds 52–61 × 21 cells = 1,680 runs): `PREREGISTRO.md` + `PREREGISTRO.sha256`, `raw/<variant>_seed*.csv`, `por_semente.csv`, `analise.txt`, `RESULTADO.md` | yes (no logs) |
| `scripts/a0_analises.py`, `scripts/passo2_oficial/a0_regimes_td3.py` | **A0**: cheap analyses (roadmap v4 §4.2). `a0_analises.py massa` re-runs B2.6 variants logging per-client weights; `analisar` computes attacker weight mass, the signal-selection ceiling and signal agreement. `a0_regimes_td3.py` (official venv) builds the TD3 drift / state-dependence table for the three regimes | yes |
| `results/a0_analises/` | A0 outputs: `PLANO.md` + `PLANO.sha256`, `massa/*.csv` (48 runs: `sr_only`/`sr_bin` × 8 cells × seeds 52–54), `analise.txt`, `concordancia_sinais.csv`, `analise_regimes_td3.txt`, `regimes_td3_por_checkpoint.csv`, `RESULTADO.md` | yes (no logs) |
| `scripts/b27_horizonte.py`, `scripts/run_grid_b27.sh` | **B2.7**: horizon 15/50/150 in our framework (CPU only): TD3, LinUCB (FedStrategist b) and DQN (GRADF v1) against fixed versions in the same action space, random arms and ceilings (`celula`, `teto`, `verificar` and `analisar` subcommands) | yes |
| `results/b27_horizonte/` | B2.7 outputs (8 cells × seeds 42–51 × 20 systems × 150 rounds, nested horizons, + 30 ceiling jobs): `PLANO.md` + `PLANO.sha256`, `raw/celula_*.csv`, `raw/teto_*.csv`, `grade_raw.csv`, `tetos_raw.csv`, `delta_por_celula.csv`, `analise.txt`, `verificacao.txt`, `RESULTADO.md` | yes (no logs) |
| `results/b25_ipm_oficial/` | **B2.5**: real IPM on the official code, TD3 vs. fixed, seeds 105–114, 500 rounds. `PLANO.md` + `.sha256`, `teste_unitario.txt`, `sanity/` + `sanity.txt` (ε choice on seed 100), `ADENDO1.md` (ε = 2), `raw/` (20 runs), `resumo_runs.csv`, `mecanismo.csv`, `analise.txt`, `RESULTADO.md` | yes (no logs; actor checkpoints only at steps 0 and 500) |

Every `results/<experiment>/` folder follows the same pattern: `PREREGISTRO.md` (or `PLANO.md`) + `.sha256`, `grid.log` (START/DONE/FAIL/GRID_END per run), `raw/` (one JSON per run with per-round accuracy, loss, executed action, weight mass on real attackers and resets; `*.partial.json` checkpoints every 25 rounds while running), then `analise.txt`, `resumo_runs.csv` and `RESULTADO.md` after the analysis. B2.1/B2.3 additionally save the states observed by the policy (`raw/obs/*.npy`, shape `(steps, 10, 4)`) and TD3 actor checkpoints every 50 steps (`raw/actors/*_stepNNN.pt`).

### Scripts (`scripts/passo2_oficial/`)

| Script | Role |
|--------|------|
| `run_oficial.py` | Runs the official environment unmodified through a Gymnasium adapter; conditions `td3` (exactly as in the official `main.py`), `fixed` (`[0.475]*5`, the center of the action box), `random` |
| `run_b21.py` | Reuses `run_oficial.py` and adds logging of observed states and actor checkpoints |
| `run_b23.py` | Reuses `run_b21.py`; changes only the TD3 `learning_rate` and `learning_starts` |
| `run_b23b.py` | Reuses `run_b21.py`; wraps the environment in a reward-only `VecNormalize` and changes `learning_rate` / `learning_starts` |
| `run_b25.py`, `test_b25_ipm.py` | B2.5: real IPM (Xie et al.: forged delta = −ε × mean honest delta) injected by swapping the official `LMP_attack` in-process (the only attack hook that receives the honest updates); extra conditions `fedavg` (uniform aggregation) and `none` (no attack, same partition). The unit test checks cos = −1 and norm ratio = ε, and that the official `IPM_attack` sends a zero update |
| `run_grid.sh`, `run_grid_b21.sh`, `run_grid_b23.sh`, `run_b23_escalonado.sh`, `run_grid_b23b.sh`, `run_grid_b25.sh` | Resumable grid launchers (skip runs whose final JSON exists), 5–6 GPU processes, `OMP_NUM_THREADS=3` |
| `analisar.py`, `analisar_b21.py`, `analisar_b23.py`, `analisar_b23b.py`, `analisar_b25.py` | Pre-registered analyses (TOST ±1.0 p.p., Wilcoxon, Holm; B2.2/B2.3/B2.3b rebuild the deterministic policy from the actor checkpoints) |

```bash
# official code: always with the isolated venv; 1.51 GPU-hours per 500-round run (RTX 3050),
# the GPU saturates at 4–6 processes (~9 h per batch of 6)
bash scripts/passo2_oficial/run_grid_b21.sh                                          # resumable
external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_b21.py           # only with the full grid

# our framework: project venv, CPU only (TensorFlow does not see the GPU there), so it can
# run alongside the GPU queue at low priority
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.c0_teto_oraculo
```

C.0 reads two inputs that are not versioned yet: `results/frente1_ablacao_adaaggrl/grade_combined_raw.csv` and `results/tables/exp9_dominance_grid_10seeds_ALL_root100_raw.csv`.

To recreate `external/`: `git clone https://github.com/yjEugenia/AdaAggRL.git external/AdaAggRL && git -C external/AdaAggRL checkout 27b9c18`, then build the venv with the versions listed in `external/.gitignore`.

### Status

| Experiment | Result |
|------------|--------|
| Passo 2 (seeds 100–104) | Equivalence not established (TOST p = 0.18, one late reset); exploratory: the TD3 policy does not move away from its initialization |
| B2.1 (seeds 105–114) | **Equivalence confirmed**: fixed − TD3 = −0.22 p.p., 90% CI (−0.87, +0.43), TOST p = 0.027 |
| B2.2 | **H3–H5 confirmed**: action correlation across attacks r = 0.991; policy drift ≈ 1/5 of the exploration noise; input sensitivity of the final actor equals that of a freshly initialized one |
| B2.3 (exploratory) | **The steelman does not beat the fixed action** (−10.2 p.p., p = 0.084) and is worse than the published TD3 (−10.4 p.p., p = 0.002). The policy moves (drift 0.47) but to a **constant corner** of the action box, independent of the state; under EB, corners with threshold a₅ ≈ 0 switch the filtering off and the model collapses (2/5 seeds). Limitation: one configuration, unnormalized official reward |
| C.0 (descriptive) | By the frozen criterion (gap > 2 p.p. against the oracle ceiling, 95% CI > 0), **2/19 cells have headroom**: `label_flipping` at α = 0.05 and 0.1. The skeleton loses to a classic fixed rule in 4 cells. The oracle ceiling is not a ceiling at α ≤ 0.1 (methods under attack exceed it), so "no headroom" there is not established |
| B2.8 (exploratory) | By the frozen criterion, **"granularity explains"**: in 8/14 target cells (skeleton beats the best static rule) not even an ideal per-round rule choice, which picks by looking at the test set, reaches per-client filtering. The verdict depends on three α = 0.5 cells with differences < 0.5 p.p.; requiring \|Δ\| > 1 p.p. gives 5/14 (inconclusive). The strong evidence is at α ≤ 0.1 with model attacks (oracle 4–14 p.p. below the skeleton); under `label_flipping` the per-round oracle beats the skeleton. The oracle picks FLTrust (root-anchored) in most rounds at α ≤ 0.1 |
| B2.3b (exploratory) | **Last steelman: does not beat the fixed action** (+0.95 p.p., p = 0.066; secondary metric 0.00 p.p.) → gate B-a is definitive and the configuration search ends. Reward normalization removed the B2.3 collapses (EB resets 0.8 vs. 15.8). The policy keeps drifting (0.17 at round 500, not saturated) towards a seed-specific interior action; its state dependence triples over training but stays at ~1/3.5 of the exploration noise. Across the published, B2.3 and B2.3b regimes, TD3 never beats the fixed action in 500 rounds |
| B2.6 (confirmatory) | H1 (memory helps) **not confirmed**: the effect depends on the attack and cancels out. H2 **confirmed**: the official weak memory (0.9^flag) beats λ = 2 by 0.63 p.p. H3 (binary ≈ soft) **not confirmed** because the binary mask is *better* (+1.02 p.p.). H4 (`cos_server` ≈ S_R) **not confirmed** after Holm: the signals are complementary (S_R wins model attacks at α = 0.05, `cos_server` wins `label_flipping` by 11.5 p.p.). Exploratory: the threshold b has the largest effect (b = 0.75: −10.5 p.p.) |
| A0 (descriptive) | **Attacker mass:** the 48 runs reproduce B2.6 exactly. Mean weight mass on attackers ≥ 0.05 in only **2/8** negative-gap cells (6 cells ≈ 0), so the C.0 negative gaps come from weighting honest clients, not from exploiting attackers (P3 direction 6 loses motivation). **Signal-selection ceiling:** choosing the signal per cell (leave-one-seed-out) gains **+1.75 p.p.** (95% CI +1.31 to +2.19) over the best single signal per seed (~+0.8 over `sr_b025`). **Signal agreement** (preliminary): S_R and `cos_server` are nearly orthogonal (Spearman −0.36 to +0.35) and fail on different attacks. **TD3 regimes:** final drift 0.0097 (published) / 0.47 (B2.3, saturated) / 0.17 (B2.3b); only B2.3b gains state dependence, reaching ~¼ of the exploration noise |
| B2.7 (exploratory) | By the frozen criterion (Δ > 0 with 95% CI > 0 in ≥ 3/8 cells), **no agent pays off at any horizon (15/50/150 rounds)**. TD3 vs. fixed: 0/8, 0/8, 1/8, mean Δ −1.65 → −1.12 → +1.33 p.p. (the trend comes from `label_flipping` and `low_mag_backdoor`). LinUCB vs. best fixed arm: −11.6 → −8.2 → −8.6 p.p., roughly tied with the random arm. DQN vs. best fixed arm: −13.3 → −11.6 → −9.5 p.p., and **below its own random arm** in 22/24 cell × horizon pairs. The oracle FedAvg-8 ceiling does not converge in 15 rounds at α ≤ 0.1 (+8–10 p.p. by 150), so part of the C.0 negative gaps was ceiling under-convergence; the `label_flipping` headroom persists (robustness). Reproducibility findings: the framework's `seed` does not seed the global `np.random` (results depend on execution order; B2.7 reseeds before every system), and the GRADF v1 DQN is non-deterministic run to run (~4 p.p.) |
| B2.5 (confirmatory) | Audit finding: the official `IPM_attack` sends a **null update** (the attacker returns the global model). With the real IPM (Xie et al., ε = 2 chosen by the pre-registered sanity check), H1 is **inconclusive**: fixed − TD3 = +2.78 p.p., 95% CI (−6.37, +11.92), TOST p = 0.66, Wilcoxon p = 0.63. The IPM triggers reset cascades in the official environment (median 2.5 extra resets per run; late cascades at seeds 110/111), so the sd of the per-seed differences is 12.8 p.p. (0.3 in B2.1). Mechanism unchanged: TD3 drift 0.011 and state dependence 0.003, both ≪ exploration noise (0.0475) — the policy does not learn. No evidence that TD3 contributes; equivalence not established |

Planning and audit documents (`references/roadmap_tese_gradf_v4.md` — the current plan —, `references/roadmap_tese_gradf_v3.md` — measured costs —, `references/auditoria_fidelidade_adaaggrl.md`, `references/c0_fechamento.md`, `references/slr_protocolo.md`, `references/proximos_passos_pos_ablacao.md`) are local-only (`references/` is git-ignored).

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
- `fedstrategist_selector.py` — `DiagnosticStateVector`/`LinUCBAgent`: a from-scratch reproduction of FedStrategist (Haque, Kamal & Hossain, 2025), a competing baseline in `exp4_adaptive.py` and `exp10_selector_comparison.py`. A 3-dim diagnostic state (variance of update norms, average pairwise cosine similarity, mean update norm) feeds a LinUCB contextual bandit over `{fedavg, median, krum}` (widened to the full 7-rule arsenal in `exp10`), rewarded by accuracy gain minus a per-rule cost (`C_fedavg=0.1, C_median=0.4, C_krum=0.8`, from the paper's own Appendix). The paper links a public repo, but it's a standalone PyTorch harness built around a different model representation than this codebase's flat-numpy-vector models — reproduced here from the paper's closed-form spec instead of ported.
- `adaaggrl_agent.py` — `AdaAggRLAgent`: a from-spec reproduction of AdaAggRL (Wang, Zhang, Wen, Qiu & Guo, AAAI 2025), the continuous per-client weighting family in `exp10_selector_comparison.py`. Gradient-inversion distribution learning + MMD-based environmental cues + TD3 over a continuous `[0,1]^5` action, with a persistent exponential penalty for repeatedly-flagged clients. Deviations and instantiation choices are documented in the module docstring.
- `dp_accountant.py` — `GaussianDPMechanism`: opt-in replacement for Layer 2's legacy per-coordinate Laplace noise. Real (ε,δ) accounting via the `dp_accounting` library's RDP accountant (not hand-rolled composition), L2 clip + Gaussian noise added ONCE to the aggregated output (not per-client before aggregation), with per-aggregation-rule sensitivity (tight for `fedavg`/`fedprox`/`fltrust`, a documented conservative bound for `median`/`trimmed_mean`/`krum`/`clustering`). Passed via `HardeningPipeline(dp_mechanism=...)`; default (`None`) keeps the legacy Laplace mechanism so existing experiment numbers are unaffected. See its docstring for the explicit trust model (model-release DP, not a defense against the server itself).
- `hardening.py` — `HardeningPipeline`: Layer 1 (magnitude/skewness validation) → Layer 2 (DP clipping + Laplace noise by default, or clip-only + `dp_mechanism.privatize_aggregate` after Layer 3 if a `GaussianDPMechanism` is passed) → Layer 3 (classifier + selector pick the aggregation strategy, majority vote across accepted clients) → Layer 4 (accuracy-regression check via a caller-supplied `evaluate_fn`) → Layer 5 (HE stub, not evaluated in the paper).

### XAI (`src/xai/`)

- `explanation_generator.py` — `XAIExplainer` generates a narrative + a *computed* counterfactual (from real old/new accuracy, not hardcoded text) for every ACCEPT/REJECT decision.
- `audit_trail.py` — `AuditTrail` persists explanations as JSON Lines (`.add/.load/.query/.clear`).

### Utilities (`src/utils/`)

- `data_loader.py` — `generate_detection_dataset()` and `generate_selector_experiences()` run real (small-scale) FL rounds with injected attacks to produce labeled training data for the classifier/combiner/selector, replacing the original `np.random` placeholders. `load_dataset_participants()` loads MNIST/CIFAR-10 shards.
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

## Dataset Setup

### MNIST + CIFAR-10

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
