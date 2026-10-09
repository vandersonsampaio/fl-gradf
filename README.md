# GRADF — Generic RL-based Adaptive Detection & Defense Framework

GRADF is a research framework for studying **adaptive defenses against Byzantine clients in Federated Learning (FL)**. It is domain-agnostic: it works on flat model-update vectors and is validated on non-IID MNIST and CIFAR-10. Per FL round, it combines:

1. multi-modal attack detection (gradient, accuracy, temporal and clustering signals);
2. attack classification (11+ attack types);
3. RL-based defense selection (7 aggregation rules);
4. a 5-layer hardening pipeline;
5. an XAI audit trail for every aggregation decision.

It also contains from-spec reproductions of four competing adaptive defenses: **TARS** (Ahmed et al., 2025), **AdaBFL** (Tang, Liu & Huang, 2026), **FedStrategist** (Haque, Kamal & Hossain, 2025) and **AdaAggRL** (Wang et al., AAAI 2025).

| Document | Contents |
|---|---|
| this file | overview, the p1.0.0 paper, quick start, audit summary |
| [`src/README.md`](src/README.md) | architecture, implementation status, known simplifications, design notes, tests |
| [`results/README.md`](results/README.md) | the post-p1.0.0 AdaAggRL audit: layout, conventions, how to run, full results |

---

## Version p1.0.0 — the paper

Tag **`p1.0.0`** is the code state behind the paper **"Rule Selection Fails Where Client Weighting Succeeds: A Characterization of Adaptive Defense in Byzantine-Robust Federated Learning"**. Use it to reproduce the paper.

The paper is a **characterization study**, not a new method; GRADF is used as a probe for discrete rule selection. Findings (MNIST):

1. **Root dataset size governs fixed-rule dominance.** FLTrust's apparent dominance comes from an oversized root (3,000 samples). At the canonical ~100 samples (Cao et al., NDSS 2022), its win rate drops from 18/21 to 8/21 grid cells, and from 5/7 to 0/7 under mild heterogeneity (α = 0.5).
2. **With a canonical root, there is headroom for adaptive defense**: no fixed rule dominates across attack × heterogeneity, and an oracle selector captures the margin.
3. **Capturing it depends on the mechanism class.** Over 10 seeds, two discrete rule selectors (GRADF's DQN and FedStrategist) do not beat random selection, while continuous per-client weighting (AdaAggRL) captures most of the headroom.
4. **Methodology.** Reference-anchored (FLTrust) vs. purely relational (Krum, Median, …) defenses explain why heterogeneity hurts some rules and not others. Evaluating adaptive defense needs a canonical root, strong baselines, multiple seeds, and random (floor) and oracle (ceiling) controls.

Experiments behind the paper: `exp9_dominance_grid`, `exp10_selector_comparison`, and `exp1_baseline` / `exp4_adaptive` with `--root_size 100`.

---

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requiriments.txt

python data/download_datasets.py      # MNIST + CIFAR-10, IID and non-IID (Dirichlet α = 0.5) shards, ~3 GB
python test_setup.py
pytest                                # 95 unit + integration tests
```

Experiments are run as modules (so `src.*` imports resolve); each writes `*_raw.csv` and `*_summary.csv` (mean, std, 95% CI, paired tests) to `results/tables/`:

```bash
# paper p1.0.0 (canonical root = 100, seeds 42–51)
python -m src.experiments.exp9_dominance_grid --seeds 42 43 44 45 46 47 48 49 50 51
python -m src.experiments.exp10_selector_comparison --variant b
python -m src.experiments.exp1_baseline --root_size 100 --seeds 42 43 44 45 46 47 48 49 50 51
python -m src.experiments.exp4_adaptive --root_size 100 --seeds 42 43 44 45 46 47 48 49 50 51

# other experiments
python -m src.experiments.exp2_robustness --seeds 42 43 44
python -m src.experiments.exp7_modality_ablation --seeds 42 43 44
python -m src.experiments.exp8_xai_examples        # after exp2 (reads its audit trails)
python -m src.experiments.runner                   # exp1–4, 6, 7 in sequence
```

Notebooks `02`–`08` in `notebooks/` are executed on MNIST (baseline, detection, classifier/selector training, attack simulation, results). `01` (MIMIC-III) needs credentialed PhysioNet access and is not executed.

---

## Repository layout

```
src/            GRADF and the reproduced defenses (FL loop, attacks, detection, defense, XAI, experiments)
data/           dataset download and partitioning scripts (data files are not versioned)
notebooks/      exploratory notebooks
tests/          integration tests (unit tests live in src/**/tests/)
scripts/        post-p1.0.0 audit experiments (in-house framework and official AdaAggRL code)
results/        experiment outputs; one folder per audit experiment
external/       official AdaAggRL clone and its venv (not versioned; see results/README.md)
config/         FL and network hyperparameters
```

---

## After p1.0.0: AdaAggRL audit

Follow-up work toward a second paper: **does AdaAggRL's RL component (TD3) contribute, or does its fixed skeleton (per-client signal, threshold and memory) do the work?** All experiments are pre-registered (plan + hashes frozen before any run) and live outside `src/`, so the p1.0.0 code is untouched. Details and full numbers: [`results/README.md`](results/README.md).

| Experiment | Result in one line |
|---|---|
| Step 2, B2.1, B2.2 | In the official code (MNIST, 500 rounds), TD3 is **equivalent** to a fixed action (±1 p.p.); the policy barely moves from its initialization and ignores its input |
| B2.3, B2.3b | Two steelman configurations of TD3 never beat the fixed action; one collapses into a constant corner of the action space |
| B2.4r | The threshold a₅ is causal (below 0.25 the model collapses), but no constant beats the center under EB |
| B2.5 | The official IPM attack is a null update; with the real IPM, equivalence is not established (reset cascades) and TD3 still does not learn |
| B2.6 | Memory does not explain the headroom; a binary mask is as good as or better than soft weighting; S_R and `cos_server` are complementary; the threshold has the largest effect |
| B2.7, B2.7a/b | No agent (TD3, LinUCB, DQN) pays off at 15, 50 or 150 rounds; TD3's only positive cell is not learning, and a fixed b = 0.25 beats it |
| B2.8, B2.8s | Even a per-round oracle over 7 rules does not reach per-client filtering at α ≤ 0.1 with model attacks ("granularity explains") |
| C.0, C0b | At 150 rounds, the best existing method closes almost all remaining headroom; only `label_flipping` α = 0.05 is left (+2.8 p.p.) |
| A0 | The negative gaps come from uniform vs. sample-size weighting, not from exploiting attackers; signal selection per cell adds only ~1–2 p.p. |
| D2 | P1's conclusion is robust to run-to-run variance; GRADF's DQN is strongly non-deterministic per cell |
| B3.0, B3.1, B3.2 | On BloodMNIST (health data), TD3 ≈ fixed again (±3 p.p.) and the policy stays static, but both run in a degraded, periodic-reset regime |
| B3.3 | Threshold sweep on BloodMNIST — planned, not run yet |
