# AdaAggRL audit (after p1.0.0)

Back to the [main README](../README.md).

**Question:** does AdaAggRL's RL component (TD3) contribute, or does its fixed skeleton (per-client signal, threshold and memory) capture the headroom? The experiments run in two settings:

- **official code:** the published AdaAggRL (`external/AdaAggRL`, commit `27b9c18`), never edited, driven by `scripts/passo2_oficial/`;
- **in-house framework:** our reproduction in `src/`, driven by `scripts/` (logistic MNIST, 10 clients, 2 Byzantine, root 100, CPU only).

- [Conventions](#conventions)
- [Reproducing](#reproducing)
- [Experiments](#experiments)
- [Inputs not versioned](#inputs-not-versioned)

---

## Conventions

- **Pre-registration.** Each experiment has a `PREREGISTRO.md` or `PLANO.md` (plus `ADENDO*.md` for later changes), frozen with a `.sha256` before any run. Analyses are pre-written and run once on the complete grid; accuracies are not looked at before that.
- **Language.** The frozen documents are in Portuguese and stay untouched; each has an English translation next to it (`*.en.md`). Results (`RESULTADO.md`) and outputs (`analise.txt`) are in English. [`HASH_PROVENANCE.md`](HASH_PROVENANCE.md) explains how to verify the frozen hashes against the commit before the translation.
- **Folder contents.** `raw/` (one JSON or CSV per run: per-round accuracy, loss, executed action, weight mass on real attackers, resets), `analise.txt`, `resumo_runs.csv`, `RESULTADO.md`. Runs on the official code also keep the observed states (`raw/obs/`) and the TD3 actor at steps 0 and 500 (`raw/actors/`). Logs are not versioned.
- **Statistics.** The unit is the seed (or attack × seed pair). Equivalence by paired TOST, differences by Wilcoxon, Holm for multiple comparisons; CI90 and CI95 are reported.
- **GPU non-reproducibility.** The official code on GPU is not bit-reproducible: the same seed fixes the initialization, not the trajectory. Pairing by seed pairs the configuration; runs with a complete result are never re-run.

---

## Reproducing

### Setup

```bash
# in-house framework: the project venv (see the main README); CPU only
python -m venv venv && venv/bin/pip install -r requiriments.txt

# official code: AdaAggRL at commit 27b9c18, isolated venv (versions in external/.gitignore); GPU
git clone https://github.com/yjEugenia/AdaAggRL.git external/AdaAggRL && git -C external/AdaAggRL checkout 27b9c18

# BloodMNIST (B3.x only): data/raw/medmnist/bloodmnist.npz from MedMNIST v2 (Zenodo record 10519652), then
external/.venv_adaaggrl/bin/python scripts/passo2_oficial/treinar_extrator_bloodmnist.py
```

In the tables below, `$OF` stands for `external/.venv_adaaggrl/bin/python scripts/passo2_oficial` (official code) and `$IN` for `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts` (in-house framework). `G` stands for `bash scripts/passo2_oficial` in the official-code table and for `bash scripts` in the in-house table.

- **Launchers are resumable:** they skip runs whose final file exists and write `raw/grid.log` (START/DONE/FAIL/GRID_END).
- **Analyses run on the complete grid,** once. Each writes the files listed in its `RESULTADO.md`.
- **Cost:** ~1.5 GPU-hours per 500-round official run (RTX 3050); 4–6 processes saturate the GPU. In-house runs are CPU-only and can share the machine with the GPU queue.
- **Optional:** `bash scripts/janela_execucao.sh <PGID> <name>` suspends a grid during working hours (Mon–Fri 7h–18h) and resumes it afterwards; the B3.1 and B3.3 launchers write the PGID to `raw/grid.pgid`.
- **Exact numbers will differ on GPU** (see Conventions); the pre-registered verdicts are what should reproduce.

### Official code (`scripts/passo2_oficial/`)

| ID | Result | Seeds (runs) | Run | Analyze |
|---|---|---|---|---|
| Step 2 | [`frente1_passo2_oficial/`](frente1_passo2_oficial/RESULTADO.md) | 100–104 × {LMP, EB} × {td3, fixed, random} (30) | `G/run_grid.sh` | `$OF/analisar.py` |
| B2.1 + B2.2 | [`b21_replicacao_oficial/`](b21_replicacao_oficial/RESULTADO.md) | 105–114 × {LMP, EB} × {td3, fixed} (40) | `G/run_grid_b21.sh` | `$OF/analisar_b21.py` |
| B2.3 | [`b23_steelman_oficial/`](b23_steelman_oficial/RESULTADO.md) | 100–104 × {LMP, EB}, TD3 steelman (10) | `G/run_grid_b23.sh` | `$OF/analisar_b23.py` |
| B2.3b | [`b23b_steelman_normalizado/`](b23b_steelman_normalizado/RESULTADO.md) | 100–104 × {LMP, EB} (10) | `G/run_grid_b23b.sh` | `$OF/analisar_b23b.py` |
| B2.4r | [`b24r_limiar_oficial/`](b24r_limiar_oficial/RESULTADO.md) | a₅ ∈ {0, 0.1, 0.25, 0.475, 0.95} × 100–104, EB (25) | `G/run_grid_b24r.sh` | `$OF/analisar_b24r.py` |
| B2.5 | [`b25_ipm_oficial/`](b25_ipm_oficial/RESULTADO.md) | sanity: seed 100, 100 rounds (5); grid: 105–114 × {td3, fixed}, real IPM ε = 2 (20) | `$OF/test_b25_ipm.py`; `G/run_grid_b25.sh sanity`; `G/run_grid_b25.sh grade 2` | `$OF/analisar_b25.py sanity`; `$OF/analisar_b25.py grade --eps 2` |
| B3.0 | [`b30_medmnist_sanity/`](b30_medmnist_sanity/RESULTADO.md) | 130–132 × {BloodMNIST none/LMP/EB, MNIST none}, FedAvg (12) | `G/run_grid_b30.sh` | `$OF/analisar_b30.py` |
| B3.1 + B3.2 | [`b31_medmnist_oficial/`](b31_medmnist_oficial/RESULTADO.md) | 135–144 × {LMP, EB} × {td3, fixed}, BloodMNIST (40) | `G/run_grid_b31.sh "LMP EB"` | `$OF/analisar_b31.py --margem 3.0 --ataques LMP EB`; post hoc `$OF/b31_pi0_referencia.py` |
| B3.3 | [`b33_limiar_bloodmnist/`](b33_limiar_bloodmnist/RESULTADO.md) | a₅ ∈ {0.475, 0.75, 0.95} × 145–149, EB, BloodMNIST (15) | `G/run_grid_b33.sh` | `$OF/analisar_b33.py` |

All official runs use q = 0.5 and 500 rounds unless stated. Row order is the order of execution; A0's TD3-regime analysis reads B2.1/B2.3/B2.3b outputs.

### In-house framework (`scripts/`)

MNIST, logistic model, 10 clients (clients 0 and 1 Byzantine), root 100. Cells are attack × α over 7 attacks and α ∈ {0.5, 0.1, 0.05}.

| ID | Result | Seeds (runs) | Run | Analyze |
|---|---|---|---|---|
| Front 1 ablation | [`frente1_ablacao_adaaggrl/`](frente1_ablacao_adaaggrl/RESULTADO.md) | 42–51 × 4 variants × 21 cells, 15 rounds (840) | `$IN.frente1_ablacao_adaaggrl run --mode <fixed\|sr_only\|cosmed_only\|cosserver_only> --seeds 42 … 51` | `$IN.frente1_ablacao_adaaggrl analyze` (step 1: `auc`); V0: `$IN.frente1_v0_diagnostico` |
| C.0 | [`c0_espaco_restante/`](c0_espaco_restante/NOTAS.md) (+ `CORRECAO_2026-10-04.md`) | 42–51, from the exp9 table and Front 1 outputs, 15 rounds | `$IN.c0_espaco_restante`; `$IN.c0_teto_oraculo`; `$IN.c0_teto_corrigido` | (same scripts) |
| B2.6 | [`b26_decomposicao/`](b26_decomposicao/RESULTADO.md) | 52–61 × 8 variants (80 jobs × 21 cells) | `G/run_grid_b26.sh` | `$IN.b26_decomposicao analisar` |
| B2.8 | [`b28_oraculo_por_regra/`](b28_oraculo_por_regra/RESULTADO.md) | 42–51 | `$IN.b28_oraculo_por_regra run --seeds 42 43` (and so on, in pairs) | `$IN.b28_oraculo_por_regra analisar` |
| B2.8s | [`b28s_sensibilidade_metrica/`](b28s_sensibilidade_metrica/RESULTADO.md) | 42–51 × 21 cells (210) | `G/run_grid_b28s.sh` | `$IN.b28s_sensibilidade_metrica analisar` |
| A0 | [`a0_analises/`](a0_analises/RESULTADO.md) | mass: 52–54; the rest reuses Front 1, B2.6, B2.1, B2.3 and B2.3b outputs | `$IN.a0_analises massa --seed 52` (53, 54) | `$IN.a0_analises analisar`; `$OF/a0_regimes_td3.py` |
| B2.7 | [`b27_horizonte/`](b27_horizonte/RESULTADO.md) | 42–51 × 8 cells, H = 150 (80) + 30 ceilings | `G/run_grid_b27.sh` | `$IN.b27_horizonte verificar`; `$IN.b27_horizonte analisar` |
| B2.7a/b | [`b27b_td3_constante/`](b27b_td3_constante/RESULTADO.md) | 42–51 × 3 cells, H = 150: B2.7a (30), B2.7b × 4 systems (120) | `G/run_grid_b27a.sh`; `G/run_grid_b27b.sh` | `$IN.b27b_td3_constante analisar_b27a`; `$IN.b27b_explicativo analisar` |
| C0b | [`c0b_espaco_h150/`](c0b_espaco_h150/RESULTADO.md) (+ `VERIFICACOES.md`) | 72–81 × 19 cells, H = 150 + 30 ceilings | `G/run_grid_c0b.sh`; uniform-oracle sensitivity: `$IN.c0b_sensibilidade_oraculo_uniforme teto --alpha <α> --seed <s>` | `$IN.c0b_espaco_h150 analisar`; `$IN.c0b_sensibilidade_oraculo_uniforme analisar` |
| C0b-iii | [`c0b_iii_metrica/`](c0b_iii_metrica/CANCELADO.md) | planned 72–81; **cancelled before running** | — | — |
| D2 | [`d2_variancia_p1/`](d2_variancia_p1/RESULTADO.md) | 42–44 × 3 repetitions of exp10 from tag `p1.0.0` (9) | `git worktree add ../p1 p1.0.0`; `G/run_grid_d2.sh ../p1` | `venv/bin/python scripts/d2_variancia_p1.py analisar` |

Some analyses read the outputs of earlier rows (C.0 and A0 read Front 1; A0 reads B2.6). The in-house analyses also read the P1 tables that are versioned in `results/tables/` (`exp9_dominance_grid_10seeds_ALL_root100_raw.csv`, `exp10_selector_comparison_variantb_full_seed*_raw.csv`, `exp10_FULL_variantb_raw.csv`), produced at tag `p1.0.0`.

---

## Experiments

### Official code (`scripts/passo2_oficial/`)

| Folder | Scripts | Question | Result |
|---|---|---|---|
| `frente1_passo2_oficial/` (Step 2) | `run_oficial.py`, `analisar.py` | Does TD3 beat a fixed or a random action? (MNIST, 500 rounds, seeds 100–104) | Inconclusive (one late reset); exploratory: the policy does not move from its initialization |
| `b21_replicacao_oficial/` (B2.1 + B2.2) | `run_b21.py`, `analisar_b21.py` | Confirmatory replication and mechanism (seeds 105–114) | **Equivalence confirmed**: fixed − TD3 = −0.22 p.p., CI90 (−0.87, +0.43). Drift ≈ 1/5 of the exploration noise; input sensitivity equals a fresh network's |
| `b23_steelman_oficial/` (B2.3) | `run_b23.py`, `analisar_b23.py` | Does TD3 learn with lr 1e-3 and a short warm-up? | No: −10.2 p.p. vs. fixed; the policy saturates at a constant corner and collapses under EB in 2/5 seeds |
| `b23b_steelman_normalizado/` (B2.3b) | `run_b23b.py`, `analisar_b23b.py` | Last steelman: normalized reward, lr 1e-4 | No: +0.95 p.p. (n.s.; 0.00 on the secondary). Collapses gone; state dependence grows but stays at ~1/3.5 of the noise. Configuration search ends |
| `b24r_limiar_oficial/` (B2.4r) | `run_b24r.py`, `analisar_b24r.py` | Is the threshold a₅ causal? Is there a better constant? (EB) | a₅ ≤ 0.1 collapses (34% vs. 97%); 0.25–0.95 is a plateau; **no constant beats the center** |
| `b25_ipm_oficial/` (B2.5) | `run_b25.py`, `test_b25_ipm.py`, `analisar_b25.py` | Fixed vs. TD3 under the IPM attack | The official IPM is a **null update**. With the real IPM (ε = 2): inconclusive (reset cascades, sd 12.8 p.p.); TD3 still does not learn |
| `b30_medmnist_sanity/` (B3.0) | `bloodmnist_shim.py`, `treinar_extrator_bloodmnist.py`, `run_b3.py`, `analisar_b30.py` | Does the official code converge on BloodMNIST? Which attacks and margin? | Converges (77.8%); LMP and EB included; margin capped at ±3.0 p.p. (weak equivalence) |
| `b31_medmnist_oficial/` (B3.1 + B3.2) | `run_b3.py`, `analisar_b31.py`, `b31_pi0_referencia.py` | Fixed vs. TD3 on BloodMNIST (seeds 135–144) | **Equivalence confirmed** (+0.13 p.p., CI90 −0.64 to +0.90); H4/H5 confirmed, H3 not. Both run in a degraded, periodic-reset regime (~31% accuracy) |
| `b33_limiar_bloodmnist/` (B3.3) | `run_b33.py`, `analisar_b33.py` | Does a higher threshold fix the BloodMNIST regime? (EB, seeds 145–149) | **No constant beats the center** (Δ −0.64 and −0.14 p.p., CI95 include 0). Higher thresholds exclude more clients and halve the resets, but accuracy stays at ~31%: a limit of the published mechanism, not a tuning problem |

### In-house framework (`scripts/`)

| Folder | Scripts | Question | Result |
|---|---|---|---|
| `frente1_ablacao_adaaggrl/` (ablation) | `frente1_ablacao_adaaggrl.py`, `frente1_v0_diagnostico.py` | Does AdaAggRL work without TD3, and which per-client signal matters? (15 rounds) | Removing TD3 improves accuracy (+1.8 p.p., p = 0.049); S_R alone is as good; a median-based trivial detector fails; `cos_server` is complementary to S_R |
| `c0_espaco_restante/` (C.0) | `c0_espaco_restante.py`, `c0_teto_corrigido.py`, `c0_teto_oraculo.py` | Remaining headroom per cell at 15 rounds | 2/19 cells (`label_flipping` α ≤ 0.1). Superseded by C0b; the oracle explanation is corrected in `CORRECAO_2026-10-04.md` |
| `b26_decomposicao/` (B2.6) | `b26_decomposicao.py` | Which skeleton pieces matter: memory, weighting shape, signal? | Memory does not help on average; the official weak memory beats λ = 2; a binary mask is as good or better; S_R and `cos_server` are complementary; the threshold has the largest effect |
| `b28_oraculo_por_regra/` (B2.8) | `b28_oraculo_por_regra.py` | Can the best rule per round reach per-client filtering? | "Granularity explains" (8/14 target cells), strongly at α ≤ 0.1 with model attacks; on `label_flipping` the per-round oracle wins |
| `b28s_sensibilidade_metrica/` (B2.8s) | `b28s_sensibilidade_metrica.py` | Does B2.8 depend on the metric? | Vacuous by construction (client test sets are equal IID splits); the B2.8 verdict reproduces |
| `a0_analises/` (A0) | `a0_analises.py`, `passo2_oficial/a0_regimes_td3.py` | Attacker mass, signal-selection ceiling, signal agreement, TD3 regimes | No attacker exploitation; choosing one signal per cell adds +1.75 p.p.; S_R and `cos_server` are nearly orthogonal |
| `b27_horizonte/` (B2.7) | `b27_horizonte.py` | Does learning pay off at 15, 50 or 150 rounds? | No agent pays off at any horizon; the DQN is below its own random arm in 22/24 cases |
| `b27b_td3_constante/` (B2.7a/b) | `b27b_td3_constante.py`, `b27b_explicativo.py` | Is TD3's one positive cell at H = 150 learning? | No: the policy is state-independent, a frozen-actor TD3 matches it, and a fixed b = 0.25 beats it by 4.4–6.9 p.p. |
| `c0b_espaco_h150/` (C0b) | `c0b_espaco_h150.py`, `c0b_sensibilidade_oraculo_uniforme.py` | Remaining headroom at 150 rounds against the best existing method | 0/19 (pre-registered oracle); 1/19 with the uniform oracle (`label_flipping` α = 0.05, +2.8 p.p.). Sample-size weighting alone costs up to ~4.5 p.p. under label skew |
| `c0b_iii_metrica/` (C0b-iii) | `c0b_iii_metrica.py` | Metric sensitivity of C0b | Cancelled before running (`CANCELADO.md`) |
| `d2_variancia_p1/` (D2) | `d2_variancia_p1.py` | Are the P1 numbers robust to run-to-run variance? | Yes in aggregate (all 12 executions × seeds keep the signs); GRADF's DQN varies up to 63 p.p. per cell |

Each `RESULTADO.md` has the full numbers, caveats and consequences.

---

## Inputs not versioned

- `external/AdaAggRL` and `external/.venv_adaaggrl` (see [Running](#running)).
- `data/models/extract_feature_bloodmnist.pt` (regenerated by `scripts/passo2_oficial/treinar_extrator_bloodmnist.py`) and `data/raw/medmnist/bloodmnist.npz` (MedMNIST v2, Zenodo record 10519652).
