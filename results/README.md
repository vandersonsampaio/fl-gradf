# AdaAggRL audit (after p1.0.0)

Back to the [main README](../README.md).

**Question:** does AdaAggRL's RL component (TD3) contribute, or does its fixed skeleton (per-client signal, threshold and memory) capture the headroom? The experiments run in two settings:

- **official code:** the published AdaAggRL (`external/AdaAggRL`, commit `27b9c18`), never edited, driven by `scripts/passo2_oficial/`;
- **in-house framework:** our reproduction in `src/`, driven by `scripts/` (logistic MNIST, 10 clients, 2 Byzantine, root 100, CPU only).

- [Conventions](#conventions)
- [Running](#running)
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

## Running

```bash
# official code: isolated venv; ~1.5 GPU-hours per 500-round run (RTX 3050), 4–6 processes saturate the GPU
git clone https://github.com/yjEugenia/AdaAggRL.git external/AdaAggRL && git -C external/AdaAggRL checkout 27b9c18
# build external/.venv_adaaggrl with the versions listed in external/.gitignore
bash scripts/passo2_oficial/run_grid_b21.sh                                   # resumable grid launcher
external/.venv_adaaggrl/bin/python scripts/passo2_oficial/analisar_b21.py    # only with the complete grid

# in-house framework: project venv, CPU only, can run alongside the GPU queue
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 venv/bin/python -m scripts.b27_horizonte analisar
```

Launchers are resumable (they skip runs whose final file exists) and write `grid.log` (START/DONE/FAIL/GRID_END). `scripts/janela_execucao.sh <PGID> <name>` optionally suspends a grid during working hours (Mon–Fri 7h–18h) and resumes it afterwards.

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
| `b33_limiar_bloodmnist/` (B3.3) | `run_b33.py`, `analisar_b33.py` | Does a higher threshold fix the BloodMNIST regime? | Planned, not run yet |

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
