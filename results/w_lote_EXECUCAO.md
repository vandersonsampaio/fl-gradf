# W batch (W1–W3): execution guide

Plans frozen on 2026-10-23 (hashes in each folder's `PLANO.sha256`). The code is written and unit-tested. **Nothing has been run yet.** Every step below runs outside the Mon–Fri 8:00–18:00 window, or under `scripts/janela_execucao.sh`.

Each plan fixes the order:
1. code checks, then addendum 1 with the code hashes;
2. validity gate, then addendum 2 (W2 and W3 only);
3. confirmation grid;
4. analysis.

## Code

| file | role |
|---|---|
| `src/classification/attack_simulator.py` | new data attack `pixel_backdoor`, `apply_trigger`, `attack_success_rate` (W2) |
| `src/fl/attacked_learner.py` | routes `pixel_backdoor` in `_compute_param_updates`; `backdoor_kwargs` (W2) |
| `src/classification/tests/test_pixel_backdoor.py` | W2 unit tests (PLANO §5.2) |
| `scripts/w_common.py` | weight tap, paired stats, Holm, verdicts (W2/W3) |
| `scripts/w2_backdoor_asr.py`, `scripts/run_grid_w2.sh` | W2 runner, gate, checks, analysis, launcher |
| `scripts/w3_adaptativo_limiar.py`, `scripts/run_grid_w3.sh` | W3 runner, gate, checks, analysis, launcher |
| `tests/test_w3_adaptive.py` | W3 unit tests (PLANO §5.2) and tests of the shared helpers |
| `scripts/passo2_oficial/run_w1.py`, `scripts/passo2_oficial/run_grid_w1.sh` | W1 runner (split hook, held-out evaluation), checks, GPU launcher |

`pytest`: 106 passed on 2026-10-23.

## W2: backdoor with ASR (CPU)

```bash
# 1. checks (PLANO §5): regression vs B2.7/B2.7b at H = 15 (|Δ| = 0) + ASR sanity -> verificacao.txt
CUDA_VISIBLE_DEVICES="" venv/bin/python -m scripts.w2_backdoor_asr verificar
#    then write ADENDO1.md (check output + sha256 of the files above) and its hash
# 2. gate (PLANO §4), rung 1; if NOT VALID, repeat with rung 2, then rung 3
setsid bash scripts/run_grid_w2.sh gate 1 12 > results/w2_backdoor_asr/gate1.log 2>&1 &
#    -> results/w2_backdoor_asr/gate_rung1.txt; write ADENDO2.md with the chosen rung and its hash
# 3. confirmation (180 runs, ~5 h)
setsid bash scripts/run_grid_w2.sh confirm <rung> 12 > results/w2_backdoor_asr/grid.log 2>&1 &
bash scripts/janela_execucao.sh <PGID> W2 &
# 4. analysis (only at 180/180)
venv/bin/python -m scripts.w2_backdoor_asr analisar --rung <rung>
```

## W3: adaptive attacker against the threshold (CPU)

```bash
CUDA_VISIBLE_DEVICES="" venv/bin/python -m scripts.w3_adaptativo_limiar verificar   # regression vs B2.7 -> ADENDO1.md
setsid bash scripts/run_grid_w3.sh gate 1 12 > results/w3_adaptativo_limiar/gate1.log 2>&1 &     # 60 runs; rung 2/3 if NOT VALID
#    -> gate_rung1.txt; ADENDO2.md. If no rung passes, W3 stops (PLANO §4)
setsid bash scripts/run_grid_w3.sh confirm <rung> 12 > results/w3_adaptativo_limiar/grid.log 2>&1 &   # 120 runs, ~4 h
venv/bin/python -m scripts.w3_adaptativo_limiar analisar --rung <rung>
```

W2 and W3 can share the CPU (for example, 6 + 6 processes) while W1 uses the GPU.

## W1: independent evaluation, official code (GPU, ~130 h)

```bash
# 1. checks 1-3 (PLANO §6): sizes/disjointness, batch counts, and CPU histories with vs. without the held-out evaluation (5 rounds)
CUDA_VISIBLE_DEVICES="" external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_w1.py verificar
# 2. check 4: 10-round GPU smoke per dataset, td3 and fixed
for d in MNIST BloodMNIST; do for c in td3 fixed; do
  external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_w1.py run --dataset $d --attack EB --condition $c \
    --seed 160 --rounds 10 --out_dir results/w1_protocolo_independente_oficial/verificacao/smoke_gpu; done; done
#    then ADENDO1.md with the outputs + code hashes
# 3. grid (86 runs, interleaved, 6 processes)
setsid bash scripts/passo2_oficial/run_grid_w1.sh 6 > results/w1_protocolo_independente_oficial/grid.log 2>&1 &
bash scripts/janela_execucao.sh $(cat results/w1_protocolo_independente_oficial/raw/grid.pgid) W1 &
```

**W1 analysis:** `scripts/passo2_oficial/analisar_w1.py` (PLANO §5), already written; run it once, at 86/86 (with an incomplete grid it only lists the missing runs).
