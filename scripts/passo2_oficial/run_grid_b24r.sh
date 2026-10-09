#!/usr/bin/env bash
# B2.4r grid (results/b24r_limiar_oficial/PLANO.md + ADENDO1.md): a₅ ∈ {0; 0.1; 0.25; 0.475 (center, re-run); 0.95}
# × seeds 100-104, EB, 500 rounds (25 runs).
# Resumable: skips runs whose final JSON already exists. Usage: bash scripts/passo2_oficial/run_grid_b24r.sh
set -u
cd "$(dirname "$0")/../.."
PY=external/.venv_adaaggrl/bin/python
OUT=results/b24r_limiar_oficial/raw
mkdir -p "$OUT"
jobs() {
  for s in 100 101 102 103 104; do for a5 in 0 0.1 0.25 0.475 0.95; do
    [ -f "$OUT/a5_$a5/MNIST_EB_q0.5_fixed_seed${s}_R500.json" ] || echo "$a5 $s"
  done; done
}
jobs | OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 xargs -P 5 -L 1 bash -c \
  'echo "$(date +%F\ %T) START $0 $1"; '"$PY"' scripts/passo2_oficial/run_b24r.py --a5 $0 --seed $1 > '"$OUT"'/a5_$0_$1.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1" || echo "$(date +%F\ %T) FAIL $0 $1"'
echo "$(date +%F\ %T) GRID_END"
