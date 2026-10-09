#!/usr/bin/env bash
# Exploratory B2.3b grid (last steelman; results/b23b_steelman_normalizado/PLANO.md).
# 10 runs: seeds 100-104 x {LMP, EB}. Resumable: skips runs whose JSON already exists.
# Usage: bash scripts/passo2_oficial/run_grid_b23b.sh
set -u
cd "$(dirname "$0")/../.."
PY=external/.venv_adaaggrl/bin/python
OUT=results/b23b_steelman_normalizado/raw
mkdir -p "$OUT"
jobs() {
  for s in 100 101 102 103 104; do for a in LMP EB; do
    f="$OUT/MNIST_${a}_q0.5_td3_seed${s}_R500.json"
    [ -f "$f" ] || echo "$a $s"
  done; done
}
jobs | OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 xargs -P 6 -L 1 bash -c \
  'echo "$(date +%F\ %T) START $0 steelman_b $1"; '"$PY"' scripts/passo2_oficial/run_b23b.py --attack $0 --seed $1 > '"$OUT"'/$0_steelman_b_$1.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 steelman_b $1" || echo "$(date +%F\ %T) FAIL $0 steelman_b $1"'
echo "$(date +%F\ %T) GRID_END"
