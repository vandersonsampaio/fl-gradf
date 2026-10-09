#!/usr/bin/env bash
# Exploratory B2.3 grid (TD3 steelman; results/b23_steelman_oficial/PLANO.md).
# 10 runs: seeds 100-104 x {LMP, EB}, TD3 with lr 1e-3 and learning_starts 10.
# Resumable: skips runs whose JSON already exists.
# Usage: bash scripts/passo2_oficial/run_grid_b23.sh [--after-b21]
#   --after-b21: waits for the B2.1 GRID_END before starting (so the two grids do not share the GPU).
set -u
cd "$(dirname "$0")/../.."
PY=external/.venv_adaaggrl/bin/python
OUT=results/b23_steelman_oficial/raw
mkdir -p "$OUT"
if [ "${1:-}" = "--after-b21" ]; then
  echo "$(date +%F\ %T) WAIT b21 GRID_END"
  until grep -q GRID_END results/b21_replicacao_oficial/grid.log 2>/dev/null; do sleep 60; done
  echo "$(date +%F\ %T) b21 finished"
fi
jobs() {
  for s in 100 101 102 103 104; do for a in LMP EB; do
    f="$OUT/MNIST_${a}_q0.5_td3_seed${s}_R500.json"
    [ -f "$f" ] || echo "$a $s"
  done; done
}
jobs | OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 xargs -P 6 -L 1 bash -c \
  'echo "$(date +%F\ %T) START $0 steelman $1"; '"$PY"' scripts/passo2_oficial/run_b23.py --attack $0 --seed $1 > '"$OUT"'/$0_steelman_$1.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 steelman $1" || echo "$(date +%F\ %T) FAIL $0 steelman $1"'
echo "$(date +%F\ %T) GRID_END"
