#!/usr/bin/env bash
# Confirmatory B2.1 + B2.2 grid (results/b21_replicacao_oficial/PREREGISTRO.md §4).
# 40 runs: seeds 105-114 x {LMP, EB} x {td3, fixed}. Resumable: skips runs whose JSON already exists.
# Usage: bash scripts/passo2_oficial/run_grid_b21.sh
set -u
cd "$(dirname "$0")/../.."
PY=external/.venv_adaaggrl/bin/python
OUT=results/b21_replicacao_oficial/raw
mkdir -p "$OUT"
jobs() {
  for s in 105 106 107 108 109 110 111 112 113 114; do for a in LMP EB; do for c in td3 fixed; do
    f="$OUT/MNIST_${a}_q0.5_${c}_seed${s}_R500.json"
    [ -f "$f" ] || echo "$a $c $s"
  done; done; done
}
jobs | OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 xargs -P 6 -L 1 bash -c \
  'echo "$(date +%F\ %T) START $0 $1 $2"; '"$PY"' scripts/passo2_oficial/run_b21.py --attack $0 --condition $1 --seed $2 > '"$OUT"'/$0_$1_$2.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1 $2" || echo "$(date +%F\ %T) FAIL $0 $1 $2"'
echo "$(date +%F\ %T) GRID_END"
