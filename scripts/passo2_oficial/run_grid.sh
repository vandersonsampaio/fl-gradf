#!/usr/bin/env bash
# Confirmatory Step 2 grid (results/frente1_passo2_oficial/PREREGISTRO.md §4).
# Resumable: skips runs whose JSON already exists. Usage: bash scripts/passo2_oficial/run_grid.sh
set -u
cd "$(dirname "$0")/../.."
PY=external/.venv_adaaggrl/bin/python
OUT=results/frente1_passo2_oficial/raw
mkdir -p "$OUT"
jobs() {
  for seeds in "100 101 102" "103 104"; do
    for s in $seeds; do for a in LMP EB; do for c in td3 fixed random; do
      f="$OUT/MNIST_${a}_q0.5_${c}_seed${s}_R500.json"
      [ -f "$f" ] || echo "$a $c $s"
    done; done; done
  done
}
jobs | OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 xargs -P 6 -L 1 bash -c \
  'echo "$(date +%F\ %T) START $0 $1 $2"; '"$PY"' scripts/passo2_oficial/run_oficial.py --attack $0 --condition $1 --seed $2 --rounds 500 --q 0.5 > '"$OUT"'/$0_$1_$2.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1 $2" || echo "$(date +%F\ %T) FAIL $0 $1 $2"'
echo "$(date +%F\ %T) GRID_END"
