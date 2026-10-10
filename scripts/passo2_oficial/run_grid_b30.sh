#!/usr/bin/env bash
# B3.0 (results/b31_medmnist_oficial/PREREGISTRO.md §3): uniform FedAvg, 500 rounds, seeds 130-132:
# BloodMNIST without attack, LMP and EB; MNIST without attack (margin reference). 12 runs. Resumable.
# Usage: bash scripts/passo2_oficial/run_grid_b30.sh
set -u
cd "$(dirname "$0")/../.."
PY=external/.venv_adaaggrl/bin/python
OUT=results/b30_medmnist_sanity/raw
mkdir -p "$OUT"
jobs() {
  for s in 130 131 132; do for spec in "BloodMNIST none" "BloodMNIST LMP" "BloodMNIST EB" "MNIST none"; do
    set -- $spec
    [ -f "$OUT/$1_$2_q0.5_fedavg_seed${s}_R500.json" ] || echo "$1 $2 $s"
  done; done
}
jobs | OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 xargs -P 6 -L 1 bash -c \
  'echo "$(date +%F\ %T) START $0 $1 $2"; '"$PY"' scripts/passo2_oficial/run_b3.py --dataset $0 --attack $1 --condition fedavg --seed $2 --out_dir '"$OUT"' > '"$OUT"'/$0_$1_$2.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1 $2" || echo "$(date +%F\ %T) FAIL $0 $1 $2"'
echo "$(date +%F\ %T) GRID_END"
