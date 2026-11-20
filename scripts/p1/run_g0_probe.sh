#!/usr/bin/env bash
# FLTrust server-step diagnostic (results/p1_cifar10/e0b/ADDENDUM3.md).
# MNIST: 3 alpha x 7 attacks x roots {100, 3000} x server steps
# {native, 20, 470, median} x seeds 0-2 (504 jobs); CIFAR-10: alpha 0.05 x
# {sign_flipping, trim_attack} x roots {100, 2500}, native, seed 0, 20 rounds
# (4 jobs). CPU, 1 thread per process.
# Usage: setsid bash scripts/p1/run_g0_probe.sh [N=10] > results/p1_g0_probe/grid.log 2>&1
set -u
cd "$(dirname "$0")/../.."
N=${1:-10}
mkdir -p results/p1_g0_probe/raw
echo $$ > results/p1_g0_probe/grid.pgid
export PY=../fl-fedmed/venv/bin/python CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

jobs() {
  for a in sign_flipping trim_attack; do for r in 100 2500; do
    echo "--dataset cifar10 --alpha 0.05 --attack $a --root_size $r --server_steps native --seed 0"
  done; done
  for s in 0 1 2; do for al in 0.5 0.1 0.05; do
    for a in fltrust_aligned trim_attack krum_collusion low_mag_backdoor sign_flipping gaussian_noise label_flipping; do
      for r in 100 3000; do for st in native 20 470 median; do
        echo "--dataset mnist --alpha $al --attack $a --root_size $r --server_steps $st --seed $s"
      done; done
    done
  done; done
}

jobs | xargs -P "$N" -L 1 bash -c 'echo "$(date +%F\ %T) START $*"; if $PY scripts/p1/g0_probe.py "$@" > /dev/null 2>&1; then echo "$(date +%F\ %T) DONE $*"; else echo "$(date +%F\ %T) FAIL $*"; fi' _
echo "$(date +%F\ %T) GRID_END"
