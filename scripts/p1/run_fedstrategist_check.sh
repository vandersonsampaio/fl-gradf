#!/usr/bin/env bash
# FedStrategist's unexplained mismatch on MNIST (results/p1_parity/mnist_trailing/fedstrategist_check/PLAN.md):
# seed 45, the arm alone twice and after GRADF once.
# Usage: setsid bash scripts/p1/run_fedstrategist_check.sh > results/p1_parity/mnist_trailing/fedstrategist_check/run.log 2>&1
set -u
cd "$(dirname "$0")/../.."
OUT=results/p1_parity/mnist_trailing/fedstrategist_check
PY=../fl-fedmed/venv/bin/python
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
run() {
  dir="$OUT/raw/$1"; shift; mkdir -p "$dir"
  echo "$(date +%F\ %T) START $dir"
  if $PY -m src.experiments.exp10_selector_comparison --seeds 45 --root_size 100 --variant b --systems "$@" \
       --seed_rng --server_steps median --out_dir "$dir" > "$dir/job.log" 2>&1; then
    echo "$(date +%F\ %T) DONE $dir"
  else
    echo "$(date +%F\ %T) FAIL $dir"
  fi
}
run alone_1 fedstrategist &
run alone_2 fedstrategist &
run after_gradf gradf fedstrategist &
wait
echo "$(date +%F\ %T) GRID_END"
