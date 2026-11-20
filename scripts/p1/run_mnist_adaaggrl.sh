#!/usr/bin/env bash
# AdaAggRL on MNIST recording the trailing mean (results/p1_parity/mnist_trailing/ADDENDUM_2.md):
# both extractors, seeds 42-51, the settings of mnist_rq3. CPU, 1 thread per process. Resumable.
# Usage: setsid bash scripts/p1/run_mnist_adaaggrl.sh [n_parallel=2] > results/p1_parity/mnist_trailing/mnist_adaaggrl/run.log 2>&1
set -u
cd "$(dirname "$0")/../.."
N=${1:-2}
export OUT=results/p1_parity/mnist_trailing/mnist_adaaggrl PY=../fl-fedmed/venv/bin/python
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

run_one() {
  e=$1; s=$2; dir="$OUT/raw/${e}_seed${s}"
  [ -f "$dir/done" ] && { echo "$(date +%F\ %T) SKIP $*"; return 0; }
  mkdir -p "$dir"
  while [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 30; done
  echo "$(date +%F\ %T) START $*"
  if $PY -m src.experiments.exp10_selector_comparison --seeds "$s" --root_size 100 --variant b --systems adaaggrl \
       --adaaggrl_feature_extractor "$e" --seed_rng --server_steps median --out_dir "$dir" > "$dir/job.log" 2>&1; then
    touch "$dir/done"; echo "$(date +%F\ %T) DONE $*"
  else
    echo "$(date +%F\ %T) FAIL $*"
  fi
}
export -f run_one

for s in 42 43 44 45 46 47 48 49 50 51; do for e in random pretrained; do echo "$e $s"; done; done \
  | xargs -P "$N" -L 1 bash -c 'run_one "$@"' _
echo "$(date +%F\ %T) GRID_END"
