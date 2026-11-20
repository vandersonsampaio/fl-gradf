#!/usr/bin/env bash
# MNIST rerun recording the trailing mean (results/p1_parity/mnist_trailing/ADDENDUM.md): the jobs of
# mnist_fixed and mnist_rq3, unchanged, in turn. CPU, 1 thread per process. Resumable: finished jobs are skipped.
# Usage: setsid bash scripts/p1/run_mnist_trailing.sh [n_parallel=6] > results/p1_parity/mnist_trailing/run.log 2>&1
set -u
cd "$(dirname "$0")/../.."
N=${1:-6}
OUT=results/p1_parity/mnist_trailing
echo $$ > "$OUT/run.pgid"
export PY=../fl-fedmed/venv/bin/python OUT
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

run_one() {
  case "$1" in
    fixed)
      root=$2; s=$3; dir="$OUT/mnist_fixed/raw/root${root}_seed${s}"
      rarg="--root_size $root"; [ "$root" = 3000 ] && rarg=""
      strat="fltrust median trimmed_mean krum"; [ "$root" = 100 ] && strat="fltrust fedavg fedprox median trimmed_mean krum clustering"
      cmd="$PY -m src.experiments.exp9_dominance_grid --seeds $s $rarg --seed_rng --server_steps median --strategies $strat --out_dir $dir" ;;
    rq3)
      v=$2; s=$3; dir="$OUT/mnist_rq3/raw/variant${v}_seed${s}"
      sys="random oracle gradf fedstrategist"; [ "$v" = a ] && sys="fedstrategist"
      cmd="$PY -m src.experiments.exp10_selector_comparison --seeds $s --root_size 100 --variant $v --systems $sys --seed_rng --server_steps median --out_dir $dir" ;;
  esac
  [ -f "$dir/done" ] && { echo "$(date +%F\ %T) SKIP $*"; return 0; }
  mkdir -p "$dir"
  while [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 30; done
  echo "$(date +%F\ %T) START $*"
  if $cmd > "$dir/job.log" 2>&1; then
    touch "$dir/done"; echo "$(date +%F\ %T) DONE $*"
  else
    echo "$(date +%F\ %T) FAIL $*"
  fi
}
export -f run_one

for s in 42 43 44 45 46 47 48 49 50 51; do for root in 100 3000 50; do echo "fixed $root $s"; done; done \
  | xargs -P "$N" -L 1 bash -c 'run_one "$@"' _
echo "$(date +%F\ %T) GROUP_END mnist_fixed"
for s in 42 43 44 45 46 47 48 49 50 51; do for v in b a; do echo "rq3 $v $s"; done; done \
  | xargs -P "$N" -L 1 bash -c 'run_one "$@"' _
echo "$(date +%F\ %T) GRID_END"
