#!/usr/bin/env bash
# P-1: seeded rerun of the four D_R rules on MNIST at roots 3000/100/50,
# seeds 42-51 (21 cells each), to test the trimmed-mean root invariant
# exactly and to recompute E1 with reproducible runs. CPU only.
# Usage: setsid bash scripts/p1/run_invariance_mnist.sh [n_parallel] > results/p1_invariance/grid.log 2>&1
set -u
cd "$(dirname "$0")/../.."
N=${1:-6}
OUT=results/p1_invariance
PY=../fl-fedmed/venv/bin/python
mkdir -p "$OUT/raw"
echo $$ > "$OUT/grid.pgid"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3

jobs() {
  for root in 3000 100 50; do
    for seed in 42 43 44 45 46 47 48 49 50 51; do
      echo "$root $seed"
    done
  done
}

run_one() {
  root=$1; seed=$2
  dir="$OUT/raw/root${root}_seed${seed}"
  if [ -f "$dir/done" ]; then echo "$(date +%F\ %T) SKIP $root $seed"; return 0; fi
  mkdir -p "$dir"
  rarg=""; [ "$root" != 3000 ] && rarg="--root_size $root"
  echo "$(date +%F\ %T) START $root $seed"
  if $PY -m src.experiments.exp9_dominance_grid --seeds "$seed" $rarg --seed_rng \
       --strategies fltrust median trimmed_mean krum --out_dir "$dir" > "$dir/run.log" 2>&1; then
    touch "$dir/done"; echo "$(date +%F\ %T) DONE $root $seed"
  else
    echo "$(date +%F\ %T) FAIL $root $seed"
  fi
}
export -f run_one
export OUT PY

jobs | xargs -P "$N" -L 1 bash -c 'run_one $0 $1'
echo "$(date +%F\ %T) GRID_END"
