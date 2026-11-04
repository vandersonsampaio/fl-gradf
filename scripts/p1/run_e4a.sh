#!/usr/bin/env bash
# E4a (results/p1_e4a/PLAN.md): one job per seed, seeds 42-51. CPU, 1 thread per process. Resumable.
# Usage: setsid bash scripts/p1/run_e4a.sh [n_parallel=2] > results/p1_e4a/run.log 2>&1
set -u
cd "$(dirname "$0")/../.."
N=${1:-2}
export OUT=results/p1_e4a PY=../fl-fedmed/venv/bin/python
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

run_one() {
  s=$1; dir="$OUT/raw/seed${s}"
  [ -f "$dir/done" ] && { echo "$(date +%F\ %T) SKIP $s"; return 0; }
  mkdir -p "$dir"
  while [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 30; done
  echo "$(date +%F\ %T) START $s"
  if $PY scripts/p1/e4a_phases.py run --seed "$s" --out "$dir" > "$dir/job.log" 2>&1; then
    touch "$dir/done"; echo "$(date +%F\ %T) DONE $s"
  else
    echo "$(date +%F\ %T) FAIL $s"
  fi
}
export -f run_one

seq 42 51 | xargs -P "$N" -L 1 bash -c 'run_one "$@"' _
echo "$(date +%F\ %T) GRID_END"
