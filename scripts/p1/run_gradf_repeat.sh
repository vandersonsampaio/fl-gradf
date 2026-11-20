#!/usr/bin/env bash
# GRADF repetition (results/p1_parity/gradf_repeat/PLAN.md): 2 executions of
# seeds 42-51, the first execution first. CPU, 1 thread per process.
# Resumable: jobs whose outputs exist are skipped.
# Usage: setsid bash scripts/p1/run_gradf_repeat.sh [n_parallel=5] > results/p1_parity/gradf_repeat/run.log 2>&1
set -u
cd "$(dirname "$0")/../.."
N=${1:-5}
OUT=results/p1_parity/gradf_repeat
mkdir -p "$OUT/raw"
echo $$ > "$OUT/run.pgid"
export PY=../fl-fedmed/venv/bin/python OUT
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

run_one() {
  e=$1; s=$2; f="$OUT/raw/exec${e}_seed${s}.json"
  [ -f "$f" ] && { echo "$(date +%F\ %T) SKIP $e $s"; return 0; }
  while [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 30; done
  echo "$(date +%F\ %T) START $e $s"
  if $PY scripts/p1/gradf_repeat.py --seed "$s" --execution "$e" --out "$f" > "$OUT/raw/exec${e}_seed${s}.log" 2>&1; then
    echo "$(date +%F\ %T) DONE $e $s"
  else
    echo "$(date +%F\ %T) FAIL $e $s"
  fi
}
export -f run_one

for e in 1 2; do for s in 42 43 44 45 46 47 48 49 50 51; do echo "$e $s"; done; done \
  | xargs -P "$N" -L 1 bash -c 'run_one "$@"' _
echo "$(date +%F\ %T) GRID_END"
