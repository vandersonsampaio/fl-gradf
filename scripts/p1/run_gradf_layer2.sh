#!/usr/bin/env bash
# GRADF Layer 2 split (results/p1_parity/gradf_layer2/PLAN.md): one job per
# seed 42-51. CPU, 1 thread per process. Resumable: jobs whose outputs exist are skipped.
# Usage: setsid bash scripts/p1/run_gradf_layer2.sh [n_parallel=10] > results/p1_parity/gradf_layer2/run.log 2>&1
set -u
cd "$(dirname "$0")/../.."
N=${1:-10}
OUT=results/p1_parity/gradf_layer2
mkdir -p "$OUT/raw"
echo $$ > "$OUT/run.pgid"
export PY=../fl-fedmed/venv/bin/python OUT
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

run_one() {
  s=$1; f="$OUT/raw/seed${s}.json"
  [ -f "$f" ] && { echo "$(date +%F\ %T) SKIP $s"; return 0; }
  while [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 30; done
  echo "$(date +%F\ %T) START $s"
  if $PY scripts/p1/gradf_layer2_split.py --seed "$s" --out "$f" > "$OUT/raw/seed${s}.log" 2>&1; then
    echo "$(date +%F\ %T) DONE $s"
  else
    echo "$(date +%F\ %T) FAIL $s"
  fi
}
export -f run_one

for s in 42 43 44 45 46 47 48 49 50 51; do echo "$s"; done \
  | xargs -P "$N" -L 1 bash -c 'run_one "$@"' _
echo "$(date +%F\ %T) GRID_END"
