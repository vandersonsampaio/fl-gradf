#!/usr/bin/env bash
# CIFAR-10 step pilot (results/p1_parity/cifar_step_pilot/ADDENDUM.md): 12 jobs.
# CPU, 1 thread per process. Resumable: jobs whose outputs exist are skipped.
# Usage: setsid bash scripts/p1/run_cifar_step_pilot.sh [n_parallel=5] > results/p1_parity/cifar_step_pilot/run.log 2>&1
set -u
cd "$(dirname "$0")/../.."
N=${1:-5}
OUT=results/p1_parity/cifar_step_pilot
mkdir -p "$OUT/raw"
echo $$ > "$OUT/run.pgid"
export PY=../fl-fedmed/venv/bin/python OUT
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

run_one() {
  a=$1; r=$2; s=$3; n="${a}_r${r}_s${s}"
  [ -f "$OUT/raw/$n.json" ] && { echo "$(date +%F\ %T) SKIP $n"; return 0; }
  while [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 30; done
  echo "$(date +%F\ %T) START $n"
  if $PY scripts/p1/cifar_step_pilot.py --attack "$a" --root_size "$r" --seed "$s" > "$OUT/raw/$n.log" 2>&1; then
    echo "$(date +%F\ %T) DONE $n"
  else
    echo "$(date +%F\ %T) FAIL $n"
  fi
}
export -f run_one

for a in sign_flipping trim_attack; do for r in 100 2500; do for s in 0 1 2; do echo "$a $r $s"; done; done; done \
  | xargs -P "$N" -L 1 bash -c 'run_one "$@"' _
echo "$(date +%F\ %T) GRID_END"
