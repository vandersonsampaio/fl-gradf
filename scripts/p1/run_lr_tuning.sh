#!/usr/bin/env bash
# Global learning rate tuning (results/p1_parity/lr_tuning/ADDENDUM.md): 72 jobs, MNIST first.
# CPU, 1 thread per process. Resumable: jobs whose outputs exist are skipped.
# Usage: setsid bash scripts/p1/run_lr_tuning.sh [n_parallel=6] > results/p1_parity/lr_tuning/run.log 2>&1
set -u
cd "$(dirname "$0")/../.."
N=${1:-6}
OUT=results/p1_parity/lr_tuning
mkdir -p "$OUT/raw"
echo $$ > "$OUT/run.pgid"
export PY=../fl-fedmed/venv/bin/python OUT
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

run_one() {
  b=$1; u=$2; s=$3; d=$4; n="${b}_${u}_s${s}_f${d}"
  [ -f "$OUT/raw/$n.json" ] && { echo "$(date +%F\ %T) SKIP $n"; return 0; }
  while [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 30; done
  echo "$(date +%F\ %T) START $n"
  if $PY scripts/p1/lr_tuning.py --dataset "$b" --rule "$u" --seed "$s" --den "$d" > "$OUT/raw/$n.log" 2>&1; then
    echo "$(date +%F\ %T) DONE $n"
  else
    echo "$(date +%F\ %T) FAIL $n"
  fi
}
export -f run_one

for b in mnist cifar10; do
  for d in 1 3 5 10; do for u in fedavg trimmed_mean fltrust; do for s in 0 1 2; do echo "$b $u $s $d"; done; done; done \
    | xargs -P "$N" -L 1 bash -c 'run_one "$@"' _
  echo "$(date +%F\ %T) DATASET_END $b"
done
echo "$(date +%F\ %T) GRID_END"
