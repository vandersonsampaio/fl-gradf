#!/usr/bin/env bash
# Common factor runs on CIFAR-10 (results/p1_parity/cifar_factor_rules/ADDENDUM.md): 42 jobs.
# CPU, 1 thread per process. Resumable: jobs whose outputs exist are skipped.
# Usage: setsid bash scripts/p1/run_cifar_factor_rules.sh [n_parallel=6] > results/p1_parity/cifar_factor_rules/run.log 2>&1
set -u
cd "$(dirname "$0")/../.."
N=${1:-6}
OUT=results/p1_parity/cifar_factor_rules
mkdir -p "$OUT/raw"
echo $$ > "$OUT/run.pgid"
export PY=../fl-fedmed/venv/bin/python OUT
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

run_one() {
  u=$1; a=$2; s=$3; d=$4; n="${u}_${a}_s${s}_f${d}"
  [ -f "$OUT/raw/$n.json" ] && { echo "$(date +%F\ %T) SKIP $n"; return 0; }
  while [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 30; done
  echo "$(date +%F\ %T) START $n"
  if $PY scripts/p1/cifar_factor_rules.py --rule "$u" --attack "$a" --seed "$s" --den "$d" > "$OUT/raw/$n.log" 2>&1; then
    echo "$(date +%F\ %T) DONE $n"
  else
    echo "$(date +%F\ %T) FAIL $n"
  fi
}
export -f run_one

{
  for a in sign_flipping trim_attack; do for s in 0 1 2; do echo "trimmed_mean $a $s 1"; done; done
  for d in 3 5 10; do for u in trimmed_mean fedavg; do for a in sign_flipping trim_attack; do for s in 0 1 2; do echo "$u $a $s $d"; done; done; done; done
} | xargs -P "$N" -L 1 bash -c 'run_one "$@"' _
echo "$(date +%F\ %T) GRID_END"
