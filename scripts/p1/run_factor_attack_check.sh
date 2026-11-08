#!/usr/bin/env bash
# Verification under attack on CIFAR-10 (results/p1_parity/factor_attack_check/ADDENDUM.md): 18 jobs.
# CPU, 1 thread per process. Resumable: jobs whose outputs exist are skipped.
# Usage: setsid bash scripts/p1/run_factor_attack_check.sh [n_parallel=6] > results/p1_parity/factor_attack_check/run.log 2>&1
set -u
cd "$(dirname "$0")/../.."
N=${1:-6}
OUT=results/p1_parity/factor_attack_check
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
  if $PY scripts/p1/factor_attack_check.py --rule "$u" --attack "$a" --seed "$s" --den "$d" > "$OUT/raw/$n.log" 2>&1; then
    echo "$(date +%F\ %T) DONE $n"
  else
    echo "$(date +%F\ %T) FAIL $n"
  fi
}
export -f run_one

for job in "trimmed_mean 1" "trimmed_mean 3" "fedavg 3"; do
  set -- $job
  for a in sign_flipping trim_attack; do for s in 0 1 2; do echo "$1 $a $s $2"; done; done
done | xargs -P "$N" -L 1 bash -c 'run_one "$@"' _
echo "$(date +%F\ %T) GRID_END"
