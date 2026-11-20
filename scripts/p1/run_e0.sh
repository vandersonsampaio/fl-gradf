#!/usr/bin/env bash
# E0 pilot launcher (CPU, one thread per process). Resumable: jobs whose
# JSON exists are skipped by e0_pilot.py.
# Usage: setsid bash scripts/p1/run_e0.sh <separation|curves|e0b> [n_parallel] > results/p1_cifar10/e0/<mode>.log 2>&1
set -u
cd "$(dirname "$0")/../.."
MODE=$1; N=${2:-12}
ROUNDS=50; [ "$MODE" = e0b ] && ROUNDS=180
OUT=results/p1_cifar10/e0
mkdir -p "$OUT"
echo $$ > "$OUT/$MODE.pgid"
export PY=../fl-fedmed/venv/bin/python
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

jobs() {
  if [ "$MODE" = separation ]; then
    # E0.2: the 7 fixed rules at alpha = 0.05, one blind and one informed attack
    for attack in sign_flipping trim_attack; do
      for rule in fltrust median trimmed_mean krum fedavg fedprox clustering; do
        for seed in 0 1 2; do echo "--mode separation --alpha 0.05 --attack $attack --rule $rule --seed $seed"; done
      done
    done
  elif [ "$MODE" = e0b ]; then
    # E0b (results/p1_cifar10/e0b/PLAN.md): 180 rounds, 2 local epochs, seeds 0-2
    for seed in 0 1 2; do
      for alpha in 0.5 0.1 0.05; do echo "--mode e0b --alpha $alpha --seed $seed"; done
      for attack in sign_flipping trim_attack; do
        echo "--mode e0b --alpha 0.05 --attack $attack --rule fedavg --seed $seed"
        for root in 100 2500; do
          echo "--mode e0b --alpha 0.05 --attack $attack --rule fltrust --root_size $root --seed $seed"
        done
      done
    done
  else
    # E0.1: FedAvg without attack, three alphas, local epochs 1 and 2
    for alpha in 0.5 0.1 0.05; do
      for ep in 1 2; do
        for seed in 0 1 2; do echo "--mode curves --alpha $alpha --local_epochs $ep --seed $seed"; done
      done
    done
  fi
}

# Memory guard: a job starts only when at least MIN_AVAIL_GB of RAM is
# available (a CNN process peaks at ~1.9 GB; the 2026-10-29 OOM).
MIN_AVAIL_GB=${MIN_AVAIL_GB:-4}
run_one() {
  while [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -lt "$MIN_AVAIL_GB" ]; do sleep 30; done
  echo "$(date +%F\ %T) START $*"
  if $PY scripts/p1/e0_pilot.py "$@" --rounds "$ROUNDS" >> "$OUT/$MODE.jobs.log" 2>&1; then
    echo "$(date +%F\ %T) DONE $*"
  else
    echo "$(date +%F\ %T) FAIL $*"
  fi
}
export -f run_one
export OUT MODE MIN_AVAIL_GB ROUNDS

jobs | xargs -P "$N" -L 1 bash -c 'run_one "$@"' _
echo "$(date +%F\ %T) GRID_END"
