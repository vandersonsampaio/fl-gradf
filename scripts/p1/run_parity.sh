#!/usr/bin/env bash
# P1 step-parity reruns (results/p1_parity/PLAN.md). CPU, 1 thread per process.
# Resumable: jobs whose outputs exist are skipped.
#   mnist_fixed  item 1: exp9 seeded, server_steps=median, seeds 42-51; root 100
#                with the 7 fixed rules, roots 3000 and 50 with fltrust/median/
#                trimmed_mean/krum (as results/p1_invariance)
#   mnist_rq3    item 2: exp10 seeded, server_steps=median, root 100, seeds 42-51;
#                variant b (Random, attack-aware, GRADF, FedStrategist) and
#                variant a (FedStrategist with GRADF's detector); no AdaAggRL
#   cifar_blockC item 3: E0b block C with server_steps=median (12 jobs, 180 rounds)
# Usage: setsid bash scripts/p1/run_parity.sh <mode> [n_parallel] > results/p1_parity/<mode>.log 2>&1
set -u
cd "$(dirname "$0")/../.."
MODE=$1; N=${2:-10}
OUT=results/p1_parity
mkdir -p "$OUT/$MODE/raw"
echo $$ > "$OUT/$MODE.pgid"
export PY=../fl-fedmed/venv/bin/python OUT MODE
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
SEEDS="42 43 44 45 46 47 48 49 50 51"

jobs() {
  case "$MODE" in
    mnist_fixed)
      for s in $SEEDS; do for root in 100 3000 50; do echo "$root $s"; done; done ;;
    mnist_rq3)
      for s in $SEEDS; do for v in b a; do echo "$v $s"; done; done ;;
    cifar_blockC)
      for s in 0 1 2; do for a in sign_flipping trim_attack; do for r in 100 2500; do
        echo "$a $r $s"; done; done; done ;;
  esac
}

# A CNN process peaks at ~1.9 GB (the 2026-10-29 OOM): start a job only with
# at least MIN_AVAIL_GB of RAM available.
MIN_AVAIL_GB=${MIN_AVAIL_GB:-4}
run_one() {
  while [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -lt "$MIN_AVAIL_GB" ]; do sleep 30; done
  case "$MODE" in
    mnist_fixed)
      root=$1; s=$2; dir="$OUT/$MODE/raw/root${root}_seed${s}"
      [ -f "$dir/done" ] && { echo "$(date +%F\ %T) SKIP $*"; return 0; }
      mkdir -p "$dir"
      rarg="--root_size $root"; [ "$root" = 3000 ] && rarg=""
      strat="fltrust median trimmed_mean krum"; [ "$root" = 100 ] && strat="fltrust fedavg fedprox median trimmed_mean krum clustering"
      cmd="$PY -m src.experiments.exp9_dominance_grid --seeds $s $rarg --seed_rng --server_steps median --strategies $strat --out_dir $dir" ;;
    mnist_rq3)
      v=$1; s=$2; dir="$OUT/$MODE/raw/variant${v}_seed${s}"
      [ -f "$dir/done" ] && { echo "$(date +%F\ %T) SKIP $*"; return 0; }
      mkdir -p "$dir"
      sys="random oracle gradf fedstrategist"; [ "$v" = a ] && sys="fedstrategist"
      cmd="$PY -m src.experiments.exp10_selector_comparison --seeds $s --root_size 100 --variant $v --systems $sys --seed_rng --server_steps median --out_dir $dir" ;;
    cifar_blockC)
      a=$1; r=$2; s=$3; dir="$OUT/$MODE/raw"
      [ -f "$dir/a0.05_${a}_fltrust_e2_s${s}_r${r}_stmedian.json" ] && { echo "$(date +%F\ %T) SKIP $*"; return 0; }
      cmd="$PY scripts/p1/e0_pilot.py --mode e0b --alpha 0.05 --attack $a --rule fltrust --root_size $r --seed $s --rounds 180 --server_steps median" ;;
  esac
  echo "$(date +%F\ %T) START $*"
  log="$OUT/$MODE/raw/$(echo "$*" | tr ' ' '_').log"
  if $cmd > "$log" 2>&1; then
    [ "$MODE" != cifar_blockC ] && touch "$dir/done"
    echo "$(date +%F\ %T) DONE $*"
  else
    echo "$(date +%F\ %T) FAIL $*"
  fi
}
export -f run_one
export MIN_AVAIL_GB

jobs | xargs -P "$N" -L 1 bash -c 'run_one "$@"' _
echo "$(date +%F\ %T) GRID_END"
