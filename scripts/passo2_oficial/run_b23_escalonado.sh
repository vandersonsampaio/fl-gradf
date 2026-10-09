#!/usr/bin/env bash
# Alternative B2.3 scheduler (used instead of `run_grid_b23.sh --after-b21`).
# Same 10 runs and the same runner (run_b23.py); only the order/parallelism changes:
#   - immediately: 2 runs started early (LMP and EB, seed 100) in the idle GPU slots while B2.1 finishes;
#   - after the B2.1 GRID_END: the remaining 8 with 4 in parallel (6 in total with the early ones).
# Resumable: skips runs whose final JSON already exists. Execution order is not part of the design.
set -u
cd "$(dirname "$0")/../.."
PY=external/.venv_adaaggrl/bin/python
OUT=results/b23_steelman_oficial/raw
mkdir -p "$OUT"
one() {  # $1=attack $2=seed
  echo "$(date +%F\ %T) START $1 steelman $2"
  if OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 $PY scripts/passo2_oficial/run_b23.py --attack "$1" --seed "$2" > "$OUT/$1_steelman_$2.runner.log" 2>&1; then
    echo "$(date +%F\ %T) DONE $1 steelman $2"
  else
    echo "$(date +%F\ %T) FAIL $1 steelman $2"
  fi
}
export -f one; export PY OUT
EARLY="LMP 100
EB 100"
while read -r a s; do
  [ -f "$OUT/MNIST_${a}_q0.5_td3_seed${s}_R500.json" ] || one "$a" "$s" &
done <<< "$EARLY"
echo "$(date +%F\ %T) WAIT b21 GRID_END (2 early runs in progress)"
until grep -q GRID_END results/b21_replicacao_oficial/grid.log 2>/dev/null; do sleep 60; done
echo "$(date +%F\ %T) b21 finished; starting the remaining 8 (-P 4)"
for s in 100 101 102 103 104; do for a in LMP EB; do
  [ "$s" = 100 ] && continue
  [ -f "$OUT/MNIST_${a}_q0.5_td3_seed${s}_R500.json" ] || echo "$a $s"
done; done | xargs -P 4 -L 1 bash -c 'one "$0" "$1"'
wait
echo "$(date +%F\ %T) GRID_END"
