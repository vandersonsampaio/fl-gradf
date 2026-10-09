#!/usr/bin/env bash
# B3.3 (results/b33_limiar_bloodmnist/PLANO.md): BloodMNIST, EB, fixed action [0.475]*4 + [a5],
# a5 in {0.475 (center), 0.75, 0.95}, seeds 145-149, 500 rounds. Queue INTERLEAVED by seed.
# Usage: setsid bash scripts/passo2_oficial/run_grid_b33.sh [n_procs] & ; then
#      bash scripts/janela_execucao.sh $(cat results/b33_limiar_bloodmnist/raw/grid.pgid) B3.3 &  (optional execution window). Resumable.
set -u
cd "$(dirname "$0")/../.."
P=${1:-6}
PY=external/.venv_adaaggrl/bin/python
OUT=results/b33_limiar_bloodmnist/raw
mkdir -p "$OUT"
echo $$ > "$OUT/grid.pgid"  # PGID (launched with setsid) for the execution-window controller
jobs() {
  for s in 145 146 147 148 149; do for a5 in 0.475 0.75 0.95; do
    [ -f "$OUT/a5_${a5}/BloodMNIST_EB_q0.5_fixed_seed${s}_R500.json" ] || echo "$a5 $s"
  done; done
}
jobs | OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 xargs -P "$P" -L 1 bash -c \
  'echo "$(date +%F\ %T) START a5=$0 seed=$1"; '"$PY"' scripts/passo2_oficial/run_b33.py --a5 $0 --seed $1 > '"$OUT"'/a5_$0_seed$1.runner.log 2>&1 && echo "$(date +%F\ %T) DONE a5=$0 seed=$1" || echo "$(date +%F\ %T) FAIL a5=$0 seed=$1"'
echo "$(date +%F\ %T) GRID_END"
