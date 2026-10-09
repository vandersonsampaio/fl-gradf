#!/usr/bin/env bash
# D2 (results/d2_variancia_p1/PLANO.md): 3 repetitions × seeds 42-44 of exp10 from tag p1.0.0, one new process per job.
# Usage: bash scripts/run_grid_d2.sh <worktree_of_tag_p1.0.0> [n_procs]. Resumable.
set -u
cd "$(dirname "$0")/.."
WT=${1:?worktree of tag p1.0.0}
P=${2:-9}
OUT=results/d2_variancia_p1/raw
mkdir -p "$OUT"
jobs() { for s in 42 43 44; do for r in 1 2 3; do [ -f "$OUT/rep${r}_seed${s}.csv" ] || echo "$s $r"; done; done; }
jobs | CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
  xargs -P "$P" -L 1 bash -c \
  'echo "$(date +%F\ %T) START $0 $1"; nice -n 19 '"$PWD"'/venv/bin/python '"$PWD"'/scripts/d2_variancia_p1.py run --seed $0 --rep $1 --worktree '"$WT"' > '"$OUT"'/rep$1_seed$0.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1" || echo "$(date +%F\ %T) FAIL $0 $1"'
echo "$(date +%F\ %T) GRID_END"
