#!/usr/bin/env bash
# B2.7c reformulated (results/b27c_b025_vs_td3/PLANO.md): 2 systems × 3 cells × seeds 82-91, H = 150 (60 runs).
# Resumable: skips jobs whose CSV already exists. Usage: bash scripts/run_grid_b27c.sh [n_procs]
set -u
cd "$(dirname "$0")/.."
P=${1:-12}
OUT=results/b27c_b025_vs_td3/raw
mkdir -p "$OUT"
jobs() {
  for s in 82 83 84 85 86 87 88 89 90 91; do
    for cell in "0.05 label_flipping" "0.1 label_flipping" "0.05 low_mag_backdoor"; do
      for sys in td3_ref fixed_b025; do
        set -- $cell
        [ -f "$OUT/${sys}_$2_a$1_seed${s}_R150.csv" ] || echo "$1 $2 $s $sys"
      done
    done
  done
}
jobs | CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
  xargs -P "$P" -L 1 bash -c \
  'echo "$(date +%F\ %T) START $0 $1 $2 $3"; nice -n 19 venv/bin/python -m scripts.b27c_b025_vs_td3 run --alpha $0 --attack $1 --seed $2 --system $3 > '"$OUT"'/$3_$0_$1_$2.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1 $2 $3" || echo "$(date +%F\ %T) FAIL $0 $1 $2 $3"'
echo "$(date +%F\ %T) GRID_END"
