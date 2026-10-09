#!/usr/bin/env bash
# B2.7 grid (results/b27_horizonte/PLANO.md). Resumable: skips jobs whose CSV already exists.
# 80 cell×seed jobs (150 rounds, 20 systems) + 30 α×seed ceiling jobs.
# Usage: bash scripts/run_grid_b27.sh [n_procs]
set -u
cd "$(dirname "$0")/.."
P=${1:-10}
OUT=results/b27_horizonte/raw
mkdir -p "$OUT"
jobs() {
  for s in 42 43 44 45 46 47 48 49 50 51; do
    for cell in "0.05 label_flipping" "0.1 label_flipping" "0.05 sign_flipping" "0.1 sign_flipping" \
                "0.05 gaussian_noise" "0.1 krum_collusion" "0.05 low_mag_backdoor" "0.5 trim_attack"; do
      set -- $cell
      [ -f "$OUT/celula_$2_a$1_seed${s}_R150.csv" ] || echo "celula $1 $2 $s"
    done
  done
  for s in 42 43 44 45 46 47 48 49 50 51; do for a in 0.05 0.1 0.5; do
    [ -f "$OUT/teto_a${a}_seed${s}_R150.csv" ] || echo "teto $a - $s"
  done; done
}
jobs | CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
  xargs -P "$P" -L 1 bash -c \
  'if [ "$0" = celula ]; then A="--alpha $1 --attack $2 --seed $3"; else A="--alpha $1 --seed $3"; fi; echo "$(date +%F\ %T) START $0 $1 $2 $3"; nice -n 19 venv/bin/python -m scripts.b27_horizonte $0 $A > '"$OUT"'/$0_$1_$2_$3.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1 $2 $3" || echo "$(date +%F\ %T) FAIL $0 $1 $2 $3"'
echo "$(date +%F\ %T) GRID_END"
