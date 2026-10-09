#!/usr/bin/env bash
# C0b (results/c0b_espaco_h150/PLANO.md + ADENDO1.md): 19 cells × seeds 72-81 (8 systems, nested H 15/50/150)
# + 30 ceilings (3 α × 10 seeds). Resumable: skips jobs whose CSV already exists. Usage: bash scripts/run_grid_c0b.sh [n_procs]
set -u
cd "$(dirname "$0")/.."
P=${1:-12}
OUT=results/c0b_espaco_h150/raw
mkdir -p "$OUT"
jobs() {
  for s in 72 73 74 75 76 77 78 79 80 81; do
    for a in 0.5 0.1 0.05; do
      for t in fltrust_aligned trim_attack krum_collusion low_mag_backdoor sign_flipping gaussian_noise label_flipping; do
        if [ "$t" = fltrust_aligned ] && [ "$a" != 0.5 ]; then continue; fi
        [ -f "$OUT/celula_${t}_a${a}_seed${s}_R150.csv" ] || echo "celula $a $t $s"
      done
    done
  done
  for s in 72 73 74 75 76 77 78 79 80 81; do for a in 0.05 0.1 0.5; do
    [ -f "$OUT/teto_a${a}_seed${s}_R150.csv" ] || echo "teto $a - $s"
  done; done
}
jobs | CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
  xargs -P "$P" -L 1 bash -c \
  'if [ "$0" = celula ]; then A="--alpha $1 --attack $2 --seed $3"; else A="--alpha $1 --seed $3"; fi; echo "$(date +%F\ %T) START $0 $1 $2 $3"; nice -n 19 venv/bin/python -m scripts.c0b_espaco_h150 $0 $A > '"$OUT"'/$0_$1_$2_$3.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1 $2 $3" || echo "$(date +%F\ %T) FAIL $0 $1 $2 $3"'
echo "$(date +%F\ %T) GRID_END"
