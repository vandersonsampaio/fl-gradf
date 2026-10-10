#!/usr/bin/env bash
# B2.8s (results/b28s_sensibilidade_metrica/PLANO.md): 21 cells × seeds 42-51 (9 systems per job). Resumable.
# Usage: bash scripts/run_grid_b28s.sh [n_procs]
set -u
cd "$(dirname "$0")/.."
P=${1:-10}
OUT=results/b28s_sensibilidade_metrica/raw
mkdir -p "$OUT"
jobs() {
  for s in 42 43 44 45 46 47 48 49 50 51; do for a in 0.5 0.1 0.05; do
    for t in fltrust_aligned trim_attack krum_collusion low_mag_backdoor sign_flipping gaussian_noise label_flipping; do
      [ -f "$OUT/celula_${t}_a${a}_seed${s}.csv" ] || echo "$a $t $s"
    done; done; done
}
jobs | CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
  xargs -P "$P" -L 1 bash -c \
  'echo "$(date +%F\ %T) START $0 $1 $2"; nice -n 19 venv/bin/python -m scripts.b28s_sensibilidade_metrica celula --alpha $0 --attack $1 --seed $2 > '"$OUT"'/$1_$0_$2.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1 $2" || echo "$(date +%F\ %T) FAIL $0 $1 $2"'
echo "$(date +%F\ %T) GRID_END"
