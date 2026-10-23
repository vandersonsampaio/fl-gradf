#!/usr/bin/env bash
# W1 (results/w1_protocolo_independente_oficial/PLANO.md §3): MNIST (seeds 160-169) and BloodMNIST
# (170-179) × LMP/EB × td3/fixed, 500 rounds, + fedavg/none on 160-162 and 170-172 (86 runs).
# INTERLEAVED queue (seed × dataset × attack × condition). Resumable.
# Usage: setsid bash scripts/passo2_oficial/run_grid_w1.sh [n_procs] & ; then
#        bash scripts/janela_execucao.sh $(cat results/w1_protocolo_independente_oficial/raw/grid.pgid) W1 &
set -u
cd "$(dirname "$0")/../.."
P=${1:-6}
PY=external/.venv_adaaggrl/bin/python
OUT=results/w1_protocolo_independente_oficial/raw
mkdir -p "$OUT"
echo $$ > "$OUT/grid.pgid"
jobs() {
  for i in 0 1 2 3 4 5 6 7 8 9; do
    for d in MNIST BloodMNIST; do
      [ "$d" = MNIST ] && s=$((160 + i)) || s=$((170 + i))
      for a in EB LMP; do for c in td3 fixed; do
        [ -f "$OUT/${d}_${a}_q0.5_${c}_seed${s}_R500.json" ] || echo "$d $a $c $s"
      done; done
      if [ "$i" -lt 3 ]; then
        [ -f "$OUT/${d}_none_q0.5_fedavg_seed${s}_R500.json" ] || echo "$d none fedavg $s"
      fi
    done
  done
}
jobs | OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 xargs -P "$P" -L 1 bash -c \
  'echo "$(date +%F\ %T) START $0 $1 $2 $3"; '"$PY"' scripts/passo2_oficial/run_w1.py run --dataset $0 --attack $1 --condition $2 --seed $3 --out_dir '"$OUT"' > '"$OUT"'/$0_$1_$2_$3.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1 $2 $3" || echo "$(date +%F\ %T) FAIL $0 $1 $2 $3"'
echo "$(date +%F\ %T) GRID_END"
