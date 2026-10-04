#!/usr/bin/env bash
# B3.1 + B3.2 (results/b31_medmnist_oficial/PREREGISTRO.md §4 + ADENDO1.md): BloodMNIST, fixed × td3,
# ataques do Adendo 1, sementes 135-144, 500 rodadas. Fila INTERCALADA (td3 e fixed alternados por semente e ataque).
# Uso: setsid bash scripts/passo2_oficial/run_grid_b31.sh "LMP EB" [n_procs] & ; depois
#      bash scripts/janela_execucao.sh $(cat results/b31_medmnist_oficial/raw/grid.pgid) B3.1 &  (nada roda 7h-18h seg-sex). Retomável.
set -u
cd "$(dirname "$0")/../.."
ATAQUES=${1:?ataques do Adendo 1}
P=${2:-6}
PY=external/.venv_adaaggrl/bin/python
OUT=results/b31_medmnist_oficial/raw
mkdir -p "$OUT"
echo $$ > "$OUT/grid.pgid"  # PGID (lançado com setsid) para o controlador da janela (scripts/janela_execucao.sh)
jobs() {
  for s in 135 136 137 138 139 140 141 142 143 144; do for a in $ATAQUES; do for c in td3 fixed; do
    [ -f "$OUT/BloodMNIST_${a}_q0.5_${c}_seed${s}_R500.json" ] || echo "$a $c $s"
  done; done; done
}
jobs | OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 xargs -P "$P" -L 1 bash -c \
  'echo "$(date +%F\ %T) START $0 $1 $2"; '"$PY"' scripts/passo2_oficial/run_b3.py --dataset BloodMNIST --attack $0 --condition $1 --seed $2 --out_dir '"$OUT"' > '"$OUT"'/$0_$1_$2.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1 $2" || echo "$(date +%F\ %T) FAIL $0 $1 $2"'
echo "$(date +%F\ %T) GRID_END"
