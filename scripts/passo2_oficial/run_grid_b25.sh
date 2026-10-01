#!/usr/bin/env bash
# B2.5 (results/b25_ipm_oficial/PLANO.md). Retomável: pula runs cujo JSON já existe.
#   sanity:  5 runs, semente 100, 100 rodadas (fedavg sem ataque; fedavg e fixed com ε ∈ {2, 10})
#   grade:   20 runs, sementes 105-114 x {td3, fixed}, 500 rodadas, ε escolhido no sanity
# Uso: bash scripts/passo2_oficial/run_grid_b25.sh sanity
#      bash scripts/passo2_oficial/run_grid_b25.sh grade <eps>
set -u
cd "$(dirname "$0")/../.."
PY=external/.venv_adaaggrl/bin/python
MODE=${1:?modo: sanity | grade <eps>}
if [ "$MODE" = sanity ]; then
  OUT=results/b25_ipm_oficial/sanity; R=100
  jobs() {
    for spec in "none - fedavg" "IPM_real 2 fedavg" "IPM_real 10 fedavg" "IPM_real 2 fixed" "IPM_real 10 fixed"; do
      set -- $spec
      lab=$([ "$1" = none ] && echo none || echo "IPMr$2")
      [ -f "$OUT/MNIST_${lab}_q0.5_$3_seed100_R$R.json" ] || echo "$1 $2 $3 100"
    done
  }
  P=5
else
  EPS=${2:?ε obrigatório}
  OUT=results/b25_ipm_oficial/raw; R=500
  jobs() {
    for s in 105 106 107 108 109 110 111 112 113 114; do for c in td3 fixed; do
      [ -f "$OUT/MNIST_IPMr${EPS}_q0.5_${c}_seed${s}_R$R.json" ] || echo "IPM_real $EPS $c $s"
    done; done
  }
  P=5
fi
mkdir -p "$OUT"
jobs | OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 xargs -P $P -L 1 bash -c \
  'E=$([ "$1" = - ] && echo "" || echo "--eps $1"); echo "$(date +%F\ %T) START $0 $1 $2 $3"; '"$PY"' scripts/passo2_oficial/run_b25.py --attack $0 $E --condition $2 --seed $3 --rounds '"$R"' --out_dir '"$OUT"' > '"$OUT"'/$0_$1_$2_$3.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1 $2 $3" || echo "$(date +%F\ %T) FAIL $0 $1 $2 $3"'
echo "$(date +%F\ %T) GRID_END"
