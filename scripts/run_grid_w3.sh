#!/usr/bin/env bash
# W3 (results/w3_adaptativo_limiar/PLANO.md). Resumable: skips jobs whose CSV already exists.
#   gate:    fedavg + fixed_center × {adaptive, ipm2} × 3 α × seeds 125-129 (60 runs) for one rung, then the verdict
#   confirm: 4 systems × adaptive × 3 α × seeds 190-199 (120 runs) for the rung chosen in addendum 2
# Usage: bash scripts/run_grid_w3.sh gate|confirm <rung> [n_procs]
set -u
cd "$(dirname "$0")/.."
PHASE=${1:?gate|confirm}
RUNG=${2:?rung}
P=${3:-12}
if [ "$PHASE" = gate ]; then
  OUT=results/w3_adaptativo_limiar/raw_gate; SEEDS="125 126 127 128 129"; SYSTEMS="fixed_center fedavg"; ATTACKS="adaptive ipm2"; FLAG=--gate
else
  OUT=results/w3_adaptativo_limiar/raw; SEEDS="190 191 192 193 194 195 196 197 198 199"
  SYSTEMS="td3_ref fixed_center fixed_b025 fedavg"; ATTACKS="adaptive"; FLAG=
fi
mkdir -p "$OUT"
jobs() {
  for s in $SEEDS; do for a in 0.05 0.1 0.5; do for at in $ATTACKS; do for sys in $SYSTEMS; do
    [ -f "$OUT/${sys}_${at}_a${a}_seed${s}_rung${RUNG}_R150.csv" ] || echo "$sys $at $a $s"
  done; done; done; done
}
jobs | CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
  xargs -P "$P" -L 1 bash -c \
  'echo "$(date +%F\ %T) START $0 $1 $2 $3"; nice -n 19 venv/bin/python -m scripts.w3_adaptativo_limiar run --system $0 --attack $1 --alpha $2 --seed $3 --rung '"$RUNG $FLAG"' > '"$OUT"'/$0_$1_$2_$3.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1 $2 $3" || echo "$(date +%F\ %T) FAIL $0 $1 $2 $3"'
[ "$PHASE" = gate ] && venv/bin/python -m scripts.w3_adaptativo_limiar gate --rung "$RUNG"
echo "$(date +%F\ %T) GRID_END"
