#!/usr/bin/env bash
# W2 (results/w2_backdoor_asr/PLANO.md). Resumable: skips jobs whose CSV already exists.
#   gate:   fedavg_clean + fedavg × 3 α × seeds 92-96 (30 runs) for one rung, then the gate verdict
#   confirm: 6 systems × 3 α × seeds 180-189 (180 runs) for the rung chosen in addendum 2
# Usage: bash scripts/run_grid_w2.sh gate|confirm <rung> [n_procs]
set -u
cd "$(dirname "$0")/.."
PHASE=${1:?gate|confirm}
RUNG=${2:?rung}
P=${3:-12}
if [ "$PHASE" = gate ]; then
  OUT=results/w2_backdoor_asr/raw_gate; SEEDS="92 93 94 95 96"; SYSTEMS="fedavg_clean fedavg"; FLAG=--gate
else
  OUT=results/w2_backdoor_asr/raw; SEEDS="180 181 182 183 184 185 186 187 188 189"
  SYSTEMS="td3_ref fixed_center fixed_b025 cosserver_only fedavg fedavg_clean"; FLAG=
fi
mkdir -p "$OUT"
jobs() {
  for s in $SEEDS; do for a in 0.05 0.1 0.5; do for sys in $SYSTEMS; do
    [ -f "$OUT/${sys}_a${a}_seed${s}_rung${RUNG}_R150.csv" ] || echo "$sys $a $s"
  done; done; done
}
jobs | CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 \
  xargs -P "$P" -L 1 bash -c \
  'echo "$(date +%F\ %T) START $0 $1 $2"; nice -n 19 venv/bin/python -m scripts.w2_backdoor_asr run --system $0 --alpha $1 --seed $2 --rung '"$RUNG $FLAG"' > '"$OUT"'/$0_$1_$2.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1 $2" || echo "$(date +%F\ %T) FAIL $0 $1 $2"'
[ "$PHASE" = gate ] && venv/bin/python -m scripts.w2_backdoor_asr gate --rung "$RUNG"
echo "$(date +%F\ %T) GRID_END"
