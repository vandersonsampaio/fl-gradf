#!/usr/bin/env bash
# E0b part D cost probe (results/p1_cifar10/e0b/PLAN.md, Sec. 5): N identical
# probe processes at once (the campaign's load, 12 x 1 thread), then the
# aggregate into results/p1_cifar10/e0b/costs.json.
# Usage: setsid bash scripts/p1/run_cost_probe_d.sh [N=12] > results/p1_cifar10/e0b/cost_probe.log 2>&1
set -u
cd "$(dirname "$0")/../.."
N=${1:-12}
OUT=results/p1_cifar10/e0b/cost_probe
mkdir -p "$OUT"
echo $$ > "$OUT/cost_probe.pgid"
PY=../fl-fedmed/venv/bin/python
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3 \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

# The probe needs N processes of ~2 GB at the same time; refuse to start otherwise.
NEED=$((2 * N + 4))
AVAIL=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
if [ "$AVAIL" -lt "$NEED" ]; then
  echo "$(date +%F\ %T) ABORT: $AVAIL GB available, $NEED GB needed for $N processes"
  exit 1
fi

for i in $(seq 1 "$N"); do
  $PY scripts/p1/cost_probe_d.py --index "$i" > "$OUT/probe_$i.log" 2>&1 &
done
wait
$PY scripts/p1/cost_probe_d.py --aggregate
echo "$(date +%F\ %T) PROBE_END"
