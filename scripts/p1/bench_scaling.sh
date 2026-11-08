#!/usr/bin/env bash
# E0 cost probe: k simultaneous CPU processes of bench_device.py with t
# threads each; per-process s/round and peak RSS go to
# results/p1_cifar10/e0_bench/scale_<system>_k<k>_t<t>_<i>.json.
# Usage: bash scripts/p1/bench_scaling.sh <system> <k> <t>
set -u
cd "$(dirname "$0")/../.."
SYS=$1; K=$2; T=$3
PY=../fl-fedmed/venv/bin/python
export CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=$T \
       TF_NUM_INTRAOP_THREADS=$T TF_NUM_INTEROP_THREADS=1
for i in $(seq 1 "$K"); do
  $PY scripts/p1/bench_device.py --tag "scale_${SYS}_k${K}_t${T}_${i}" --deterministic \
      --systems "$SYS" > "results/p1_cifar10/e0_bench/scale_${SYS}_k${K}_t${T}_${i}.log" 2>&1 &
done
wait
echo "$(date +%T) done $SYS k=$K t=$T"
