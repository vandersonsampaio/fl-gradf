#!/usr/bin/env bash
# Confirmatory B2.6 grid (results/b26_decomposicao/PREREGISTRO.md). 80 jobs =
# 8 variants x seeds 52-61 (each job runs the 21 cells). CPU, low priority.
# Resumable: skips (variant, seed) whose CSV already exists.
# Usage: bash scripts/run_grid_b26.sh [--after-b28]
set -u
cd "$(dirname "$0")/.."
OUT=results/b26_decomposicao/raw
mkdir -p "$OUT"
if [ "${1:-}" = "--after-b28" ]; then
  echo "$(date +%F\ %T) WAIT B2.8 (10 CSVs)"
  until [ "$(ls results/b28_oraculo_por_regra/raw/seed*.csv 2>/dev/null | wc -l)" -ge 10 ]; do sleep 60; done
  echo "$(date +%F\ %T) B2.8 finished"
fi
for s in 52 53 54 55 56 57 58 59 60 61; do
  for v in sr_only sr_nomem sr_memof sr_bin cosserver_only sr_cosserver sr_b025 sr_b075; do
    [ -f "$OUT/${v}_seed${s}.csv" ] || echo "$v $s"
  done
done | CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  xargs -P 8 -L 1 bash -c 'echo "$(date +%F\ %T) START $0 $1"; nice -n 19 venv/bin/python -m scripts.b26_decomposicao run --variant $0 --seed $1 > '"$OUT"'/$0_$1.runner.log 2>&1 && echo "$(date +%F\ %T) DONE $0 $1" || echo "$(date +%F\ %T) FAIL $0 $1"'
echo "$(date +%F\ %T) GRID_END"
