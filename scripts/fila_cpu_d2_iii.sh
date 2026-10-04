#!/usr/bin/env bash
# Fila da CPU (pedido do autor, 2026-10-04): espera o B2.8s terminar -> D2 -> C0b-iii.
set -u
cd "$(dirname "$0")/.."
WT=${1:?worktree da tag p1.0.0}
until grep -q GRID_END results/b28s_sensibilidade_metrica/raw/grid.log 2>/dev/null; do sleep 60; done
mkdir -p results/d2_variancia_p1/raw
bash scripts/run_grid_d2.sh "$WT" 9 > results/d2_variancia_p1/raw/grid.log 2>&1
mkdir -p results/c0b_iii_metrica/raw
bash scripts/run_grid_c0b_iii.sh 12 > results/c0b_iii_metrica/raw/grid.log 2>&1
