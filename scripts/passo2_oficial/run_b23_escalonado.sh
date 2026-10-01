#!/usr/bin/env bash
# Escalonamento do B2.3 (substitui o disparo `run_grid_b23.sh --after-b21` em 2026-09-29).
# Mesmos 10 runs e o mesmo runner (run_b23.py); só muda a ordem/paralelismo:
#   - agora: 2 runs antecipados (LMP e EB, semente 100) nas vagas ociosas do fim do B2.1;
#   - após o GRID_END do B2.1: os 8 restantes com 4 em paralelo (6 no total com os antecipados).
# Retomável: pula runs cujo JSON final já existe. Ordem de execução não entra no desenho.
set -u
cd "$(dirname "$0")/../.."
PY=external/.venv_adaaggrl/bin/python
OUT=results/b23_steelman_oficial/raw
mkdir -p "$OUT"
one() {  # $1=ataque $2=semente
  echo "$(date +%F\ %T) START $1 steelman $2"
  if OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 $PY scripts/passo2_oficial/run_b23.py --attack "$1" --seed "$2" > "$OUT/$1_steelman_$2.runner.log" 2>&1; then
    echo "$(date +%F\ %T) DONE $1 steelman $2"
  else
    echo "$(date +%F\ %T) FAIL $1 steelman $2"
  fi
}
export -f one; export PY OUT
EARLY="LMP 100
EB 100"
while read -r a s; do
  [ -f "$OUT/MNIST_${a}_q0.5_td3_seed${s}_R500.json" ] || one "$a" "$s" &
done <<< "$EARLY"
echo "$(date +%F\ %T) WAIT b21 GRID_END (2 antecipados rodando)"
until grep -q GRID_END results/b21_replicacao_oficial/grid.log 2>/dev/null; do sleep 60; done
echo "$(date +%F\ %T) b21 terminou; iniciando os 8 restantes (-P 4)"
for s in 100 101 102 103 104; do for a in LMP EB; do
  [ "$s" = 100 ] && continue
  [ -f "$OUT/MNIST_${a}_q0.5_td3_seed${s}_R500.json" ] || echo "$a $s"
done; done | xargs -P 4 -L 1 bash -c 'one "$0" "$1"'
wait
echo "$(date +%F\ %T) GRID_END"
