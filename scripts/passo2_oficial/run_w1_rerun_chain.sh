#!/usr/bin/env bash
# W1 rerun (results/w1_rerun_global_eval/PLAN.md): waits for the P1 parity chain to end, runs the
# checks of PLAN §2 and, if they pass, the grid. No pause window (author, 2026-11-01).
# Usage: setsid bash scripts/passo2_oficial/run_w1_rerun_chain.sh > results/w1_rerun_global_eval/chain.log 2>&1
set -u
cd "$(dirname "$0")/../.."
OUT=results/w1_rerun_global_eval
P1=../fl-fedmed-p1/results/p1_parity
JLOG=results/janela_execucao.log
echo $$ > "$OUT/chain.pgid"
log() { echo "$(date +%F\ %T) $*"; echo "$(date +%F\ %T) W1 rerun: $*" >> "$JLOG"; }

log "waiting for the P1 parity chain (CHAIN_END, no process left)"
P1_PGID=$(cat "$P1/chain.pgid")
while ! grep -q CHAIN_END "$P1/chain.log" || pgrep -g "$P1_PGID" > /dev/null; do
  sleep 300
done

log "P1 chain ended; checks start (PLAN §2)"
if ! external/.venv_adaaggrl/bin/python scripts/passo2_oficial/run_w1.py verificar_global > "$OUT/verification.log" 2>&1; then
  log "CHECKS_FAILED: the grid was not started; see $OUT/verification.log"
  exit 1
fi
log "checks passed; grid starts (86 runs, 6 processes, PGID $$, no pause window)"
bash scripts/passo2_oficial/run_grid_w1.sh 6 > "$OUT/grid.log" 2>&1
log "grid ended: $(grep -c DONE "$OUT/grid.log")/86 done, $(grep -c FAIL "$OUT/grid.log") failed"
log "CHAIN_END"
