#!/usr/bin/env bash
# Runs the P1 step-parity items (results/p1_parity/PLAN.md) after W1 finishes,
# in this order: items 1 and 3 together, then item 2, then item 4 (part D,
# alone, since it measures costs under the campaign's load).
# W1 is finished when its 86 final JSONs exist and no W1 process is alive.
# Authorized by the author on 2026-10-30 to start as soon as W1 finishes,
# inside the Mon-Fri 8-18 window if need be.
# Usage: setsid bash scripts/p1/run_parity_chain.sh > results/p1_parity/chain.log 2>&1
set -u
cd "$(dirname "$0")/../.."
W1=../fl-fedmed/results/w1_protocolo_independente_oficial/raw
JLOG=../fl-fedmed/results/janela_execucao.log
mkdir -p results/p1_parity
echo $$ > results/p1_parity/chain.pgid
log() { echo "$(date +%F\ %T) $*"; echo "$(date +%F\ %T) P1 parity chain: $*" >> "$JLOG"; }

log "waiting for W1 (86 final JSONs, no W1 process)"
while [ "$(ls "$W1"/*_R500.json 2>/dev/null | wc -l)" -lt 86 ] || pgrep -f "run_w1.py|run_grid_w1" > /dev/null; do
  sleep 300
done
log "W1 finished; items 1 (mnist_fixed, 8 processes) and 3 (cifar_blockC, 12 processes) start (PGID $$)"

bash scripts/p1/run_parity.sh cifar_blockC 12 > results/p1_parity/cifar_blockC.log 2>&1 &
bash scripts/p1/run_parity.sh mnist_fixed 8 > results/p1_parity/mnist_fixed.log 2>&1 &
wait
log "items 1 and 3 ended: $(grep -c DONE results/p1_parity/mnist_fixed.log)/30 and $(grep -c DONE results/p1_parity/cifar_blockC.log)/12 done, $(cat results/p1_parity/mnist_fixed.log results/p1_parity/cifar_blockC.log | grep -c FAIL) failed"

log "item 2 (mnist_rq3, 10 processes) starts"
bash scripts/p1/run_parity.sh mnist_rq3 10 > results/p1_parity/mnist_rq3.log 2>&1
log "item 2 ended: $(grep -c DONE results/p1_parity/mnist_rq3.log)/20 done, $(grep -c FAIL results/p1_parity/mnist_rq3.log) failed"

log "item 4 (part D cost probe, 12 processes, alone) starts"
bash scripts/p1/run_cost_probe_d.sh 12 > results/p1_cifar10/e0b/cost_probe.log 2>&1
log "item 4 ended: $(tail -1 results/p1_cifar10/e0b/cost_probe.log)"
log "CHAIN_END"
