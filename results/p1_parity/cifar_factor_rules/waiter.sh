#!/usr/bin/env bash
cd /home/notebook/workspace-personal/fl-fedmed-p1
while ! grep -q GRID_END results/p1_parity/cifar_factor_sweep/run.log; do sleep 30; done
echo "$(date +%F\ %T) sweep ended; launching" > results/p1_parity/cifar_factor_rules/waiter.log
exec bash scripts/p1/run_cifar_factor_rules.sh 6 > results/p1_parity/cifar_factor_rules/run.log 2>&1
