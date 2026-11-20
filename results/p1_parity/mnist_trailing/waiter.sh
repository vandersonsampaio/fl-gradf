#!/usr/bin/env bash
# Starts the MNIST rerun once the verification under attack has ended.
cd "$(dirname "$0")/../../.."
while ! grep -q GRID_END results/p1_parity/factor_attack_check/run.log; do sleep 30; done
exec bash scripts/p1/run_mnist_trailing.sh 6 > results/p1_parity/mnist_trailing/run.log 2>&1
