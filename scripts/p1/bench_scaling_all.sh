#!/usr/bin/env bash
# Runs the E0 CPU scaling configurations in sequence, skipping any whose
# processes would not fit in the available RAM (1.5 GB per process).
# Usage: bash scripts/p1/bench_scaling_all.sh > results/p1_cifar10/e0_bench/scaling.log 2>&1
cd "$(dirname "$0")/../.."
for cfg in "trimmed_mean 4 1" "trimmed_mean 8 1" "trimmed_mean 12 1" "trimmed_mean 16 1" "trimmed_mean 8 2" \
           "adaaggrl 4 1" "adaaggrl 8 1" "adaaggrl 12 1" "adaaggrl 4 2" "adaaggrl 6 2" "adaaggrl 4 4"; do
  set -- $cfg
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  need=$(( $2 * 3 / 2 ))
  if [ "$avail" -lt "$need" ]; then echo "$(date +%T) SKIP $cfg (avail ${avail} GB < ${need} GB)"; continue; fi
  echo "$(date +%T) START $cfg (avail ${avail} GB)"
  bash scripts/p1/bench_scaling.sh "$@"
done
echo "$(date +%T) SCALING_END"
