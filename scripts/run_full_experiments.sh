#!/usr/bin/env bash
#
# Launches exp1/exp2/exp4/exp7 in parallel, at production scale, to generate
# the final results for the conference paper (see
# references/PLANO_ARTIGO_CONFERENCIA.md). They are independent of each other
# and across seeds, so they run concurrently instead of sequentially. exp8 is
# chained to run automatically as soon as exp2 finishes, since it depends on
# the audit trails exp2 saves to results/audit_trail/.
#
# Usage:
#   ./scripts/run_full_experiments.sh
#   SEEDS="42 43 44" N_ROUNDS=15 ./scripts/run_full_experiments.sh   # overrides the defaults
#
# Run this when the machine is otherwise idle — it uses roughly 16-20
# concurrent threads; adjust the thread env vars below to match your machine.
#
# To survive the terminal closing, run with nohup:
#   nohup ./scripts/run_full_experiments.sh > results/logs/overall.log 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source venv/bin/activate

SEEDS="${SEEDS:-42 43 44 45 46}"
N_CLIENTS="${N_CLIENTS:-10}"
N_ROUNDS="${N_ROUNDS:-20}"
ATTACK_TYPES="${ATTACK_TYPES:-sign_flipping gaussian_noise label_flipping}"

# Avoids the concurrent processes contending for the same cores: 4 training
# processes (exp1/exp2/exp4/exp7) x 4 threads each ~= 16 cores used, leaving
# headroom for the OS. Adjust to match your machine's core count.
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

LOG_DIR="results/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Seeds: $SEEDS | n_clients=$N_CLIENTS n_rounds=$N_ROUNDS | attack_types: $ATTACK_TYPES"
echo "Logs at: $LOG_DIR"
echo ""

# exp2 is the heaviest (attack_types x byzantine_fractions x seeds) — launch
# it first. exp8 depends on the audit trails exp2 saves, so it is sequenced
# INSIDE the same subshell rather than awaited via an outer `wait $PID`:
# `wait` only works on direct children of the current shell, and a `wait`
# issued from inside a background subshell on exp2's PID (which is a child of
# the parent shell, not of the subshell) fails silently and would trigger
# exp8 immediately, without actually waiting for exp2 to finish.
(
    python -m src.experiments.exp2_robustness \
        --seeds $SEEDS --n_clients "$N_CLIENTS" --n_rounds "$N_ROUNDS" \
        > "$LOG_DIR/exp2_robustness.log" 2>&1
    EXIT_EXP2=$?
    if [ "$EXIT_EXP2" -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] exp2_robustness finished — running exp8_xai_examples..." > "$LOG_DIR/exp8_xai_examples.log"
        python -m src.experiments.exp8_xai_examples >> "$LOG_DIR/exp8_xai_examples.log" 2>&1
    else
        echo "exp2_robustness failed (exit=$EXIT_EXP2) — skipping exp8_xai_examples." > "$LOG_DIR/exp8_xai_examples.log"
        exit "$EXIT_EXP2"
    fi
) &
PID_EXP2_CHAIN=$!
echo "exp2_robustness+exp8   (PID $PID_EXP2_CHAIN) -> $LOG_DIR/exp2_robustness.log, $LOG_DIR/exp8_xai_examples.log"

python -m src.experiments.exp1_baseline \
    --seeds $SEEDS --n_clients "$N_CLIENTS" --n_rounds "$N_ROUNDS" --attack_types $ATTACK_TYPES \
    > "$LOG_DIR/exp1_baseline.log" 2>&1 &
PID_EXP1=$!
echo "exp1_baseline          (PID $PID_EXP1) -> $LOG_DIR/exp1_baseline.log"

python -m src.experiments.exp4_adaptive \
    --seeds $SEEDS --n_clients "$N_CLIENTS" --n_rounds "$N_ROUNDS" \
    > "$LOG_DIR/exp4_adaptive.log" 2>&1 &
PID_EXP4=$!
echo "exp4_adaptive          (PID $PID_EXP4) -> $LOG_DIR/exp4_adaptive.log"

python -m src.experiments.exp7_modality_ablation \
    --seeds $SEEDS --n_clients "$N_CLIENTS" --n_rounds "$N_ROUNDS" --attack_types $ATTACK_TYPES \
    > "$LOG_DIR/exp7_modality_ablation.log" 2>&1 &
PID_EXP7=$!
echo "exp7_modality_ablation (PID $PID_EXP7) -> $LOG_DIR/exp7_modality_ablation.log"

echo ""
echo "Follow along with: tail -f $LOG_DIR/*.log"
echo ""

FAILED=0
for entry in "exp1_baseline:$PID_EXP1" "exp2_robustness+exp8:$PID_EXP2_CHAIN" "exp4_adaptive:$PID_EXP4" "exp7_modality_ablation:$PID_EXP7"; do
    name="${entry%%:*}"
    pid="${entry##*:}"
    if wait "$pid"; then
        echo "[OK]     $name"
    else
        echo "[FAILED] $name — see logs in $LOG_DIR/"
        FAILED=1
    fi
done

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "=== All experiments completed successfully ==="
else
    echo "=== Completed with failures — review the logs above ==="
fi
echo "Results in results/tables/ and results/figures/"
echo "Full logs in $LOG_DIR/"

exit "$FAILED"
