#!/usr/bin/env bash
# Starts the campaign once the TRIAGE dispatcher has drained; opens this PC's slots when the MNIST rerun ends.
cd /home/notebook/workspace-personal/fl-fedmed-p1-lr
R=results/p1_campaign
while kill -0 "$1" 2>/dev/null; do sleep 10; done
echo 0 > $R/dispatch.log.slots.local
../fl-fedmed/venv/bin/python scripts/p1/campaign_invariance.py --log $R/dispatch.log > $R/invariance.out 2>&1 &
( until grep -q GRID_END ../fl-fedmed-p1/results/p1_parity/mnist_trailing/run.log; do sleep 30; done; echo 6 > $R/dispatch.log.slots.local ) &
exec ../fl-fedmed/venv/bin/python scripts/p1/dispatch.py --jobs $R/jobs.txt --log $R/dispatch.log --results $R --project fl-fedmed-p1 --infra-retries 0 \
  --env 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3 PYTHONHASHSEED=0 ONEDNN_MAX_CPU_ISA=AVX2 OPENBLAS_CORETYPE=Haswell' \
  --host "vm2:4:$HOME/.ssh/experimentos-vm2_key.pem:experimentos@$VM2_HOST" \
  --host "vm3:4:$HOME/.ssh/experimentos-vm3_key.pem:experimentos@$VM3_HOST" \
  --host local:12
