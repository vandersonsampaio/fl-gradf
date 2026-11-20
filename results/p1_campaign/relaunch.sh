#!/usr/bin/env bash
# Restarts the campaign's dispatcher with one repeat after a lost connection (ADDENDUM_1.md), once the running one has drained.
cd /home/notebook/workspace-personal/fl-fedmed-p1-lr
R=results/p1_campaign
while kill -0 "$1" 2>/dev/null; do sleep 5; done
if grep -q INVARIANCE_FAIL $R/invariance.txt 2>/dev/null; then echo "not restarted: invariance failed"; exit 1; fi
rm -f $R/dispatch.log.stop
exec ../fl-fedmed/venv/bin/python scripts/p1/dispatch.py --jobs $R/jobs.txt --log $R/dispatch.log --results $R --project fl-fedmed-p1 --infra-retries 1 \
  --env 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3 PYTHONHASHSEED=0 ONEDNN_MAX_CPU_ISA=AVX2 OPENBLAS_CORETYPE=Haswell' \
  --host "vm2:4:$HOME/.ssh/experimentos-vm2_key.pem:experimentos@$VM2_HOST" \
  --host "vm3:4:$HOME/.ssh/experimentos-vm3_key.pem:experimentos@$VM3_HOST" \
  --host local:12
