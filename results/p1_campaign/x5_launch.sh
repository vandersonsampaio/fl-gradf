#!/usr/bin/env bash
# Starts X5 (ADDENDUM_3.md) on this PC and the two VMs; VM2_HOST and VM3_HOST hold the VMs' addresses.
cd /home/notebook/workspace-personal/fl-fedmed-p1-lr
R=results/p1_campaign
: "${VM2_HOST:?set VM2_HOST}" "${VM3_HOST:?set VM3_HOST}"
exec ../fl-fedmed/venv/bin/python scripts/p1/dispatch.py --jobs $R/x5_jobs.txt --log $R/x5_dispatch.log --results $R --project fl-fedmed-p1 --infra-retries 1 \
  --env 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3 PYTHONHASHSEED=0 ONEDNN_MAX_CPU_ISA=AVX2 OPENBLAS_CORETYPE=Haswell' \
  --host "vm2:4:$HOME/.ssh/experimentos-vm2_key.pem:experimentos@$VM2_HOST" \
  --host "vm3:4:$HOME/.ssh/experimentos-vm3_key.pem:experimentos@$VM3_HOST" \
  --host local:8
