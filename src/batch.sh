#!/bin/bash
#PBS -q auto_free
#PBS -N 1pa_vs_2pa_IMRI_tail
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:mpiprocs=1:ompthreads=4:ngpus=1:mem=250gb
#PBS -o /scratch/e1583490/logs/pbs_output.log
#PBS -e /scratch/e1583490/logs/pbs_error.log
#PBS -k oed

set -euo pipefail   # Keep safety flags, but drop set -x

# ONE timestamp used everywhere
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_3/logs_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

# Redirect everything ONCE, early, before any real work
exec > "$LOG_DIR/paris_inference_${TIMESTAMP}.log" 2>&1

# NOW enable -x if you want it — trace goes cleanly into your log
# set -x   # ← uncomment only when actively debugging a crash

echo "Job started at $(date)"
echo "Running on $(hostname)"

cd /home/svu/e1583490/bias_inference_emri/src

export CUDA_VISIBLE_DEVICES=0
module load cuda12.4/toolkit/12.4.1
source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate

which python
python --version
nvidia-smi
echo "$CUDA_VISIBLE_DEVICES"

python inference.py

echo "Job finished at $(date)"