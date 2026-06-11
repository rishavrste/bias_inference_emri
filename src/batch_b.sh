#!/bin/bash
#PBS -q auto_free
#PBS -N 1pa_2pa_tail_13_25
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:mpiprocs=1:ompthreads=4:ngpus=1:mem=250gb
#PBS -o /scratch/e1583490/logs/pbs_output_b.log
#PBS -e /scratch/e1583490/logs/pbs_error_b.log
#PBS -k oed

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/scratch/e1583490/SuperKludege_Optimizations/IMRI_TAIL_1PA_with_Phase_4/logs_B_${TIMESTAMP}
mkdir -p "$LOG_DIR"

exec > "$LOG_DIR/paris_inference_${TIMESTAMP}.log" 2>&1

echo "Job started at $(date)"
echo "Running on $(hostname)"

cd /home/svu/e1583490/bias_inference_emri/src

export CUDA_VISIBLE_DEVICES=0
module load cuda12.4/toolkit/12.4.1
source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate

which python
python --version
nvidia-smi

export START_INDEX=14
export END_INDEX=25
echo "Running cases START_INDEX=$START_INDEX to END_INDEX=$END_INDEX"

python inference.py

echo "Job finished at $(date)"
