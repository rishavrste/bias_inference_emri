#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -N paris_inference
#PBS -l walltime=15:00:00
#PBS -l select=1:ngpus=1
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -k oed

# Define log directory and create it if it doesn't exist
LOG_DIR=/scratch/e1583490/emri_with_noise_dev_a_1_batch_sigma_75/logs_8
mkdir -p "$LOG_DIR"

# Redirect all stdout and stderr to our chosen files from this point on
exec > "$LOG_DIR/paris_inference.out" 2>"$LOG_DIR/paris_inference.err"

cd /home/svu/e1583490/bias_inference_emri/src
module load singularity
singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc '
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/bias_inference_emri/src
    python inference.py
'
