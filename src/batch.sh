#!/bin/bash
# ==============================================================================
# Example PBS batch script for running inference.py (or tests/check_overlaps.py)
# on a single GPU node.
#
# NOTE: this script is written for the NUS HPC "Hopper" cluster -- it assumes
# the PBS scheduler, the `module load singularity` environment, and the CUDA
# singularity image path shown below. It will need to be adapted (PBS
# directives, module/container setup) to run on any other cluster.
# ==============================================================================

#PBS -P <YOUR_PROJECT_CODE>          # replace with your PBS project/allocation code
#PBS -N <JOB_NAME>
#PBS -l walltime=48:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -k oed

# --- user-editable run parameters -------------------------------------------
REPO_DIR="${REPO_DIR:-$HOME/bias_inference_emri}"
SCRIPT="${SCRIPT:-inference.py}"     # inference.py, or tests/check_overlaps.py
START_INDEX="${START_INDEX:-0}"      # first case index to process (inclusive)
END_INDEX="${END_INDEX:-1}"          # last case index to process (exclusive)
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif}"
# ------------------------------------------------------------------------------

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${REPO_DIR}/src/logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/run_${TIMESTAMP}.log" 2>&1

echo "Job started at $(date)"
echo "Running on $(hostname)"

module load singularity
singularity exec --nv -e \
    "$SINGULARITY_IMAGE" \
    bash -lc "
        source '${REPO_DIR}/.venv/bin/activate'
        cd '${REPO_DIR}/src'
        START_INDEX=${START_INDEX} END_INDEX=${END_INDEX} python ${SCRIPT}
    "

echo "Job finished at $(date)"
