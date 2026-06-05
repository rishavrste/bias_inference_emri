#!/bin/bash
# Submit re-run jobs for all 0PA EMRI points with overlap < 0.97.
# Uses improved settings: temperature=100, seeds=100, larger DE/NM refinement.
# Each job warm-starts PARIS from the previous best point.
#
# Usage:
#   bash src/rerun_stuck.sh

STUCK=(1 2 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)

echo "Submitting ${#STUCK[@]} re-run jobs..."
for IDX in "${STUCK[@]}"; do
    JOB=$(qsub -v "GRID_START=${IDX},RUN_TYPE=0pa_vs_2pa" src/inference.pbs)
    echo "  pt ${IDX}: ${JOB}"
done
echo "Done."
