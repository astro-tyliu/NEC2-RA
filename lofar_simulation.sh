#!/bin/bash
#SBATCH --job-name=pynec_sim
#SBATCH --output=logs/lofar_sim/%x_%A_%a.out
#SBATCH --error=logs/lofar_sim/%x_%A_%a.err
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --exclude=compute-103
#SBATCH --array=0-1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# [154, 410]
START=169
STEP=3

# Compute parameter value from array task id
PARAM=$((START + SLURM_ARRAY_TASK_ID * STEP))

# Bounds check (inclusive of endpoint 410)
if [ "$PARAM" -gt 173 ]; then
    echo "PARAM=$PARAM > 410, exiting."
    exit 0
fi

echo "Running with PARAM = $PARAM"

# ========== Environment setup ==========
source ~/venv/other/bin/activate
mkdir -p logs

# ========== Run ==========
python /users/liutianyang/projects/other/NEC2-RA/Tianyang.py "$PARAM"

