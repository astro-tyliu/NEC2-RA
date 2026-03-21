#!/bin/bash
#SBATCH --job-name=power_sim
#SBATCH --output=logs/power_sim/%x_%A_%a.out
#SBATCH --error=logs/power_sim/%x_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --exclude=compute-103
#SBATCH --array=0-85

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Channel range: [154, 410]
# Corresponds to np.arange(ch_low, ch_high, 3)
# Single channel: START=154, STEP=3, array=0-0
# Multi-channel: START=154, STEP=3, array=0-85 processes 154,157,...,409
START=154
STEP=3

# Compute channel index from array task id
CH=$((START + SLURM_ARRAY_TASK_ID * STEP))

# Bounds check (inclusive of endpoint 410)
if [ "$CH" -gt 410 ]; then
    echo "CH=$CH > 410, exiting."
    exit 0
fi

echo "Running power_simulation with channel = $CH"

# ========== Environment setup ==========
source ~/venv/other/bin/activate
mkdir -p logs

# ========== Run ==========
cd "$(dirname "$0")"
python Tianyang.py "$CH"
