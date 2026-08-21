#!/bin/bash
#SBATCH --job-name=full-prepare
#SBATCH --partition=standard
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/full-prepare-%j.out
#SBATCH --error=logs/full-prepare-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"

srun python -m data_preprocessing.prepare_sharpness_ab_experiment \
  --raw-dir data/raw/music_dataset \
  --experiment-dir data/experiments/full_recording_training \
  --full-recordings \
  --validation-fraction 0.2 \
  --seed 42
