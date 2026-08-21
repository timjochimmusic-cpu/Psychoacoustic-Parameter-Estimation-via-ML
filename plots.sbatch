#!/bin/bash
#SBATCH --job-name=plot_stats
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/plot-stats-%j.out
#SBATCH --error=logs/plot-stats-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"

srun python -m training_data_visualization.visualize_data_distribution
