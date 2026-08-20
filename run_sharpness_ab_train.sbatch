#!/bin/bash
#SBATCH --job-name=sharpness-ab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sharpness-ab-%j.out
#SBATCH --error=logs/sharpness-ab-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"

experiment_dir="data/experiments/sharpness_ab_full"

srun python -m DL_model.compare_sharpness_training_modes \
  --mapping-csv "${experiment_dir}/segments/segment_mapping.csv" \
  --segment-dir "${experiment_dir}/segments" \
  --reference-dir "${experiment_dir}/references" \
  --contextual-targets "${experiment_dir}/hybrid_targets.pt" \
  --isolated-targets "${experiment_dir}/isolated_sharpness_targets.pt" \
  --output-dir "${experiment_dir}/ab_comparison_60" \
  --epochs 60 \
  --batch-size 8 \
  --learning-rate 0.001 \
  --seed 42
