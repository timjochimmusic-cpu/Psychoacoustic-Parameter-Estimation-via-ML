#!/bin/bash
#SBATCH --job-name=full-merge
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/full-merge-%j.out
#SBATCH --error=logs/full-merge-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

experiment_dir="data/experiments/full_recording_training"

srun python -m data_preprocessing.merge_target_shards \
  --shards "${experiment_dir}"/target_shards/hybrid_targets_*.pt \
  --output "${experiment_dir}/hybrid_targets.pt"

srun python -m training_data_visualization.summarize_consolidated_targets \
  --mapping-csv "${experiment_dir}/segments/segment_mapping.csv" \
  --targets "${experiment_dir}/hybrid_targets.pt" \
  --output-dir "${experiment_dir}/training_statistics" \
  --split train
