#!/bin/bash
#SBATCH --job-name=mapped-baseline
#SBATCH --partition=standard
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mapped-baseline-%j.out
#SBATCH --error=logs/mapped-baseline-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

experiment_dir="data/experiments/sharpness_ab_full"
mapping_csv="${experiment_dir}/segments/segment_mapping.csv"
segment_dir="${experiment_dir}/segments"
targets="${experiment_dir}/hybrid_targets.pt"
statistics_dir="${experiment_dir}/training_statistics"
training_dir="${experiment_dir}/mapped_training"

srun python -m training_data_visualization.summarize_consolidated_targets \
  --mapping-csv "${mapping_csv}" \
  --targets "${targets}" \
  --output-dir "${statistics_dir}" \
  --split train

srun python -m DL_model.train_mapped_model \
  --mapping-csv "${mapping_csv}" \
  --segment-dir "${segment_dir}" \
  --targets "${targets}" \
  --statistics-dir "${statistics_dir}" \
  --output-dir "${training_dir}" \
  --run-name baseline_lr1e-3_scheduler_p10 \
  --epochs 100 \
  --learning-rate 0.001 \
  --batch-size 128 \
  --scheduler-patience 10 \
  --scheduler-factor 0.5 \
  --seed 42
