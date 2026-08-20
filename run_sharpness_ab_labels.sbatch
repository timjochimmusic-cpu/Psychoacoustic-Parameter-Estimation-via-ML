#!/bin/bash
#SBATCH --job-name=sharpness-labels
#SBATCH --partition=standard
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sharpness-labels-%j.out
#SBATCH --error=logs/sharpness-labels-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"

experiment_dir="data/experiments/sharpness_ab_full"
mapping_csv="${experiment_dir}/segments/segment_mapping.csv"
segment_dir="${experiment_dir}/segments"
reference_dir="${experiment_dir}/references"
segment_starts=(5000 15000 30000 45000)

srun python -m data_preprocessing.prepare_sharpness_ab_experiment \
  --raw-dir data/raw/music_dataset \
  --experiment-dir "${experiment_dir}" \
  --reference-duration-s 60 \
  --validation-fraction 0.2 \
  --seed 42

srun python -m data_preprocessing.generate_hybrid_targets \
  --mapping-csv "${mapping_csv}" \
  --segment-dir "${segment_dir}" \
  --reference-dir "${reference_dir}" \
  --output "${experiment_dir}/hybrid_targets.pt" \
  --segment-starts-ms "${segment_starts[@]}" \
  --checkpoint-every 1

srun python -m data_preprocessing.generate_isolated_sharpness_targets \
  --mapping-csv "${mapping_csv}" \
  --segment-dir "${segment_dir}" \
  --output "${experiment_dir}/isolated_sharpness_targets.pt" \
  --segment-starts-ms "${segment_starts[@]}"
