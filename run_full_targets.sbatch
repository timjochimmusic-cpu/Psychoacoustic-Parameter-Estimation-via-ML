#!/bin/bash
#SBATCH --job-name=full-targets
#SBATCH --partition=standard
#SBATCH --array=0-11
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/full-targets-%A_%a.out
#SBATCH --error=logs/full-targets-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

experiment_dir="data/experiments/full_recording_training"
shard_dir="${experiment_dir}/target_shards"
mkdir -p "${shard_dir}"

srun python -m data_preprocessing.generate_hybrid_targets \
  --mapping-csv "${experiment_dir}/segments/segment_mapping.csv" \
  --segment-dir "${experiment_dir}/segments" \
  --reference-dir "${experiment_dir}/references" \
  --output "${shard_dir}/hybrid_targets_${SLURM_ARRAY_TASK_ID}.pt" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
  --checkpoint-every 25
