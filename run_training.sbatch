#!/bin/bash
#SBATCH --job-name=full-song-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=training-%j.out
#SBATCH --error=training-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source /beegfs/home/users/t/tjochim/projects/Psychoacoustic-Parameter-Estimation-via-ML/.venv/bin/activate
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/training-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
srun python -m DL_model.train_model \
  --run-dir data/standardized_audio_files/music_references_1923255 \
  --output-dir "DL_model/epochs/full_song_${SLURM_JOB_ID}" \
  --epochs "${1:-2}" --batch-size 128 --lr 0.001 --audio-workers 12 --require-cuda
