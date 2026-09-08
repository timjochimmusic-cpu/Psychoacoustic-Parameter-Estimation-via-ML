#!/bin/bash
#SBATCH --job-name=music-references
#SBATCH --partition=standard
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=music-references-%j.out
#SBATCH --error=music-references-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "/beegfs/home/users/t/tjochim/projects/Psychoacoustic-Parameter-Estimation-via-ML/.venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/music-references-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
input="/beegfs/home/users/t/tjochim/projects/Psychoacoustic-Parameter-Estimation-via-ML/data/raw/music_dataset"
output="data/standardized_audio_files/music_references_${SLURM_JOB_ID}"
python main.py "$input" "$output" --full-workers 2 --segment-workers "${SLURM_CPUS_PER_TASK}"
