#!/bin/bash
#SBATCH --job-name=reference-bench
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=reference-bench-%j.out
#SBATCH --error=reference-bench-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "/beegfs/home/users/t/tjochim/projects/Psychoacoustic-Parameter-Estimation-via-ML/.venv/bin/activate"
: "${1:?Pass a real, complete 180–300 second source recording}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
output="data/experiments/reference_benchmark_${SLURM_JOB_ID}"
mkdir -p "${output%/*}"
python benchmarks/reference_benchmark.py "$1" "$output" --workers "${SLURM_CPUS_PER_TASK}"
