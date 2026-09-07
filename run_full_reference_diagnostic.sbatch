#!/bin/bash
#SBATCH --job-name=full-ref-diagnostic
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=full-ref-diagnostic-%j.out
#SBATCH --error=full-ref-diagnostic-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "/beegfs/home/users/t/tjochim/projects/Psychoacoustic-Parameter-Estimation-via-ML/.venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
input="${1:-data/experiments/reference_benchmark_1922355/full_wav}"
test -d "$input"
output="data/experiments/full_reference_diagnostic_${SLURM_JOB_ID}"
mkdir "$output"
for parameter in loudness_zwtv sharpness_din_tv; do
    echo "Starting ${parameter}: $(date -Is)"
    start_seconds=$SECONDS
    srun --ntasks=1 --job-name="$parameter" python data_preprocessing/calculate_reference_values.py \
        "$input" "$output/$parameter" --workers 1 --parameter "$parameter"
    echo "Completed ${parameter}: $((SECONDS - start_seconds)) seconds"
done
echo "Diagnostic passed: $output"
