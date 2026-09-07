#!/bin/bash
#SBATCH --job-name=workflow-smoke
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=workflow-smoke-%j.out
#SBATCH --error=workflow-smoke-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source "/beegfs/home/users/t/tjochim/projects/Psychoacoustic-Parameter-Estimation-via-ML/.venv/bin/activate"

test_dir="data/experiments/workflow_smoke_test_${SLURM_JOB_ID}"
input_dir="${test_dir}/input"
full_wav_dir="${test_dir}/full_wav"
segment_dir="${test_dir}/segments"
full_label_dir="${test_dir}/full_labels"
segment_label_dir="${test_dir}/segment_labels"
combined_csv="${test_dir}/all_psychoacoustic_labels.csv"

mkdir -p \
  "${input_dir}" \
  "${full_wav_dir}" \
  "${segment_dir}" \
  "${full_label_dir}" \
  "${segment_label_dir}"

cp \
  "Demo/sound files used/Dual_1000_1070_1_1_-20_dBFS_48k_Float_LR.wav" \
  "${input_dir}/"

python data_preprocessing/convert_to_wav.py \
  "${input_dir}" \
  "${full_wav_dir}" \
  --training-segment-folder "${segment_dir}"

python data_preprocessing/calculate_reference_values.py \
  "${full_wav_dir}" \
  "${full_label_dir}"

python data_preprocessing/calculate_reference_values.py \
  "${segment_dir}" \
  "${segment_label_dir}" \
  --one-second

python data_preprocessing/calculate_reference_values.py \
  --merge \
  "${segment_dir}/segment_mapping.csv" \
  "${full_label_dir}" \
  "${segment_label_dir}" \
  "${combined_csv}"

row_count=$(($(wc -l < "${combined_csv}") - 1))
if [[ "${row_count}" -ne 2000 ]]; then
  echo "Expected 2000 label rows, found ${row_count}" >&2
  exit 1
fi

echo "Workflow smoke test passed"
echo "Segments: 4"
echo "Combined label rows: ${row_count}"
echo "Results: ${test_dir}"
