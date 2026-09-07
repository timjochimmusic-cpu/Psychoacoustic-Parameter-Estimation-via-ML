# Full music reference run

Prepared for the project owner, 7 September 2026. Scope: only the original
repository's data/raw/music_dataset, all channels treated as separate songs.

## Evidence and allocation

User-supplied results for job 1923035 show a successful end-to-end benchmark:
183 complete seconds, 91,500 validated rows, 1277.08 seconds inside the runner,
and 21:36 Slurm elapsed. Reported batch MaxRSS was 9,797,492 KiB, approximately
9.34 GiB. Shared Sharpness took 0.2459 seconds after 504.5734 seconds for Loudness.
These are measurements for one 183.78-second song, not bounds for longer inputs.

The prepared launcher requests 12 CPUs, 64 GiB, and a user-selected 12-hour limit,
with two full-song workers and twelve one-second workers in successive stages.
The supplied inventory has 39 source files, 78 mono songs, 272.41 mono minutes,
and a longest song of 363.32 seconds. Memory and runtime scaling are not established
by a single recording. The earlier 272-minute linear estimate was 31.5 hours;
stereo duration must count both channels and plotting adds work.

## Before submission

Push the local commit and pull it in the HPC full-song checkout, then run:

```bash
source /beegfs/home/users/t/tjochim/projects/Psychoacoustic-Parameter-Estimation-via-ML/.venv/bin/activate
python data_preprocessing/reference_inventory.py /beegfs/home/users/t/tjochim/projects/Psychoacoustic-Parameter-Estimation-via-ML/data/raw/music_dataset --summary
scontrol show partition standard
```

Inspect total mono minutes, longest song duration, and the partition's time limit
before selecting the final allocation. The inventory reports ambiguous conversion
names and unreadable/too-short sources as errors. Most formats use header reads;
unsupported formats use the converter's decoder and can take longer to inspect.

Once the allocation is suitable:

```bash
sbatch run_all_music_references.sbatch
```

The launcher contains the actual music_dataset path. It writes into a new
data/standardized_audio_files/music_references_JOBID directory. Original source
files and previous benchmark outputs are preserved. Restarting with a new job
creates a new directory and recomputes results; automatic resume is not implemented.

## Workflow and outputs

Inventory → conversion and segmentation → shared full-song Loudness/Sharpness →
isolated one-second Roughness/TNR/SII → merge → chunked validation → distribution
analysis. A failed stage prevents later stages from starting. Training never runs.
Incomplete final seconds remain in the full-song reference and are excluded from
training segments, matching the benchmark. Conversion and segment counts are
checked against the inventory and converted audio.

Validation checks mapping coverage, exact 500-row groups and frame indices, finite
target values, and expected padding in bounded chunks. Analysis begins only after
validation succeeds. The existing visualization function now accepts explicit
paths and an all-data mode; it generates histograms, per-frame averages, value
statistics and per-segment frame counts. calculate_biases in main.py was an alias
for this function, not a separate bias computation. There are no train/val splits
at this stage and no model-based bias evaluation. Plots use a headless backend.
The visualization loader still holds the selected label columns in memory; its
memory use should be reviewed for datasets much larger than the stated size.

The run directory contains run_report.json, per-stage logs, source inventory,
full and one-second WAVs/labels, all_psychoacoustic_labels.csv, and visualization/.
The parent Slurm files are music-references-JOBID.out and .err.

## Verification and limits

Local checks: nine regression tests, including a chunk-boundary validation failure
and stereo inventory counts; a separate 1.05-second synthetic end-to-end run passed
and generated four PNG plots. Python and shell syntax checks passed. Plot existence
was checked; visual rendering quality was not reviewed. No HPC access or full-data
run was performed by the assistant. Peak disk requirements and longest-source
memory are not yet measured. Research was limited to supplied job records and
the repository implementation because private dataset details require the user's
HPC inventory. The planning tool was unavailable.
