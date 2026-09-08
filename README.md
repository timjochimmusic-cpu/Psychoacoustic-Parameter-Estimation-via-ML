# Psychoacoustic reference preprocessing

`main.py` converts source audio into complete mono channels,
creates isolated one-second training WAVs, calculates and merges reference labels,
validates the resulting CSV, then generates distribution plots and statistics.
Each segment has 500 rows; shorter parameter trajectories use NaN padding.

- Loudness and Sharpness use complete recordings, sharing the Loudness computation.
- Roughness, TNR and SII use isolated one-second WAVs.
- Incomplete final seconds are excluded from training segments.
- `segment_mapping.csv` links each segment to its source channel and time interval.

## HPC run

From the full-song checkout on the HPC:

```bash
sbatch run_all_music_references.sbatch
```

The launcher selects `music_dataset` from the original repository and requests
12 CPUs, 64 GiB and 12 hours. It uses two full-song workers followed by twelve
one-second workers. Results are written to a fresh directory:

```text
data/standardized_audio_files/music_references_JOBID/
```

This contains `run_report.json`, per-stage logs, audio, reference CSVs,
`all_psychoacoustic_labels.csv`, and `visualization/`. Existing results are not
overwritten; automatic resume is not implemented. Training is not launched.

For other inputs, activate the project environment and run:

```bash
MPLBACKEND=Agg python main.py INPUT_DIR NEW_OUTPUT_DIR --full-workers 2 --segment-workers 12
```

## Before training

Split an existing run's segments using the mapping's original-track identities:

```bash
python data_preprocessing/split_train_val.py data/standardized_audio_files/music_references_1923255/sound_files --val-split 0.2 --seed 42
```

This moves WAVs into `train/` and `val/` and saves `split_mapping.csv` alongside
the original segment mapping. Both channels and all segments of a track stay
together. The split is approximately 80/20 by track count, not by duration.
Rerunning the same command resumes partial moves or leaves a completed split
unchanged; conflicting assignments and missing files are rejected.

After the split, generate training and validation statistics with the existing
visualization script:

```bash
MPLBACKEND=Agg python tranings_data_visualization/visualize_data_distribution.py --run-dir data/standardized_audio_files/music_references_1923255
```

It writes separate `*_train` and `*_val` statistics and plots into the run's
`visualization/` directory, preserving the earlier `*_all` outputs.

The preprocessing plots and statistics describe **all** data. Split by original
track, keeping both stereo channels and all their segments in the same split,
then calculate training-only statistics. Training currently loads
`parameter_average_per_time_segment_train.csv` for initial temporal biases and
`parameter_value_stats_train.csv` for loss normalization under
`data/standardized_audio_files/training_set/visualization/`. Its input paths must
also point to the prepared training and validation data.

## Checks

```bash
python -m unittest discover -s tests
```

Generated data, Slurm logs, Python caches, and local benchmark notes are ignored.
