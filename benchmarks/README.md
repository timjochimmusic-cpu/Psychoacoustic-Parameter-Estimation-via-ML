# Reference preprocessing benchmark

Submit from the repository on the HPC, using a real, complete 180–300 second
recording from the dataset:

```bash
sbatch run_reference_benchmark.sbatch /absolute/path/to/song.wav
```

The benchmark keeps the first channel as one complete mono song. Thus a
four-minute stereo source supplies four mono-minutes, not eight. It does not
crop or repeat audio. Select a representative recording, not the synthetic
smoke-test tone. Fractional final seconds remain in full-song references and
are discarded when making training segments, matching main.py.

Every job uses a fresh directory under data/experiments/reference_benchmark_JOBID.
It records package versions, thread settings, conversion/splitting/reference/
merge wall times, and parameter task durations in report.json. Reference logs
contain individual timings. The launcher requires no external timing executable.
After completion, collect CPU and memory usage from Slurm accounting:

```bash
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,AllocCPUS,TotalCPU,MaxRSS
```

Validation requires one group per complete second, 500 rows per group, and
500/500/9/2/1 finite values for Loudness/Sharpness/Roughness/TNR/SII respectively.
NaN padding beyond a parameter's expected frames is allowed. Failed calculations
cannot pass merely because the merged CSV has the right number of rows.

## Performance questions

- Production currently defaults to 12 workers; the previous smoke allocation
  requested 4 CPUs. This benchmark explicitly matches workers to allocated CPUs
  and limits native numerical threads to one. Production defaults are unchanged.
- The default full-song stage now creates one task per mono song. It calculates
  Loudness once and derives Sharpness using sharpness_din_from_loudness, preserving
  sharpness_din_tv's default skip=0 timestamps and existing CSV column names.
  The large specific-loudness array stays inside the worker. Explicit --parameter
  diagnostics still use the independent original functions for comparison.
- One-second parameter tasks independently read their WAV. Multiple full-song
  tasks can still exceed available memory; the shared computation does not make
  arbitrary song lengths or worker counts safe under 16 GiB.
- The merge caches full-song CSVs, but scans full trajectories for each segment.
  Use the measured merge time to decide whether this needs optimization.
- Task duration sums overlap and are not stage wall times or CPU usage. Compare
  stage wall times with Slurm TotalCPU and MaxRSS before drawing conclusions.
- The reported 272-minute linear estimate is a rough scenario, not a forecast:
  song lengths, signal content, concurrency, startup overhead, and storage matter.
  Confirm whether the historical 272 minutes count mono channels the same way.

Job 1922333 failed before Python started because the compute node lacked
`/usr/bin/time`. The launcher now invokes Python directly; timing remains in
`report.json`, and resource usage is collected with `sacct`. No preprocessing
performance measurements were produced by that failed job.

## Isolate full-song memory usage after job 1922355

Job 1922355 reported an OOM kill and both full-song futures failed with a broken
process pool. The old implementation saved an invalid CSV and continued; merging
then failed because Loudness had no time axis. This does not identify which
parameter caused the OOM event. Reference failures now raise an error, cancel
queued tasks, and prevent the failed recording's CSV from being published.
Already-running tasks may finish during executor shutdown. Existing CSVs are
still skipped; do not reuse the failed job's full_labels directory as output.

Run the diagnostic from the HPC checkout:

```bash
sbatch run_full_reference_diagnostic.sbatch
```

It reuses job 1922355's complete mono WAV, runs Loudness then Sharpness in separate
Slurm steps with one worker each, and keeps the original 16 GiB allocation. An
optional first argument selects another full-WAV directory. Results go into a
fresh full_reference_diagnostic_JOBID directory with separate parameter folders.
These folders are diagnostic outputs, not a combined full-label directory for
merging. No conversion, segmentation, one-second calculations, or training runs.
If Loudness fails, the job stops before Sharpness.

Collect step-level timing and memory after completion:

```bash
sacct -j JOBID --format=JobID,JobName%24,State%20,ExitCode,Elapsed,TotalCPU,MaxRSS
cat full-ref-diagnostic-JOBID.out
cat full-ref-diagnostic-JOBID.err
```
