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
contain individual timings. GNU time records resource usage in the Slurm error
log. After completion, collect Slurm accounting as well:

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
- One mono song creates only two full-recording tasks. Extra CPUs cannot speed
  that stage through the existing task scheduler; the full dataset may achieve
  better utilization by processing multiple songs concurrently.
- Each parameter task independently reads its WAV. Full-song Loudness and
  Sharpness are separate calculations. Inspect the installed MoSQITo implementation
  before deciding whether shared intermediates could safely avoid duplicated work.
- The merge caches full-song CSVs, but scans full trajectories for each segment.
  Use the measured merge time to decide whether this needs optimization.
- Task duration sums overlap and are not stage wall times or CPU usage. Compare
  stage wall times with Slurm TotalCPU and MaxRSS before drawing conclusions.
- The reported 272-minute linear estimate is a rough scenario, not a forecast:
  song lengths, signal content, concurrency, startup overhead, and storage matter.
  Confirm whether the historical 272 minutes count mono channels the same way.

Initial state: benchmark prepared locally; HPC execution has not occurred because
gateway SSH authentication failed. No measured benchmark findings are available yet.
