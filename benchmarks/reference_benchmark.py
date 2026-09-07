"""Benchmark one complete 3–5 minute mono song using production preprocessing."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="A real, complete 180–300 second recording")
    parser.add_argument("output", type=Path, help="New, isolated benchmark directory")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import soundfile as sf
    import numpy as np
    import pandas as pd
    from data_preprocessing.convert_to_wav import convert_to_wav

    report = {
        "source": str(source), "channel": 1, "workers": args.workers,
        "host": platform.node(), "python": sys.version,
        "packages": {p: version(p) for p in ("mosqito", "numpy", "scipy", "pandas", "librosa")},
        "environment": {k: os.environ.get(k) for k in (
            "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "stages": {}, "status": "running",
    }
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2))

    def stage(name, arguments):
        start = time.perf_counter()
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run([sys.executable, "-u", *map(str, arguments)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        report["stages"][name] = time.perf_counter() - start
        save()
        result.check_returncode()
        print(f"{name}: {report['stages'][name]:.2f}s", flush=True)

    start = time.perf_counter()
    try:
        # Keep channel 1 complete; do not concatenate clips or repeat synthetic tones.
        inputs = output / "input"
        inputs.mkdir()
        (inputs / source.name).symlink_to(source)
        t0 = time.perf_counter()
        convert_to_wav(inputs, output / "full_wav", number_samples=1)
        report["stages"]["convert"] = time.perf_counter() - t0
        references = list((output / "full_wav").glob("*.wav"))
        if len(references) != 1:
            raise ValueError("Expected exactly one converted mono song")
        info = sf.info(references[0])
        if info.channels != 1 or not 180 <= info.duration <= 300:
            raise ValueError(f"Choose a complete 180–300 second song; got {info}")
        seconds = info.frames // info.samplerate
        report.update(audio_seconds=info.duration, complete_seconds=seconds)
        save()
        stage("split", ["-c", "from pathlib import Path; from data_preprocessing.convert_to_wav import split_reference_into_training_segments as split; import sys; split(Path(sys.argv[1]), Path(sys.argv[2]))", output / "full_wav", output / "segments"])
        script = ROOT / "data_preprocessing/calculate_reference_values.py"
        stage("full_references", [script, output / "full_wav", output / "full_labels", "--workers", args.workers])
        stage("segment_references", [script, output / "segments", output / "segment_labels", "--one-second", "--workers", args.workers])
        stage("merge", [script, "--merge", output / "segments/segment_mapping.csv", output / "full_labels", output / "segment_labels", output / "labels.csv"])
        from data_preprocessing.calculate_reference_values import FRAME_COUNTS
        frame = pd.read_csv(output / "labels.csv")
        groups = frame.groupby("segment_id", sort=False)
        if len(groups) != seconds or not (groups.size() == 500).all():
            raise ValueError("Incorrect segment count or rows per segment")
        for name, count in FRAME_COUNTS.items():
            if not (groups[name].count() == count).all():
                raise ValueError(f"Missing or unexpected values for {name}")
            if not np.isfinite(frame[name].dropna()).all():
                raise ValueError(f"Nonfinite values for {name}")
        metrics = {}
        for name in ("full_references", "segment_references"):
            log = (output / f"{name}.log").read_text()
            if "FAILED" in log or "Failed " in log:
                raise ValueError(f"Parameter failure in {name}.log")
            for metric, elapsed in re.findall(r"\] (\w+): ([0-9.]+)s", log):
                metrics.setdefault(metric, []).append(float(elapsed))
        report["parameter_task_seconds"] = {
            k: {"tasks": len(v), "sum": sum(v), "median": float(np.median(v)), "max": max(v)}
            for k, v in metrics.items()
        }
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - start
        if "audio_seconds" in report:
            report["wall_seconds_per_audio_second"] = report["wall_seconds"] / report["audio_seconds"]
            report["linear_272_minute_estimate_hours"] = report["wall_seconds_per_audio_second"] * 272 / 60
        save()


if __name__ == "__main__":
    main()
