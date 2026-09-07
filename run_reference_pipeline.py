"""Run full-dataset references, validation, and optional distribution analysis."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--segment-workers", type=int, default=4)
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()
    if args.segment_workers < 1:
        parser.error("segment-workers must be positive")
    source = args.input_folder.resolve(strict=True)
    output = args.output_folder.resolve()
    if source == output or source in output.parents:
        parser.error("Output must be outside the source directory")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "input_folder": str(source),
              "full_workers": 1, "segment_workers": args.segment_workers,
              "job_id": os.environ.get("SLURM_JOB_ID"), "stages": {}}
    started = time.perf_counter()

    def save():
        (output / "run_report.json").write_text(json.dumps(report, indent=2))

    def stage(name, command):
        print(f"Starting {name}", flush=True)
        start = time.perf_counter()
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run([sys.executable, "-u", *map(str, command)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        report["stages"][name] = {"seconds": time.perf_counter() - start,
                                  "exit_code": result.returncode}
        save()
        result.check_returncode()
        print(f"Completed {name}: {report['stages'][name]['seconds']:.1f}s", flush=True)

    try:
        stage("inventory", ["data_preprocessing/reference_inventory.py", source])
        inventory = json.loads((output / "inventory.log").read_text())
        report["inventory"] = inventory
        print(f"Sources: {inventory['source_files']}; mono songs: {inventory['mono_songs']}; "
              f"mono minutes: {inventory['mono_minutes']:.2f}; "
              f"longest song: {inventory['longest_song_seconds']:.2f}s", flush=True)
        stage("conversion", ["data_preprocessing/convert_to_wav.py", source,
                             output / "full_sound_files",
                             "--training-segment-folder", output / "sound_files"])
        import soundfile as sf
        import pandas as pd
        reference_files = sorted((output / "full_sound_files").glob("*.wav"))
        if len(reference_files) != inventory["mono_songs"]:
            raise ValueError("Converted song count differs from input inventory; inspect conversion.log")
        expected_segments = 0
        for path in reference_files:
            info = sf.info(path)
            if info.channels != 1 or info.samplerate != 48000:
                raise ValueError(f"Unexpected converted audio format: {path}")
            expected_segments += info.frames // info.samplerate
        mapping = pd.read_csv(output / "sound_files/segment_mapping.csv")
        if len(mapping) != expected_segments or len(list((output / "sound_files").glob("*.wav"))) != expected_segments:
            raise ValueError("Segment count differs from converted complete seconds")
        del mapping
        script = "data_preprocessing/calculate_reference_values.py"
        stage("full_references", [script, output / "full_sound_files",
                                  output / "full_recording_labels", "--workers", 1])
        stage("segment_references", [script, output / "sound_files",
                                     output / "one_second_labels", "--one-second",
                                     "--workers", args.segment_workers])
        labels = output / "all_psychoacoustic_labels.csv"
        stage("merge", [script, "--merge", output / "sound_files/segment_mapping.csv",
                        output / "full_recording_labels", output / "one_second_labels", labels])
        from data_preprocessing.validate_reference_csv import validate_reference_csv
        report["validation"] = validate_reference_csv(output / "sound_files/segment_mapping.csv", labels)
        save()
        if not args.skip_analysis:
            stage("distribution", ["-c", "import sys; from tranings_data_visualization.visualize_data_distribution import main; main(sys.argv[1], sys.argv[2], all_data=True)",
                                   labels, output / "visualization"])
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        save()
    print(f"Reference pipeline passed: {output}", flush=True)


if __name__ == "__main__":
    main()
