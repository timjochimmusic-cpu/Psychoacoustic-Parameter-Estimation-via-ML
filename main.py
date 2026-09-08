"""Run full-dataset references, validation, and optional distribution analysis."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def inventory(input_folder):
    import soundfile as sf

    extensions = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".wav"}
    sources = sorted(p for p in Path(input_folder).rglob("*")
                     if p.is_file() and p.suffix.lower() in extensions)
    if not sources:
        raise ValueError(f"No source audio found: {input_folder}")
    records = []
    names = set()
    for path in sources:
        # Conversion names include only the parent folder and source stem.
        name = f"{path.parent.name}_{path.stem}"
        if name in names:
            raise ValueError(f"Ambiguous conversion name {name}: {path}")
        names.add(name)
        try:
            info = sf.info(path)
            duration, channels = info.duration, info.channels
        except RuntimeError:
            # Match the converter's decoder for formats unsupported by libsndfile.
            import librosa
            audio, sr = librosa.load(path, sr=None, mono=False)
            duration = audio.shape[-1] / sr
            channels = 1 if audio.ndim == 1 else audio.shape[0]
            del audio
        if duration < 1 or channels < 1:
            raise ValueError(f"Source has no complete training second: {path}")
        records.append({"path": str(path.resolve()), "seconds": duration,
                        "channels": channels})
    mono_seconds = sum(r["seconds"] * r["channels"] for r in records)
    return {
        "input_folder": str(Path(input_folder).resolve()),
        "source_files": len(records),
        "mono_songs": sum(r["channels"] for r in records),
        "mono_minutes": mono_seconds / 60,
        "longest_song_seconds": max(r["seconds"] for r in records),
        "benchmark_linear_hours": mono_seconds * 6.948804621257683 / 3600,
        "estimate_note": "Single-song extrapolation only; excludes distribution analysis.",
        "sources": records,
    }


def validate_reference_csv(mapping_csv: Path, labels_csv: Path):
    import numpy as np
    import pandas as pd
    from data_preprocessing.calculate_reference_values import FRAME_COUNTS

    mapping = pd.read_csv(mapping_csv)
    if mapping.empty or mapping.segment_id.duplicated().any():
        raise ValueError("Empty or duplicate segment mapping")
    if not (mapping.duration_ms == 1000).all():
        raise ValueError("Mapping contains incomplete seconds")
    completed = 0
    with pd.read_csv(labels_csv, chunksize=50000) as reader:
        for frame in reader:
            if len(frame) % 500:
                raise ValueError("Incomplete 500-row segment")
            count = len(frame) // 500
            expected = mapping.segment_id.iloc[completed:completed + count].to_numpy()
            if not np.array_equal(frame.segment_id.to_numpy(), np.repeat(expected, 500)):
                raise ValueError("CSV segments differ from mapping order or count")
            if not np.array_equal(frame.time_index.to_numpy(), np.tile(np.arange(500), count)):
                raise ValueError("Invalid frame indices")
            for parameter, valid_count in FRAME_COUNTS.items():
                values = frame[parameter].to_numpy(dtype=float).reshape(count, 500)
                if not np.isfinite(values[:, :valid_count]).all():
                    raise ValueError(f"Missing/nonfinite {parameter} target")
                if not np.isnan(values[:, valid_count:]).all():
                    raise ValueError(f"Unexpected {parameter} values in padding")
            completed += count
    if completed != len(mapping):
        raise ValueError("CSV does not cover the complete mapping")
    return {"segments": completed, "rows": completed * 500, "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path, nargs="?")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--full-workers", type=int, default=2)
    parser.add_argument("--segment-workers", type=int, default=12)
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()
    if args.full_workers < 1 or args.segment_workers < 1:
        parser.error("full-workers and segment-workers must be positive")
    source = args.input_folder.resolve(strict=True)
    if args.inventory_only:
        report = inventory(source)
        if args.summary:
            report.pop("sources")
        print(json.dumps(report, indent=2))
        return
    if args.output_folder is None:
        parser.error("output_folder is required unless --inventory-only is used")
    output = args.output_folder.resolve()
    if source == output or source in output.parents:
        parser.error("Output must be outside the source directory")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "input_folder": str(source),
              "full_workers": args.full_workers, "segment_workers": args.segment_workers,
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
        stage("inventory", ["main.py", source, "--inventory-only"])
        source_inventory = json.loads((output / "inventory.log").read_text())
        report["inventory"] = source_inventory
        print(f"Sources: {source_inventory['source_files']}; mono songs: {source_inventory['mono_songs']}; "
              f"mono minutes: {source_inventory['mono_minutes']:.2f}; "
              f"longest song: {source_inventory['longest_song_seconds']:.2f}s", flush=True)
        stage("conversion", ["data_preprocessing/convert_to_wav.py", source,
                             output / "full_sound_files",
                             "--training-segment-folder", output / "sound_files"])
        import soundfile as sf
        import pandas as pd
        reference_files = sorted((output / "full_sound_files").glob("*.wav"))
        if len(reference_files) != source_inventory["mono_songs"]:
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
                                  output / "full_recording_labels", "--workers", args.full_workers])
        stage("segment_references", [script, output / "sound_files",
                                     output / "one_second_labels", "--one-second",
                                     "--workers", args.segment_workers])
        labels = output / "all_psychoacoustic_labels.csv"
        stage("merge", [script, "--merge", output / "sound_files/segment_mapping.csv",
                        output / "full_recording_labels", output / "one_second_labels", labels])
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
