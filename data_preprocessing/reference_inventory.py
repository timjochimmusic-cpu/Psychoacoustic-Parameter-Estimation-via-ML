"""Inspect source durations and channels before allocating a full reference job."""
import argparse
import json
from pathlib import Path


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("--summary", action="store_true", help="Omit individual source records")
    args = parser.parse_args()
    report = inventory(args.input_folder)
    if args.summary:
        report.pop("sources")
    print(json.dumps(report, indent=2))
