"""Split mapped one-second WAVs by original track, keeping stereo channels together."""
import argparse
import csv
import random
import shutil
from pathlib import Path


def split_train_val(sound_dir: Path, val_split: float = 0.2, seed: int = 42) -> tuple[Path, Path]:
    sound_dir = Path(sound_dir)
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between zero and one")
    with (sound_dir / "segment_mapping.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not {"source_file", "recording_id", "segment_id"}.issubset(fields):
            raise ValueError("Mapping must contain source_file, recording_id and segment_id")
        rows = list(reader)
    if not rows:
        raise ValueError("Segment mapping is empty")
    filenames = [row["source_file"] for row in rows]
    if len(set(filenames)) != len(rows) or len({r["segment_id"] for r in rows}) != len(rows):
        raise ValueError("Mapping contains duplicate files or segments")
    for row in rows:
        name = row["source_file"]
        if Path(name).name != name or not name.endswith(".wav") or not row["recording_id"]:
            raise ValueError(f"Invalid mapping entry: {row}")
    tracks = sorted({row["recording_id"] for row in rows})
    if len(tracks) < 2:
        raise ValueError("At least two original tracks are required")
    random.Random(seed).shuffle(tracks)
    n_val = min(len(tracks) - 1, max(1, round(len(tracks) * val_split)))
    val_tracks = set(tracks[:n_val])
    assignments = {row["source_file"]: "val" if row["recording_id"] in val_tracks else "train"
                   for row in rows}
    manifest = sound_dir / "split_mapping.csv"
    if manifest.exists():
        with manifest.open(newline="") as handle:
            saved_rows = list(csv.DictReader(handle))
        expected_rows = [dict(row, split=assignments[row["source_file"]]) for row in rows]
        if saved_rows != expected_rows:
            raise ValueError("Existing split manifest differs; do not change the mapping, seed or ratio mid-run")
    # Validate the entire filesystem before moving anything. A partial run can resume,
    # but conflicting splits or unknown WAVs must never be silently accepted.
    expected_names = set(filenames)
    for folder in (sound_dir, sound_dir / "train", sound_dir / "val"):
        unknown = {p.name for p in folder.glob("*.wav")} - expected_names
        if unknown:
            raise ValueError(f"Unmapped WAVs in {folder}: {sorted(unknown)[:3]}")
    moves = []
    for name, split in assignments.items():
        source = sound_dir / name
        target = sound_dir / split / name
        other = sound_dir / ("train" if split == "val" else "val") / name
        if other.exists() or source.exists() == target.exists():
            raise ValueError(f"Missing, duplicated or incorrectly assigned WAV: {name}")
        if source.exists():
            moves.append((source, target))
    for split in ("train", "val"):
        (sound_dir / split).mkdir(exist_ok=True)
    if not manifest.exists():
        temporary = manifest.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*fields, "split"])
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row, split=assignments[row["source_file"]]))
        temporary.replace(manifest)
    for source, target in moves:
        shutil.move(str(source), str(target))
    for split in ("train", "val"):
        count = sum(value == split for value in assignments.values())
        track_count = n_val if split == "val" else len(tracks) - n_val
        print(f"{split}: {track_count} original tracks, {count} segments ({count / 60:.2f} mono minutes)")
    print(f"Moved {len(moves)} WAVs; assignments: {manifest}")
    return sound_dir / "train", sound_dir / "val"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sound_dir", type=Path)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_train_val(args.sound_dir, args.val_split, args.seed)
