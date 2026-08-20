"""Assign deterministic train/validation/test splits by reference recording."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def assign_manifest_splits(
    mapping_csv: Path,
    output_csv: Path,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("Split fractions must not be negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than one")

    mapping = pd.read_csv(mapping_csv)
    if "reference_file" not in mapping:
        raise ValueError("Mapping must contain a reference_file column")
    group_column = (
        "recording_id" if "recording_id" in mapping else "reference_file"
    )

    recordings = sorted(mapping[group_column].dropna().unique())
    random.Random(seed).shuffle(recordings)
    count = len(recordings)
    if count == 0:
        raise ValueError("Mapping contains no reference recordings")

    validation_count = round(count * validation_fraction)
    test_count = round(count * test_fraction)
    if count >= 2 and validation_fraction > 0:
        validation_count = max(1, validation_count)
    if count >= 3 and test_fraction > 0:
        test_count = max(1, test_count)
    while validation_count + test_count >= count:
        if test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1

    validation_recordings = set(recordings[:validation_count])
    test_recordings = set(
        recordings[validation_count:validation_count + test_count]
    )

    def split_for(recording: str) -> str:
        if recording in validation_recordings:
            return "val"
        if recording in test_recordings:
            return "test"
        return "train"

    mapping["split"] = mapping[group_column].map(split_for)
    leakage = mapping.groupby(group_column)["split"].nunique()
    if (leakage != 1).any():
        raise AssertionError("An original recording was assigned to multiple splits")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_csv, index=False)

    recording_counts = (
        mapping[[group_column, "split"]]
        .drop_duplicates()["split"]
        .value_counts()
    )
    segment_counts = mapping["split"].value_counts()
    for split in ("train", "val", "test"):
        print(
            f"{split}: {int(recording_counts.get(split, 0))} recording(s), "
            f"{int(segment_counts.get(split, 0))} segment(s)"
        )
    print(f"Saved split manifest to {output_csv}")
    return mapping


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    default_mapping = (
        root / "data" / "processed" / "1s_training" / "segment_mapping.csv"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-csv", type=Path, default=default_mapping)
    parser.add_argument("--output", type=Path, default=default_mapping)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    assign_manifest_splits(
        mapping_csv=arguments.mapping_csv,
        output_csv=arguments.output,
        validation_fraction=arguments.validation_fraction,
        test_fraction=arguments.test_fraction,
        seed=arguments.seed,
    )
