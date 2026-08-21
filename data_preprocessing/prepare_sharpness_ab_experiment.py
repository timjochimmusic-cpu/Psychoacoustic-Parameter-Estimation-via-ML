"""Prepare reference audio, one-second segments, and leakage-free splits.

This command is intentionally resumable: existing WAV files are retained and
the manifest is regenerated deterministically from all available references.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data_preprocessing.assign_manifest_splits import assign_manifest_splits
from data_preprocessing.convert_to_wav import (
    convert_to_wav,
    split_reference_into_training_segments,
)


def prepare_experiment(
    raw_dir: Path,
    experiment_dir: Path,
    reference_duration_s: float | None = 60.0,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    experiment_dir = Path(experiment_dir)
    reference_dir = experiment_dir / "references"
    segment_dir = experiment_dir / "segments"

    convert_to_wav(
        input_folder=raw_dir,
        output_folder=reference_dir,
        segment_length_s=reference_duration_s,
        overlap_s=0.0,
        max_segments_per_source=1,
    )
    split_reference_into_training_segments(
        input_folder=reference_dir,
        output_folder=segment_dir,
        segment_length_s=1.0,
        context_s=1.0,
    )
    mapping_path = segment_dir / "segment_mapping.csv"
    assign_manifest_splits(
        mapping_csv=mapping_path,
        output_csv=mapping_path,
        validation_fraction=validation_fraction,
        test_fraction=0.0,
        seed=seed,
    )


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=root / "data" / "raw" / "music_dataset",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=root / "data" / "experiments" / "sharpness_ab_full",
    )
    parser.add_argument("--reference-duration-s", type=float, default=60.0)
    parser.add_argument(
        "--full-recordings",
        action="store_true",
        help="Keep complete recordings instead of only their first 60 seconds.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    prepare_experiment(
        raw_dir=arguments.raw_dir,
        experiment_dir=arguments.experiment_dir,
        reference_duration_s=(
            None if arguments.full_recordings else arguments.reference_duration_s
        ),
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
    )
