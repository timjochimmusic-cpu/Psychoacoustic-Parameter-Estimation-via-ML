"""One-second dataset backed by a segment manifest and consolidated targets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .params import FRAME_COUNTS, PARAM_NAMES


class MappedPsychoAcousticDataset(Dataset):
    """Load one-second audio and exact targets without changing the model.

    ``segment_mapping.csv`` provides the authoritative relationship between a
    one-second WAV, its original recording, and its split. ``targets_path`` is
    the consolidated target store created by ``generate_hybrid_targets.py``:
    loudness, roughness, TNR, and SII use isolated one-second calculations;
    sharpness uses three seconds of calculation context but retains only the
    center-second trajectory.
    """

    def __init__(
        self,
        mapping_csv: Path,
        segment_dir: Path,
        targets_path: Path,
        split: str | None = None,
        expected_sample_rate: int = 48_000,
    ):
        self.mapping_csv = Path(mapping_csv)
        self.segment_dir = Path(segment_dir)
        self.expected_sample_rate = expected_sample_rate

        payload = torch.load(
            targets_path,
            map_location="cpu",
            weights_only=True,
        )
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported target schema: {targets_path}")
        self._items: dict[str, dict] = payload["items"]

        mapping = pd.read_csv(self.mapping_csv)
        required = {
            "segment_id",
            "source_file",
            "reference_file",
            "recording_id",
            "sample_rate",
            "start_sample",
            "end_sample",
        }
        missing = required - set(mapping.columns)
        if missing:
            raise ValueError(
                f"{self.mapping_csv} is missing columns: {sorted(missing)}"
            )
        if mapping["segment_id"].duplicated().any():
            raise ValueError("Mapping contains duplicate segment_id values")

        if split is not None:
            if "split" not in mapping:
                raise ValueError(
                    f"{self.mapping_csv} has no split column; assign splits first"
                )
            mapping = mapping[mapping["split"] == split]

        mapping = mapping[
            mapping["segment_id"].astype(str).isin(self._items)
        ].copy()
        mapping = mapping.sort_values(
            ["recording_id", "reference_file", "start_sample"]
        )
        if mapping.empty:
            suffix = "" if split is None else f" for split={split!r}"
            raise ValueError(f"No mapped targets were found{suffix}")

        self.rows = mapping.to_dict("records")
        self.stems = [str(row["segment_id"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.rows[index]
        segment_id = str(row["segment_id"])
        audio_path = self.segment_dir / str(row["source_file"])
        waveform, sample_rate = sf.read(audio_path, dtype="float32")
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono audio: {audio_path}")
        if sample_rate != int(row["sample_rate"]):
            raise ValueError(f"Manifest/audio sample-rate mismatch: {audio_path}")
        if sample_rate != self.expected_sample_rate:
            raise ValueError(
                f"Expected {self.expected_sample_rate} Hz, got {sample_rate}: "
                f"{audio_path}"
            )
        expected_samples = int(row["end_sample"]) - int(row["start_sample"])
        if len(waveform) != expected_samples or len(waveform) != sample_rate:
            raise ValueError(
                f"Expected an exact one-second segment, got {len(waveform)} "
                f"samples: {audio_path}"
            )

        targets = self._items[segment_id]["targets"]
        validated_targets = {}
        for name in PARAM_NAMES:
            if name not in targets:
                raise ValueError(f"{segment_id} is missing target {name}")
            target = targets[name].float().flatten()
            expected_frames = FRAME_COUNTS[name]
            if target.numel() != expected_frames:
                raise ValueError(
                    f"{segment_id}/{name}: expected {expected_frames} values, "
                    f"got {target.numel()}"
                )
            validated_targets[name] = target

        return torch.from_numpy(waveform).unsqueeze(0), validated_targets


def validate_recording_splits(mapping_csv: Path) -> None:
    """Raise if an original recording occurs in more than one split."""
    mapping = pd.read_csv(mapping_csv)
    required = {"recording_id", "split"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping is missing columns: {sorted(missing)}")
    split_counts = mapping.groupby("recording_id")["split"].nunique()
    leaking = split_counts[split_counts > 1]
    if not leaking.empty:
        raise ValueError(
            f"Recording leakage detected for {len(leaking)} recording(s)"
        )
