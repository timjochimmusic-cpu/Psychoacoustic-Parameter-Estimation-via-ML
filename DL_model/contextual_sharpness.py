"""Prototype dataset and model branch for center-second sharpness prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .DL_model import PsychoacousticModel
from .params import FRAME_COUNTS, PARAM_NAMES


class ContextualSharpnessDataset(Dataset):
    """Load one-second center audio, three-second context, and hybrid targets."""

    def __init__(
        self,
        mapping_csv: Path,
        segment_dir: Path,
        reference_dir: Path,
        targets_path: Path,
        split: str | None = None,
    ):
        self.mapping_csv = Path(mapping_csv)
        self.segment_dir = Path(segment_dir)
        self.reference_dir = Path(reference_dir)
        payload = torch.load(targets_path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported hybrid target schema")
        self._items = payload["items"]

        mapping = pd.read_csv(self.mapping_csv)
        if split is not None:
            if "split" not in mapping:
                raise ValueError(
                    f"{self.mapping_csv} has no split column; run "
                    "assign_manifest_splits.py first"
                )
            mapping = mapping[mapping["split"] == split]
        mapping = mapping[mapping["segment_id"].isin(self._items)].copy()
        mapping = mapping.sort_values(["reference_file", "start_sample"])
        if mapping.empty:
            suffix = "" if split is None else f" for split={split!r}"
            raise ValueError(
                f"No mapping rows match the hybrid target store{suffix}"
            )
        self.rows = mapping.to_dict("records")
        self.stems = [str(row["segment_id"]) for row in self.rows]
        self._reference_cache: dict[str, tuple[np.ndarray, int]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_reference(self, filename: str) -> tuple[np.ndarray, int]:
        if filename not in self._reference_cache:
            signal, sample_rate = sf.read(self.reference_dir / filename)
            if signal.ndim != 1:
                raise ValueError(f"Expected mono reference: {filename}")
            self._reference_cache[filename] = (signal, sample_rate)
        return self._reference_cache[filename]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        row = self.rows[index]
        segment, segment_rate = sf.read(self.segment_dir / row["source_file"])
        reference, reference_rate = self._load_reference(row["reference_file"])
        if segment_rate != reference_rate or segment_rate != row["sample_rate"]:
            raise ValueError(f"Sample-rate mismatch for {row['segment_id']}")

        context = reference[
            int(row["context_start_sample"]):int(row["context_end_sample"])
        ]
        if len(context) != 3 * len(segment):
            raise ValueError(
                f"{row['segment_id']} does not contain exactly one second of "
                "context on each side"
            )

        targets = self._items[str(row["segment_id"])]["targets"]
        for name in PARAM_NAMES:
            expected = FRAME_COUNTS[name]
            actual = targets[name].numel()
            if actual != expected:
                raise ValueError(
                    f"{row['segment_id']}/{name}: expected {expected} targets, "
                    f"got {actual}"
                )

        return (
            torch.from_numpy(np.asarray(segment)).float().unsqueeze(0),
            torch.from_numpy(np.asarray(context)).float().unsqueeze(0),
            targets,
        )


def collate_contextual_sharpness(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    centers = torch.stack([item[0] for item in batch])
    contexts = torch.stack([item[1] for item in batch])
    targets = {
        name: torch.stack([item[2][name] for item in batch])
        for name in PARAM_NAMES
    }
    return centers, contexts, targets


class ContextualSharpnessBranch(nn.Module):
    """Predict center-second sharpness from a three-second waveform context."""

    def __init__(
        self,
        sample_rate: int = 48_000,
        n_fft: int = 1024,
        hop_length: int = 96,
        output_frames: int = FRAME_COUNTS["sharpness_din_tv"],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.output_frames = output_frames
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        frequency_bins = n_fft // 2 + 1
        self.network = nn.Sequential(
            nn.Conv1d(frequency_bins, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=8, dilation=4),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=32, dilation=16),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )

    def forward(
        self,
        context_waveform: torch.Tensor,
        center_samples: int,
    ) -> torch.Tensor:
        signal = context_waveform.squeeze(1)
        spectrum = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        features = torch.log1p(spectrum.abs())
        trajectory = self.network(features).squeeze(1)

        context_samples = signal.shape[-1]
        center_start_sample = (context_samples - center_samples) // 2
        center_end_sample = center_start_sample + center_samples
        start_frame = (
            center_start_sample + self.hop_length - 1
        ) // self.hop_length
        end_frame = (
            center_end_sample + self.hop_length - 1
        ) // self.hop_length
        center_trajectory = trajectory[:, start_frame:end_frame]
        return F.adaptive_avg_pool1d(
            center_trajectory.unsqueeze(1),
            self.output_frames,
        ).squeeze(1)


class HybridContextModel(nn.Module):
    """Existing one-second model plus a dedicated contextual sharpness branch."""

    def __init__(
        self,
        initial_temporal_biases: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.center_model = PsychoacousticModel(initial_temporal_biases)
        self.sharpness_branch = ContextualSharpnessBranch()

    def forward(
        self,
        center_waveform: torch.Tensor,
        context_waveform: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.center_model(center_waveform)
        outputs["sharpness_din_tv"] = self.sharpness_branch(
            context_waveform,
            center_samples=center_waveform.shape[-1],
        )
        return outputs

    def forward_sharpness(
        self,
        center_waveform: torch.Tensor,
        context_waveform: torch.Tensor,
    ) -> torch.Tensor:
        """Run only the prototype branch during sharpness-only training."""
        return self.sharpness_branch(
            context_waveform,
            center_samples=center_waveform.shape[-1],
        )
