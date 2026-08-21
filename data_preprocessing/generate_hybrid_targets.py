"""Generate consolidated one-second targets with contextual sharpness.

The four unchanged parameters are calculated on each isolated training WAV.
Sharpness is calculated on the mapped three-second reference window and only
the center-second trajectory is retained. Results are stored in one PyTorch
file keyed by ``segment_id``; no per-segment label CSVs are created.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from mosqito import loudness_zwtv, roughness_dw, sharpness_din_tv, sii_ansi, tnr_ecma_perseg

from data_preprocessing.calculate_reference_values import _extract_values_and_time


ISOLATED_PARAMETERS = {
    "loudness_zwtv": (loudness_zwtv, (), {}),
    "roughness_dw": (roughness_dw, (), {}),
    "tnr_ecma_perseg": (tnr_ecma_perseg, (), {}),
    "sii_ansi": (sii_ansi, ("critical", "normal"), {}),
}


def _calculate(
    signal: np.ndarray,
    sample_rate: int,
    name: str,
    function,
    args: tuple = (),
    kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    result = function(signal, sample_rate, *args, **(kwargs or {}))
    values, times = _extract_values_and_time(
        param_name=name,
        result=result,
        signal_duration_s=len(signal) / sample_rate,
    )
    value_tensor = torch.tensor(np.asarray(values), dtype=torch.float32)
    time_tensor = (
        None
        if times is None
        else torch.tensor(np.asarray(times), dtype=torch.float32)
    )
    return value_tensor, time_tensor


def generate_hybrid_targets(
    mapping_csv: Path,
    segment_dir: Path,
    reference_dir: Path,
    output_path: Path,
    segment_starts_ms: set[int] | None = None,
    max_segments: int | None = None,
    resume: bool = True,
    checkpoint_every: int = 1,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> dict:
    mapping_csv = Path(mapping_csv)
    segment_dir = Path(segment_dir)
    reference_dir = Path(reference_dir)
    output_path = Path(output_path)
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be greater than zero")
    if (shard_index is None) != (num_shards is None):
        raise ValueError("shard_index and num_shards must be provided together")
    if num_shards is not None:
        if num_shards <= 0:
            raise ValueError("num_shards must be greater than zero")
        if not 0 <= shard_index < num_shards:
            raise ValueError("shard_index must be in [0, num_shards)")

    mapping = pd.read_csv(mapping_csv)
    required = {
        "segment_id",
        "source_file",
        "reference_file",
        "sample_rate",
        "start_sample",
        "end_sample",
        "context_start_sample",
        "context_end_sample",
        "has_full_context",
    }
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(
            f"{mapping_csv} is missing contextual mapping columns: {sorted(missing)}. "
            "Regenerate it with split_reference_into_training_segments()."
        )

    mapping = mapping[mapping["has_full_context"].astype(bool)]
    if num_shards is not None:
        group_column = (
            "recording_id" if "recording_id" in mapping else "reference_file"
        )
        groups = sorted(mapping[group_column].astype(str).unique())
        selected_groups = {
            group
            for position, group in enumerate(groups)
            if position % num_shards == shard_index
        }
        mapping = mapping[
            mapping[group_column].astype(str).isin(selected_groups)
        ]
        print(
            f"Shard {shard_index + 1}/{num_shards}: "
            f"{len(selected_groups)} recording group(s), {len(mapping)} segment(s)"
        )
    if segment_starts_ms is not None:
        mapping = mapping[mapping["start_ms"].isin(segment_starts_ms)]
    if max_segments is not None:
        mapping = mapping.head(max_segments)
    if mapping.empty:
        raise ValueError("No full-context mapping rows matched the selection")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and output_path.exists():
        payload = torch.load(output_path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1:
            raise ValueError(f"Cannot resume unsupported store: {output_path}")
        items: dict[str, dict] = payload["items"]
        print(f"Resuming with {len(items)} existing target item(s)")
    else:
        items = {}

    selected_ids = set(mapping["segment_id"].astype(str))
    already_complete = selected_ids & set(items)
    if already_complete:
        print(f"Skipping {len(already_complete)} already-generated item(s)")
        existing_rows = mapping[
            mapping["segment_id"].astype(str).isin(already_complete)
        ]
        for row in existing_rows.itertuples(index=False):
            metadata = items[str(row.segment_id)]["metadata"]
            metadata["recording_id"] = (
                str(row.recording_id)
                if hasattr(row, "recording_id")
                else str(row.reference_file)
            )
            metadata["split"] = (
                str(row.split) if hasattr(row, "split") else None
            )
    mapping = mapping[~mapping["segment_id"].astype(str).isin(items)]

    reference_cache: dict[str, tuple[np.ndarray, int]] = {}

    def save_progress() -> None:
        payload = {
            "schema_version": 1,
            "label_mode": "isolated_parameters_with_contextual_sharpness",
            "items": items,
        }
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(output_path)

    for completed, row in enumerate(mapping.itertuples(index=False), 1):
        segment_path = segment_dir / row.source_file
        segment_signal, segment_rate = sf.read(segment_path)
        if segment_signal.ndim != 1:
            raise ValueError(f"Expected mono segment: {segment_path}")
        if segment_rate != row.sample_rate:
            raise ValueError(f"Sample-rate mismatch for {segment_path}")

        reference_name = str(row.reference_file)
        if reference_name not in reference_cache:
            reference_signal, reference_rate = sf.read(reference_dir / reference_name)
            if reference_signal.ndim != 1:
                raise ValueError(f"Expected mono reference: {reference_name}")
            reference_cache[reference_name] = (reference_signal, reference_rate)
        reference_signal, reference_rate = reference_cache[reference_name]
        if reference_rate != segment_rate:
            raise ValueError(f"Reference/segment rate mismatch for {row.segment_id}")

        targets: dict[str, torch.Tensor] = {}
        times: dict[str, torch.Tensor | None] = {}
        for name, (function, args, kwargs) in ISOLATED_PARAMETERS.items():
            targets[name], times[name] = _calculate(
                segment_signal,
                segment_rate,
                name,
                function,
                args,
                kwargs,
            )

        context_signal = reference_signal[
            int(row.context_start_sample):int(row.context_end_sample)
        ]
        sharpness, sharpness_times = _calculate(
            context_signal,
            reference_rate,
            "sharpness_din_tv",
            sharpness_din_tv,
        )
        if sharpness_times is None:
            raise ValueError("Contextual sharpness did not return timestamps")

        target_start_in_context_s = (
            int(row.start_sample) - int(row.context_start_sample)
        ) / reference_rate
        target_end_in_context_s = (
            int(row.end_sample) - int(row.context_start_sample)
        ) / reference_rate
        mask = (
            (sharpness_times >= target_start_in_context_s)
            & (sharpness_times < target_end_in_context_s)
        )
        targets["sharpness_din_tv"] = sharpness[mask]
        times["sharpness_din_tv"] = (
            sharpness_times[mask] - target_start_in_context_s
        )

        items[str(row.segment_id)] = {
            "targets": targets,
            "times_s": times,
            "metadata": {
                "source_file": str(row.source_file),
                "reference_file": reference_name,
                "recording_id": (
                    str(row.recording_id)
                    if hasattr(row, "recording_id")
                    else reference_name
                ),
                "sample_rate": int(row.sample_rate),
                "start_sample": int(row.start_sample),
                "end_sample": int(row.end_sample),
                "context_start_sample": int(row.context_start_sample),
                "context_end_sample": int(row.context_end_sample),
                "split": str(row.split) if hasattr(row, "split") else None,
            },
        }
        print(f"Generated {completed}/{len(mapping)}: {row.segment_id}")
        if completed % checkpoint_every == 0:
            save_progress()

    save_progress()
    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    print(f"Saved {len(items)} hybrid target item(s) to {output_path}")
    return payload


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=root / "data" / "processed" / "1s_training" / "segment_mapping.csv",
    )
    parser.add_argument(
        "--segment-dir",
        type=Path,
        default=root / "data" / "processed" / "1s_training",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=root / "data" / "processed" / "1file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "experiments" / "contextual_sharpness_prototype" / "hybrid_targets.pt",
    )
    parser.add_argument("--segment-starts-ms", type=int, nargs="+")
    parser.add_argument("--max-segments", type=int)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Replace any existing target store instead of resuming it.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    generate_hybrid_targets(
        mapping_csv=arguments.mapping_csv,
        segment_dir=arguments.segment_dir,
        reference_dir=arguments.reference_dir,
        output_path=arguments.output,
        segment_starts_ms=(
            None
            if arguments.segment_starts_ms is None
            else set(arguments.segment_starts_ms)
        ),
        max_segments=arguments.max_segments,
        resume=not arguments.no_resume,
        checkpoint_every=arguments.checkpoint_every,
        shard_index=arguments.shard_index,
        num_shards=arguments.num_shards,
    )
