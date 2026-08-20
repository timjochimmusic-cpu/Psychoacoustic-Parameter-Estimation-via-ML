"""Generate resumable isolated-one-second sharpness targets for A/B tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from mosqito import sharpness_din_tv

from data_preprocessing.generate_hybrid_targets import _calculate


def generate_isolated_sharpness_targets(
    mapping_csv: Path,
    segment_dir: Path,
    output_path: Path,
    segment_starts_ms: set[int] | None = None,
    resume: bool = True,
) -> dict:
    mapping = pd.read_csv(mapping_csv)
    required = {"segment_id", "source_file", "sample_rate", "start_ms"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping is missing columns: {sorted(missing)}")
    if segment_starts_ms is not None:
        mapping = mapping[mapping["start_ms"].isin(segment_starts_ms)]
    if mapping.empty:
        raise ValueError("No mapping rows matched the selection")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and output_path.exists():
        payload = torch.load(output_path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1:
            raise ValueError(f"Cannot resume unsupported store: {output_path}")
        items = payload["items"]
        print(f"Resuming with {len(items)} existing item(s)")
    else:
        items = {}

    mapping = mapping[~mapping["segment_id"].astype(str).isin(items)]
    if len(mapping) == 0:
        print("All selected isolated sharpness targets already exist")

    def save_progress() -> None:
        payload = {
            "schema_version": 1,
            "label_mode": "isolated_one_second_sharpness",
            "items": items,
        }
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(output_path)

    segment_dir = Path(segment_dir)
    for completed, row in enumerate(mapping.itertuples(index=False), 1):
        signal, sample_rate = sf.read(segment_dir / row.source_file)
        if signal.ndim != 1:
            raise ValueError(f"Expected mono segment: {row.source_file}")
        if sample_rate != row.sample_rate:
            raise ValueError(f"Sample-rate mismatch: {row.source_file}")
        values, times = _calculate(
            signal,
            sample_rate,
            "sharpness_din_tv",
            sharpness_din_tv,
        )
        if times is None:
            raise ValueError("Sharpness did not return timestamps")
        items[str(row.segment_id)] = {
            "target": values,
            "times_s": times,
            "metadata": {
                "source_file": str(row.source_file),
                "recording_id": (
                    str(row.recording_id)
                    if hasattr(row, "recording_id")
                    else str(row.reference_file)
                ),
                "split": str(row.split) if hasattr(row, "split") else None,
            },
        }
        save_progress()
        print(f"Generated {completed}/{len(mapping)}: {row.segment_id}")

    save_progress()
    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    print(f"Saved {len(payload['items'])} isolated sharpness item(s) to {output_path}")
    return payload


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    experiment = root / "data" / "experiments" / "contextual_sharpness_multi_recording"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=experiment / "segments" / "segment_mapping.csv",
    )
    parser.add_argument(
        "--segment-dir",
        type=Path,
        default=experiment / "segments",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=experiment / "isolated_sharpness_targets.pt",
    )
    parser.add_argument("--segment-starts-ms", type=int, nargs="+")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    generate_isolated_sharpness_targets(
        mapping_csv=arguments.mapping_csv,
        segment_dir=arguments.segment_dir,
        output_path=arguments.output,
        segment_starts_ms=(
            None
            if arguments.segment_starts_ms is None
            else set(arguments.segment_starts_ms)
        ),
        resume=not arguments.no_resume,
    )
