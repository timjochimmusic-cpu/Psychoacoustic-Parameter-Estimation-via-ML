"""Calculate train-only statistics from consolidated one-second targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from DL_model.params import FRAME_COUNTS, PARAM_NAMES


def summarize_targets(
    mapping_csv: Path,
    targets_path: Path,
    output_dir: Path,
    split: str = "train",
) -> None:
    mapping = pd.read_csv(mapping_csv)
    if "split" not in mapping:
        raise ValueError("Mapping has no split column")
    split_ids = set(
        mapping.loc[mapping["split"] == split, "segment_id"].astype(str)
    )
    if not split_ids:
        raise ValueError(f"Mapping contains no rows for split={split!r}")

    payload = torch.load(targets_path, map_location="cpu", weights_only=True)
    items = payload["items"]
    selected_ids = split_ids & set(items)
    if not selected_ids:
        raise ValueError(
            f"Target store contains no items for split={split!r}"
        )
    unavailable = len(split_ids - set(items))
    if unavailable:
        print(
            f"Ignoring {unavailable} manifest row(s) without generated targets"
        )

    arrays: dict[str, np.ndarray] = {}
    temporal_means: dict[str, np.ndarray] = {}
    for name in PARAM_NAMES:
        tensors = []
        for segment_id in sorted(selected_ids):
            target = items[segment_id]["targets"][name].float().flatten()
            if target.numel() != FRAME_COUNTS[name]:
                raise ValueError(
                    f"{segment_id}/{name}: expected {FRAME_COUNTS[name]}, "
                    f"got {target.numel()}"
                )
            tensors.append(target)
        stacked = torch.stack(tensors).numpy()
        arrays[name] = stacked.reshape(-1)
        temporal_means[name] = np.nanmean(stacked, axis=0)

    value_frame = pd.DataFrame(
        {name: pd.Series(values) for name, values in arrays.items()}
    )
    value_stats = value_frame.describe()
    max_frames = max(FRAME_COUNTS.values())
    temporal_frame = pd.DataFrame({"time_index": np.arange(max_frames)})
    for name, values in temporal_means.items():
        temporal_frame[name] = pd.Series(values)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    value_path = output_dir / f"parameter_value_stats_{split}.csv"
    temporal_path = output_dir / f"parameter_average_per_time_segment_{split}.csv"
    value_stats.to_csv(value_path)
    temporal_frame.to_csv(temporal_path, index=False)
    print(f"Summarized {len(selected_ids)} {split} target item(s)")
    print(f"Saved: {value_path}")
    print(f"Saved: {temporal_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    summarize_targets(
        mapping_csv=arguments.mapping_csv,
        targets_path=arguments.targets,
        output_dir=arguments.output_dir,
        split=arguments.split,
    )
