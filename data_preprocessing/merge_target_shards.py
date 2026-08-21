"""Merge non-overlapping consolidated target shards into one target store."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def merge_target_shards(shards: list[Path], output_path: Path) -> dict:
    if not shards:
        raise ValueError("No target shards were provided")

    merged_items = {}
    label_mode = None
    for shard in sorted(Path(path) for path in shards):
        payload = torch.load(shard, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported target schema: {shard}")
        shard_mode = payload.get("label_mode")
        if label_mode is None:
            label_mode = shard_mode
        elif shard_mode != label_mode:
            raise ValueError(f"Target label-mode mismatch: {shard}")

        overlap = set(merged_items) & set(payload["items"])
        if overlap:
            raise ValueError(
                f"Shard {shard} overlaps {len(overlap)} existing item(s)"
            )
        merged_items.update(payload["items"])
        print(f"Merged {shard}: {len(payload['items'])} item(s)")

    result = {
        "schema_version": 1,
        "label_mode": label_mode,
        "items": merged_items,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(result, temporary_path)
    temporary_path.replace(output_path)
    print(f"Saved {len(merged_items)} merged item(s) to {output_path}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    merge_target_shards(arguments.shards, arguments.output)
