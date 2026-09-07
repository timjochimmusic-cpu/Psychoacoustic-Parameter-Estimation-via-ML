"""Validate training targets in bounded chunks before distribution analysis."""
from pathlib import Path

import numpy as np
import pandas as pd


def validate_reference_csv(mapping_csv: Path, labels_csv: Path):
    from data_preprocessing.calculate_reference_values import FRAME_COUNTS

    mapping = pd.read_csv(mapping_csv)
    if mapping.empty or mapping.segment_id.duplicated().any():
        raise ValueError("Empty or duplicate segment mapping")
    if not (mapping.duration_ms == 1000).all():
        raise ValueError("Mapping contains incomplete seconds")
    completed = 0
    with pd.read_csv(labels_csv, chunksize=50000) as reader:
        for frame in reader:
            if len(frame) % 500:
                raise ValueError("Incomplete 500-row segment")
            count = len(frame) // 500
            expected = mapping.segment_id.iloc[completed:completed + count].to_numpy()
            if not np.array_equal(frame.segment_id.to_numpy(), np.repeat(expected, 500)):
                raise ValueError("CSV segments differ from mapping order or count")
            if not np.array_equal(frame.time_index.to_numpy(), np.tile(np.arange(500), count)):
                raise ValueError("Invalid frame indices")
            for parameter, valid_count in FRAME_COUNTS.items():
                values = frame[parameter].to_numpy(dtype=float).reshape(count, 500)
                if not np.isfinite(values[:, :valid_count]).all():
                    raise ValueError(f"Missing/nonfinite {parameter} target")
                if not np.isnan(values[:, valid_count:]).all():
                    raise ValueError(f"Unexpected {parameter} values in padding")
            completed += count
    if completed != len(mapping):
        raise ValueError("CSV does not cover the complete mapping")
    return {"segments": completed, "rows": completed * 500, "status": "passed"}
