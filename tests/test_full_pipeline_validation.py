import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from data_preprocessing.reference_inventory import inventory
from data_preprocessing.validate_reference_csv import validate_reference_csv


class FullPipelineTests(unittest.TestCase):
    def test_inventory_counts_channels_and_rejects_colliding_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sf.write(root / "song.wav", np.zeros((48000, 2)), 48000)
            result = inventory(root)
            self.assertEqual(result["mono_songs"], 2)
            self.assertAlmostEqual(result["mono_minutes"], 2 / 60)
            for branch in ("a", "b"):
                directory = root / branch / "same"
                directory.mkdir(parents=True)
                sf.write(directory / "collision.wav", np.zeros(48000), 48000)
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                inventory(root)

    def test_streaming_validation_and_missing_target(self):
        from data_preprocessing.calculate_reference_values import FRAME_COUNTS
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # Cross a 50,000-row read boundary, including a short final chunk.
            ids = [f"segment_{i}" for i in range(101)]
            pd.DataFrame({"segment_id": ids, "duration_ms": 1000}).to_csv(root / "mapping.csv", index=False)
            frame = pd.DataFrame({"segment_id": np.repeat(ids, 500),
                                  "time_index": np.tile(np.arange(500), len(ids))})
            for name, count in FRAME_COUNTS.items():
                frame[name] = np.tile(np.r_[np.ones(count), np.full(500 - count, np.nan)], len(ids))
            frame.to_csv(root / "labels.csv", index=False)
            result = validate_reference_csv(root / "mapping.csv", root / "labels.csv")
            self.assertEqual(result["rows"], 50500)
            frame.loc[50000, "sii_ansi"] = np.nan
            frame.to_csv(root / "labels.csv", index=False)
            with self.assertRaisesRegex(ValueError, "sii_ansi"):
                validate_reference_csv(root / "mapping.csv", root / "labels.csv")


if __name__ == "__main__":
    unittest.main()
