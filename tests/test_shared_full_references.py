import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from data_preprocessing import calculate_reference_values as refs


class SharedFullReferenceTests(unittest.TestCase):
    def test_matches_independent_mosqito_values_and_timestamps(self):
        sr = 48000
        t = np.arange(int(1.2 * sr)) / sr
        signal = (0.03 * (1 + 0.5 * np.sin(2 * np.pi * 7 * t))
                  * np.sin(2 * np.pi * 1000 * t))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "song.wav"
            sf.write(path, signal, sr)
            signal, sr = sf.read(path)
            with patch.object(refs, "loudness_zwtv", wraps=refs.loudness_zwtv) as loudness:
                shared = refs._compute_full_recording("song", str(path))
                self.assertEqual(loudness.call_count, 1)
            for result, fn in zip(shared, (refs.loudness_zwtv, refs.sharpness_din_tv)):
                expected_values, expected_times = refs._extract_values_and_time(
                    result[1], fn(signal, sr), len(signal) / sr
                )
                np.testing.assert_array_equal(result[2], expected_values)
                np.testing.assert_array_equal(result[3], expected_times)
            frame = refs._build_reference_dataframe({
                row[1]: {"values": row[2], "times": row[3]} for row in shared
            })
            self.assertEqual(set(frame.columns), {
                "loudness_zwtv", "loudness_zwtv_time_s",
                "sharpness_din_tv", "sharpness_din_tv_time_s",
            })


if __name__ == "__main__":
    unittest.main()
