import sys
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import numpy as np
from data_preprocessing import calculate_reference_values as refs


class ReferenceFailureTests(unittest.TestCase):
    def run_calculation(self, result=None, exception=None, parameter="loudness_zwtv"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.root = root
        (root / "song.wav").touch()
        future = Future()
        if exception:
            future.set_exception(exception)
        else:
            future.set_result(result)
        with patch.object(refs, "ProcessPoolExecutor") as pool:
            pool.return_value.__enter__.return_value.submit.return_value = future
            original_stdout = sys.stdout
            try:
                refs.calculate_reference_values(root, root / "labels", max_workers=1,
                                                parameter=parameter)
                if parameter is None:
                    submit = pool.return_value.__enter__.return_value.submit
                    self.assertEqual(submit.call_count, 1)
                    self.assertIs(submit.call_args.args[0], refs._compute_full_recording)
            finally:
                sys.stdout = original_stdout
        return root / "labels"

    def assert_no_csv(self):
        self.assertEqual(list(self.root.rglob("*.csv")), [])

    def test_worker_pool_failure_does_not_save_csv(self):
        with self.assertRaisesRegex(RuntimeError, "song/loudness_zwtv"):
            self.run_calculation(exception=RuntimeError("worker terminated"))
        self.assert_no_csv()

    def test_reported_parameter_error_does_not_save_csv(self):
        with self.assertRaisesRegex(RuntimeError, "calculation failed"):
            self.run_calculation(("song", "loudness_zwtv", None, None, "calculation failed"))
        self.assert_no_csv()

    def test_nonfinite_result_does_not_save_csv(self):
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            self.run_calculation(("song", "loudness_zwtv", np.array([np.nan]), None, None))
        self.assert_no_csv()

    def test_success_saves_complete_csv(self):
        labels = self.run_calculation(("song", "loudness_zwtv", np.array([1., 2.]),
                                       np.array([0., .002]), None))
        frame = refs.pd.read_csv(labels / "song.csv")
        self.assertEqual(frame["loudness_zwtv"].tolist(), [1., 2.])
        self.assertEqual(list(labels.glob("*.tmp")), [])

    def test_default_mode_saves_both_targets_from_one_task(self):
        rows = [("song", name, np.array([1., 2.]), np.array([0., .002]), None)
                for name in ("loudness_zwtv", "sharpness_din_tv")]
        labels = self.run_calculation(rows, parameter=None)
        frame = refs.pd.read_csv(labels / "song.csv")
        self.assertIn("loudness_zwtv_time_s", frame)
        self.assertIn("sharpness_din_tv_time_s", frame)

    def test_shared_task_failure_does_not_save_csv(self):
        with self.assertRaisesRegex(RuntimeError, "song/full_recording"):
            self.run_calculation(exception=RuntimeError("worker terminated"), parameter=None)
        self.assert_no_csv()


if __name__ == "__main__":
    unittest.main()
