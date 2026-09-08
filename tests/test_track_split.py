import csv
import tempfile
import unittest
from pathlib import Path
from data_preprocessing.split_train_val import split_train_val


class TrackSplitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = []
        for track in range(5):
            for channel in range(2):
                for segment in range(2):
                    stem = f'track{track}_ch{channel}_seg{segment}'
                    (self.root / f'{stem}.wav').touch()
                    self.rows.append(dict(source_file=f'{stem}.wav', segment_id=stem,
                                          recording_id=f'track{track}'))
        with (self.root / 'segment_mapping.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_track_grouping_and_repeat(self):
        train, val = split_train_val(self.root)
        self.assertEqual(len(list(train.glob('*.wav'))), 16)
        self.assertEqual(len(list(val.glob('*.wav'))), 4)
        with (self.root / 'split_mapping.csv').open() as handle:
            assignments = list(csv.DictReader(handle))
        for track in {r['recording_id'] for r in assignments}:
            self.assertEqual(len({r['split'] for r in assignments if r['recording_id'] == track}), 1)
        before = (self.root / 'split_mapping.csv').read_bytes()
        split_train_val(self.root)
        self.assertEqual(before, (self.root / 'split_mapping.csv').read_bytes())
        # Simulate an interrupted move after the manifest was saved.
        path = next(train.glob('*.wav'))
        path.rename(self.root / path.name)
        split_train_val(self.root)
        self.assertTrue(path.exists())

    def test_missing_file_prevents_any_moves(self):
        (self.root / self.rows[-1]['source_file']).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            split_train_val(self.root)
        self.assertFalse((self.root / 'train').exists())
        self.assertFalse((self.root / 'split_mapping.csv').exists())

    def test_different_split_rejected(self):
        split_train_val(self.root)
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            split_train_val(self.root, val_split=.4)


if __name__ == '__main__':
    unittest.main()
