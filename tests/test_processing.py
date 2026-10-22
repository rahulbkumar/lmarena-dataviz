"""Regression checks for missing data and invalid snapshot handling."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from lmarena_data.process_lmarena_data import aggregate_snapshots, validate_summary


class ProcessingTests(unittest.TestCase):
    def test_missing_month_and_provider_are_not_forward_filled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'monthly_2025.01.csv').write_text('organization,arena_score\nGoogle,1200\nGoogle,1250\nMistral,1100\n')
            (root / 'monthly_2025.03.csv').write_text('organization,arena_score\nOpenAI,1300\nMistral AI,1150\n')
            result = aggregate_snapshots(root)
            self.assertEqual(len(result), 3)
            self.assertEqual(result.loc[0, 'Google'], 1250)
            self.assertEqual(result.loc[2, 'Mistral AI'], 1150)
            self.assertTrue(result.loc[1].drop('Date').isna().all())
            self.assertTrue(pd.isna(result.loc[2, 'Google']))

    def test_bad_scores_fail_instead_of_skipping_month(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'monthly_2025.01.csv').write_text('organization,arena_score\nGoogle,oops\n')
            with self.assertRaises(ValueError):
                aggregate_snapshots(Path(directory))

    def test_duplicate_dates_rejected(self):
        source = pd.read_csv(Path(__file__).resolve().parents[1] / 'lmarena_data/lmarena_top_scores_by_provider.csv')
        with self.assertRaises(ValueError):
            validate_summary(pd.concat([source, source.iloc[:1]]))

    def test_missing_source_files_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'No monthly'):
                aggregate_snapshots(Path(directory))


if __name__ == '__main__':
    unittest.main()
