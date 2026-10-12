import tempfile
import unittest
import pandas as pd
from pathlib import Path
from rac_ai.synthetic import generate_synthetic_data
from rac_ai.data import load_dataset

class DataTests(unittest.TestCase):
    def test_synthetic_dataset_passes_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.csv"
            generate_synthetic_data(days=3).to_csv(path, index=False)
            df = load_dataset(path)
            self.assertEqual(len(df), 864)
            self.assertEqual(df.timestamp.is_monotonic_increasing, True)

    def test_infinite_measurement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "infinite.csv"
            df = generate_synthetic_data(days=3)
            df.loc[0, "room_temp_c"] = float("inf")
            df.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "finite numbers"):
                load_dataset(path)

    def test_out_of_range_humidity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "humidity.csv"
            df = generate_synthetic_data(days=3)
            df.loc[0, "relative_humidity_pct"] = 101
            df.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "between 0 and 100"):
                load_dataset(path)

    def test_duplicate_timestamp_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicates.csv"
            df = generate_synthetic_data(days=3)
            df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
            df.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate timestamps"):
                load_dataset(path)

if __name__ == "__main__":
    unittest.main()
