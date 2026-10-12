import unittest
import pandas as pd
from rac_ai.features import FEATURE_COLUMNS, make_feature_frame, make_single_feature_row
from rac_ai.synthetic import generate_synthetic_data

class FeatureTests(unittest.TestCase):
    def test_exact_lags_and_future_targets(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=25, freq="5min"),
            "room_temp_c": [20.0 + i * 0.1 for i in range(25)],
            "relative_humidity_pct": [50.0] * 25,
            "outdoor_temp_c": [30.0] * 25,
            "ac_set_temp_c": [24.0] * 25,
            "occupants": [2] * 25,
            "ac_running": [1] * 25,
            "ac_runtime_min": [i * 5 for i in range(25)],
        })
        featured = make_feature_frame(df, include_targets=True)
        row = featured.loc[featured.timestamp == pd.Timestamp("2026-01-01 01:00:00")].iloc[0]
        self.assertAlmostEqual(row["room_temp_lag_5m_c"], 20.0 + 11 * 0.1)
        self.assertAlmostEqual(row["target_room_temp_30m_c"], 20.0 + 18 * 0.1)
        self.assertAlmostEqual(row["target_room_temp_60m_c"], 20.0 + 24 * 0.1)
        self.assertIn("room_temp_mean_15m_c", FEATURE_COLUMNS)

    def test_single_row_feature_shape(self):
        values = {"room_temp_c": 28, "relative_humidity_pct": 55, "outdoor_temp_c": 35,
                  "ac_set_temp_c": 24, "occupants": 2, "ac_running": True, "ac_runtime_min": 20,
                  "room_temp_lag_5m_c": 28.1, "room_temp_lag_10m_c": 28.3, "room_temp_lag_15m_c": 28.5}
        X = make_single_feature_row(values)
        self.assertEqual(X.shape, (1, len(FEATURE_COLUMNS)))

    def test_synthetic_dataset_cadence_and_schema(self):
        df = generate_synthetic_data(days=3)
        self.assertEqual(len(df), 3 * 24 * 12)
        self.assertTrue((pd.to_datetime(df.timestamp).diff().dropna() == pd.Timedelta(minutes=5)).all())
        self.assertIn("room_temp_c", df.columns)

if __name__ == "__main__":
    unittest.main()
