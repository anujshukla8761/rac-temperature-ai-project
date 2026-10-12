import unittest
from pathlib import Path
import numpy as np
import joblib

from rac_ai.features import FEATURE_COLUMNS, make_single_feature_row

ROOT = Path(__file__).resolve().parents[1]


class BundledModelInferenceTests(unittest.TestCase):
    def test_both_bundled_models_match_feature_schema_and_predict(self):
        values = {
            "room_temp_c": 28.0,
            "relative_humidity_pct": 55.0,
            "outdoor_temp_c": 35.0,
            "ac_set_temp_c": 24.0,
            "occupants": 2,
            "ac_running": True,
            "ac_runtime_min": 20,
            "room_temp_lag_5m_c": 28.2,
            "room_temp_lag_10m_c": 28.4,
            "room_temp_lag_15m_c": 28.6,
        }
        X = make_single_feature_row(values)
        self.assertEqual(list(X.columns), FEATURE_COLUMNS)
        for horizon in (30, 60):
            with self.subTest(horizon=horizon):
                model_path = ROOT / "artifacts" / f"model_{horizon}min.joblib"
                self.assertTrue(model_path.exists(), f"Missing bundled model: {model_path}")
                bundle = joblib.load(model_path)
                self.assertEqual(bundle.get("horizon_minutes"), horizon)
                self.assertEqual(list(bundle.get("features", [])), FEATURE_COLUMNS)
                prediction = float(bundle["model"].predict(X[FEATURE_COLUMNS])[0])
                self.assertTrue(np.isfinite(prediction))
                self.assertGreater(prediction, -10.0)
                self.assertLess(prediction, 60.0)


if __name__ == "__main__":
    unittest.main()
