from __future__ import annotations
import argparse
from pathlib import Path
import joblib
from .data import load_dataset
from .features import FEATURE_COLUMNS, make_feature_frame


def predict_latest(data_path: str | Path, artifacts_dir: str | Path = "artifacts"):
    df = load_dataset(data_path)
    featured = make_feature_frame(df, include_targets=False).dropna(subset=FEATURE_COLUMNS)
    if featured.empty:
        raise ValueError("No rows have the complete 15-minute temperature history needed for prediction.")
    row = featured.iloc[[-1]]
    print("\nLatest eligible input observation:", row["timestamp"].iloc[0])
    print("Input room temperature:", f"{row['room_temp_c'].iloc[0]:.2f} °C")
    print("NOTE: predictions reflect the dataset used here. Sample CSV is synthetic, not real room data.")
    for horizon in (30, 60):
        p = Path(artifacts_dir) / f"model_{horizon}min.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p.resolve()}. Train first: python -m rac_ai.train --data {data_path}")
        bundle = joblib.load(p)
        prediction = float(bundle["model"].predict(row[FEATURE_COLUMNS])[0])
        print(f"Predicted room temperature in {horizon} minutes: {prediction:.2f} °C ({bundle['model_name']})")


def main():
    parser = argparse.ArgumentParser(description="Predict temperature from the newest complete row in a CSV.")
    parser.add_argument("--data", default="data/sample_temperature_data.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    predict_latest(args.data, args.artifacts)

if __name__ == "__main__":
    main()
