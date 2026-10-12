from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .data import load_dataset
from .features import FEATURE_COLUMNS, make_feature_frame


def _metric_dict(actual, predicted):
    return {
        "mae_c": float(mean_absolute_error(actual, predicted)),
        "rmse_c": float(np.sqrt(mean_squared_error(actual, predicted))),
        "n": int(len(actual)),
    }


def _make_candidates():
    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=180, min_samples_leaf=3, max_features=1.0,
            random_state=42, n_jobs=-1,
        ),
    }


def train_all(data_path: str | Path, artifacts_dir: str | Path = "artifacts", reports_dir: str | Path = "reports") -> dict:
    raw = load_dataset(data_path)
    featured = make_feature_frame(raw, include_targets=True)
    artifacts_dir, reports_dir = Path(artifacts_dir), Path(reports_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Time-based cutoffs: oldest 70% for training, next 15% for validation, final 15% for testing.
    times = np.array(sorted(featured["timestamp"].unique()))
    cut1 = pd.Timestamp(times[int(len(times) * 0.70)])
    cut2 = pd.Timestamp(times[int(len(times) * 0.85)])
    data_bytes = Path(data_path).read_bytes()
    data_sha256 = hashlib.sha256(data_bytes).hexdigest()
    results = {"data_path": Path(data_path).name, "data_sha256": data_sha256,
               "data_warning": "Bundled sample data are simulated; scores are for demonstration only.",
               "rows_in_csv": int(len(raw)), "first_timestamp": str(raw.timestamp.iloc[0]), "last_timestamp": str(raw.timestamp.iloc[-1]),
               "validation_cutoff": str(cut1), "test_cutoff": str(cut2), "horizons": {},
               "features": FEATURE_COLUMNS, "split_method": "chronological: 70% training period, 15% validation period, 15% test period; target timestamp purged across boundaries"}
    predictions_output = []
    for horizon in (30, 60):
        ycol = f"target_room_temp_{horizon}m_c"
        tcol = f"target_timestamp_{horizon}m"
        usable = featured.dropna(subset=FEATURE_COLUMNS + [ycol, tcol]).copy()
        usable = usable.sort_values("timestamp").reset_index(drop=True)
        train = usable[usable[tcol] <= cut1]
        val = usable[(usable["timestamp"] >= cut1) & (usable[tcol] <= cut2)]
        test = usable[usable["timestamp"] >= cut2]
        if min(len(train), len(val), len(test)) < 20:
            raise ValueError(
                f"Not enough usable samples for {horizon}-minute training split: "
                f"train={len(train)}, validation={len(val)}, test={len(test)}. Collect more data or check timestamps."
            )
        X_train, y_train = train[FEATURE_COLUMNS], train[ycol]
        X_val, y_val = val[FEATURE_COLUMNS], val[ycol]
        candidate_scores = {}
        for name, candidate in _make_candidates().items():
            candidate.fit(X_train, y_train)
            candidate_scores[name] = _metric_dict(y_val, candidate.predict(X_val))
        selected_name = min(candidate_scores, key=lambda name: candidate_scores[name]["mae_c"])
        # Refit the validation-selected model on all labels available before the test period.
        refit = usable[usable[tcol] <= cut2]
        X_refit, y_refit = refit[FEATURE_COLUMNS], refit[ycol]
        final_model = _make_candidates()[selected_name]
        final_model.fit(X_refit, y_refit)
        y_test = test[ycol].to_numpy()
        y_pred = final_model.predict(test[FEATURE_COLUMNS])
        persistence = test["room_temp_c"].to_numpy()
        test_metrics = {
            selected_name: _metric_dict(y_test, y_pred),
            "PersistenceBaseline_current_temp": _metric_dict(y_test, persistence),
        }
        results["horizons"][str(horizon)] = {
            "selected_model": selected_name,
            "train_rows": int(len(train)), "validation_rows": int(len(val)), "test_rows": int(len(test)),
            "validation_scores": candidate_scores,
            "test_scores": test_metrics,
            "beats_persistence_baseline_on_mae": bool(test_metrics[selected_name]["mae_c"] < test_metrics["PersistenceBaseline_current_temp"]["mae_c"]),
            "interpretation": "Test metrics describe the final held-out chronological period. Compare with persistence baseline; lower MAE/RMSE is better."
        }
        model_path = artifacts_dir / f"model_{horizon}min.joblib"
        joblib.dump({"model": final_model, "features": FEATURE_COLUMNS, "horizon_minutes": horizon,
                     "model_name": selected_name, "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                     "target": ycol, "training_data_sha256": data_sha256}, model_path)
        per_row = pd.DataFrame({"timestamp": test["timestamp"].astype(str).to_numpy(),
                                "target_timestamp": test[tcol].astype(str).to_numpy(),
                                "horizon_minutes": horizon, "actual_room_temp_c": y_test,
                                "predicted_room_temp_c": y_pred,
                                "persistence_baseline_c": persistence})
        predictions_output.append(per_row)
        # One plot for each horizon; no plot is claimed as real experimental evidence.
        fig, ax = plt.subplots(figsize=(10, 4.5))
        plot_rows = min(350, len(test))
        ax.plot(test["timestamp"].iloc[:plot_rows], y_test[:plot_rows], label="Measured / dataset target")
        ax.plot(test["timestamp"].iloc[:plot_rows], y_pred[:plot_rows], label="AI prediction")
        ax.plot(test["timestamp"].iloc[:plot_rows], persistence[:plot_rows], label="Current-temp baseline", alpha=0.7)
        ax.set_title(f"{horizon}-minute-ahead temperature prediction — held-out test period")
        ax.set_ylabel("Room temperature (°C)")
        ax.set_xlabel("Prediction time")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(reports_dir / f"actual_vs_predicted_{horizon}min.png", dpi=140)
        plt.close(fig)
        print(f"\n{horizon}-minute horizon")
        print(f"  Usable rows: train={len(train)}, validation={len(val)}, test={len(test)}")
        print(f"  Selected on validation MAE: {selected_name}")
        print(f"  Test AI: MAE={test_metrics[selected_name]['mae_c']:.3f} °C; RMSE={test_metrics[selected_name]['rmse_c']:.3f} °C")
        print(f"  Test baseline: MAE={test_metrics['PersistenceBaseline_current_temp']['mae_c']:.3f} °C; RMSE={test_metrics['PersistenceBaseline_current_temp']['rmse_c']:.3f} °C")
        print(f"  Model file: {model_path.resolve()}")
    result_path = reports_dir / "metrics.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.concat(predictions_output, ignore_index=True).to_csv(reports_dir / "test_predictions.csv", index=False)
    print(f"\nMetrics saved to: {result_path.resolve()}")
    print("Remember: these scores are only a simulated-data demonstration if you trained on the bundled sample.")
    return results
