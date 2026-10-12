"""Streamlit website for the RAC room-temperature forecasting ML project."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from rac_ai.data import load_dataset
from rac_ai.features import FEATURE_COLUMNS, make_single_feature_row
from rac_ai.modeling import train_all

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sample_temperature_data.csv"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
METRICS_PATH = REPORTS / "metrics.json"
HORIZONS = (30, 60)

st.set_page_config(
    page_title="RAC Temperature AI",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def uploaded_csv_path(uploaded_file) -> Path:
    """Store upload outside the project in the OS temp folder, without touching data/ files."""
    content = uploaded_file.getvalue()
    digest = sha256_bytes(content)
    temp_dir = Path(tempfile.gettempdir()) / "rac_temperature_ai_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{digest}.csv"
    if not path.exists():
        path.write_bytes(content)
    return path


def read_metrics() -> dict | None:
    try:
        if not METRICS_PATH.exists():
            return None
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_bundle(horizon: int) -> dict:
    path = ARTIFACTS / f"model_{horizon}min.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model file is missing: {path.name}. Train the models from the sidebar first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"{path.name} is not a recognised model bundle. Retrain the models.")
    stored_features = list(bundle.get("features", []))
    if stored_features and stored_features != FEATURE_COLUMNS:
        raise ValueError(
            f"{path.name} was trained with a different feature schema. Retrain using this project version."
        )
    return bundle


def fmt_timestamp(value) -> str:
    return pd.Timestamp(value).strftime("%d %b %Y, %H:%M")


st.title("🌡️ AI-Based Room Temperature Prediction")
st.caption(
    "A Room Air Conditioning (RAC) engineering project using supervised machine learning "
    "to estimate room temperature 30 and 60 minutes ahead."
)

existing_model_files = any((ARTIFACTS / f"model_{h}min.joblib").exists() for h in HORIZONS)
with st.sidebar:
    st.header("Project controls")
    st.markdown("**1. Choose your dataset**")
    st.caption("The default file is simulated teaching data. Upload your own measured CSV to validate or train on it.")
    uploaded = st.file_uploader("Optional measured-data CSV", type=["csv"], help="Must contain the columns described in data/README.md.")
    active_path = DATA_PATH
    active_file_label = DATA_PATH.name
    if uploaded is not None:
        try:
            active_path = uploaded_csv_path(uploaded)
            active_file_label = uploaded.name or "uploaded_data.csv"
        except OSError as exc:
            st.error(f"Could not prepare the uploaded CSV: {exc}")

    st.markdown("**2. Train the models**")
    st.caption("Training uses chronological train/validation/test periods. The existing model and report files can be replaced only after you explicitly confirm.")
    confirm_replace = False
    if existing_model_files:
        confirm_replace = st.checkbox("I confirm replacing the existing models and reports after successful training", value=False)
    train_clicked = st.button(
        "Train / retrain 30- and 60-minute models",
        type="primary",
        use_container_width=True,
        disabled=(existing_model_files and not confirm_replace),
    )
    if existing_model_files and not confirm_replace:
        st.caption("Training is disabled to protect the already-saved model files. Tick the confirmation box above to enable it.")
    if train_clicked:
        try:
            # Validate before starting a comparatively expensive training run.
            checked_data = load_dataset(active_path)
            with st.spinner(f"Training on {len(checked_data):,} validated observations. This can take a little while..."):
                train_all(active_path, ARTIFACTS, REPORTS)
            st.success("Training completed. The website has reloaded the latest metrics below.")
        except Exception as exc:
            st.error(f"Training could not complete: {exc}")

    st.divider()
    st.markdown("**Project files**")
    st.code("artifacts/model_30min.joblib\nartifacts/model_60min.joblib\nreports/metrics.json", language="text")
    st.caption("Uploaded CSV files are staged in the operating system temporary folder; the bundled sample file is not overwritten.")

metrics = read_metrics()
model_status = {}
for horizon in HORIZONS:
    model_path = ARTIFACTS / f"model_{horizon}min.joblib"
    model_status[horizon] = model_path.exists()

# Load and validate the selected dataset independently from model inference. A different
# CSV should trigger a provenance warning, not disable forecasts for manually entered inputs.
data = None
data_error = None
try:
    data = load_dataset(active_path)
except Exception as exc:
    data_error = str(exc)

active_sha = None
try:
    active_sha = sha256_bytes(active_path.read_bytes())
except OSError:
    pass
trained_data_sha = metrics.get("data_sha256") if metrics else None
data_matches_metrics = bool(active_sha and trained_data_sha and active_sha == trained_data_sha)

# Status strip
s1, s2, s3 = st.columns(3)
s1.metric("30-minute model", "Available" if model_status[30] else "Not found")
s2.metric("60-minute model", "Available" if model_status[60] else "Not found")
if data is not None:
    s3.metric("Selected dataset", f"{len(data):,} rows")
else:
    s3.metric("Selected dataset", "Needs attention")

if uploaded is None:
    st.warning(
        "The bundled sample dataset is simulated teaching data, not experimentally measured room data. "
        "Its metrics demonstrate the software workflow and must not be presented as validated real-world HVAC accuracy."
    )
if data_error:
    st.error(f"Selected dataset failed validation: {data_error}")
    st.info("The forecasting form can still use existing trained models. To evaluate or retrain, correct the CSV schema and data quality first.")
elif metrics and not data_matches_metrics:
    st.warning(
        "The selected dataset differs from the dataset recorded in the training metrics. "
        "Existing models can still make manual-input forecasts, but the Evaluation tab continues to describe the prior training run. "
        "Retrain from the sidebar to evaluate models on the selected dataset."
    )
elif metrics and data_matches_metrics:
    st.success("Dataset provenance matches the training metrics recorded by this project.")

forecast_tab, evaluation_tab, dataset_tab, guide_tab = st.tabs(
    ["🔮 Predict temperature", "📊 Model evaluation", "🗂️ Dataset explorer", "📘 How to use"]
)

with forecast_tab:
    st.subheader("Predict future room temperature")
    st.write(
        "Enter current room conditions and room-temperature readings from exactly 5, 10, and 15 minutes ago. "
        "The same input conditions are sent to both saved models."
    )
    if not all(model_status.values()):
        st.info("One or both trained model files are missing. Use the training button in the sidebar to create them.")

    with st.form("forecast_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Room temperature history**")
            room = st.number_input("Current room temperature (°C)", min_value=-5.0, max_value=60.0, value=28.0, step=0.1)
            lag5 = st.number_input("5 minutes ago (°C)", min_value=-5.0, max_value=60.0, value=28.2, step=0.1)
            lag10 = st.number_input("10 minutes ago (°C)", min_value=-5.0, max_value=60.0, value=28.4, step=0.1)
            lag15 = st.number_input("15 minutes ago (°C)", min_value=-5.0, max_value=60.0, value=28.6, step=0.1)
        with col2:
            st.markdown("**Room and outdoor conditions**")
            humidity = st.number_input("Relative humidity (%)", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
            outdoor = st.number_input("Outdoor temperature (°C)", min_value=-20.0, max_value=60.0, value=35.0, step=0.1)
            setpoint = st.number_input("AC set temperature (°C)", min_value=16.0, max_value=32.0, value=24.0, step=0.5)
            occupants = st.number_input("Number of occupants", min_value=0, max_value=100, value=2, step=1)
        with col3:
            st.markdown("**Air conditioner status**")
            ac_on = st.selectbox("Is the AC running?", options=["ON", "OFF"])
            runtime = st.number_input(
                "Minutes since current AC run began", min_value=0, max_value=1440,
                value=20, step=5, disabled=(ac_on == "OFF"),
            )
            st.caption("When AC is OFF, model input runtime is automatically set to zero.")
            st.caption("These predictions are advisory estimates only; this website does not control HVAC equipment.")
        submitted = st.form_submit_button("Predict + compare both horizons", type="primary", use_container_width=True)

    if submitted:
        if not all(model_status.values()):
            st.error("Cannot calculate both forecasts until both model files are available. Train the models from the sidebar.")
        elif max(abs(room - lag5), abs(lag5 - lag10), abs(lag10 - lag15)) > 5:
            st.warning("One of the 5-minute temperature changes exceeds 5°C. Check the input readings before relying on the forecast.")
        else:
            try:
                values = {
                    "room_temp_c": room,
                    "relative_humidity_pct": humidity,
                    "outdoor_temp_c": outdoor,
                    "ac_set_temp_c": setpoint,
                    "occupants": occupants,
                    "ac_running": ac_on == "ON",
                    "ac_runtime_min": runtime,
                    "room_temp_lag_5m_c": lag5,
                    "room_temp_lag_10m_c": lag10,
                    "room_temp_lag_15m_c": lag15,
                }
                X = make_single_feature_row(values)
                predictions = []
                for horizon in HORIZONS:
                    bundle = load_bundle(horizon)
                    prediction = float(bundle["model"].predict(X[FEATURE_COLUMNS])[0])
                    predictions.append({
                        "Horizon": f"{horizon} minutes",
                        "Minutes ahead": horizon,
                        "Predicted room temperature (°C)": prediction,
                        "Difference from current (°C)": prediction - room,
                        "Model": bundle.get("model_name", type(bundle["model"]).__name__),
                    })
                c30, c60 = st.columns(2)
                for item, col in zip(predictions, (c30, c60)):
                    with col:
                        st.metric(
                            f"Prediction in {item['Minutes ahead']} minutes",
                            f"{item['Predicted room temperature (°C)']:.2f} °C",
                            f"{item['Difference from current (°C)']:+.2f} °C vs now",
                        )
                        st.caption(f"Model: {item['Model']}")
                chart = pd.DataFrame(
                    {
                        "Time point": ["Now", "30 minutes", "60 minutes"],
                        "Room temperature (°C)": [room, predictions[0]["Predicted room temperature (°C)"], predictions[1]["Predicted room temperature (°C)"]],
                    }
                ).set_index("Time point")
                st.markdown("#### Forecast curve")
                st.line_chart(chart, use_container_width=True)
                st.dataframe(pd.DataFrame(predictions), hide_index=True, use_container_width=True)
                st.download_button(
                    "Download these predictions (CSV)",
                    data=pd.DataFrame(predictions).to_csv(index=False).encode("utf-8"),
                    file_name="rac_temperature_forecast.csv",
                    mime="text/csv",
                )
                st.caption("Model outputs are estimates. Accuracy depends on representative data, sensor quality and operating conditions; this app does not control the AC.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with evaluation_tab:
    st.subheader("Chronological hold-out evaluation")
    if not metrics:
        st.info("No valid reports/metrics.json file was found. Train the models from the sidebar to generate evaluation metrics.")
    else:
        st.caption("Metrics describe the test period from the last completed training run. Lower MAE and RMSE are better. Always compare with the simple persistence baseline.")
        st.write(f"Training dataset: `{metrics.get('data_path', 'unknown')}`")
        st.write(f"Split method: {metrics.get('split_method', 'not recorded')}")
        for horizon in HORIZONS:
            hdata = metrics.get("horizons", {}).get(str(horizon))
            if not hdata:
                st.warning(f"Metrics for the {horizon}-minute horizon are missing. Retrain the models.")
                continue
            selected = hdata.get("selected_model")
            score_map = hdata.get("test_scores", {})
            ai_score = score_map.get(selected, {})
            base_score = score_map.get("PersistenceBaseline_current_temp", {})
            st.markdown(f"### {horizon}-minute forecast")
            st.caption(f"Selected model: `{selected or 'unknown'}`")
            a, b, c, d = st.columns(4)
            a.metric("AI test MAE", f"{ai_score.get('mae_c', float('nan')):.3f} °C" if "mae_c" in ai_score else "N/A")
            b.metric("AI test RMSE", f"{ai_score.get('rmse_c', float('nan')):.3f} °C" if "rmse_c" in ai_score else "N/A")
            c.metric("Baseline MAE", f"{base_score.get('mae_c', float('nan')):.3f} °C" if "mae_c" in base_score else "N/A")
            d.metric("Test rows", f"{hdata.get('test_rows', 0):,}")
            if ai_score and base_score:
                delta = ai_score.get("mae_c", 0) - base_score.get("mae_c", 0)
                if delta < 0:
                    st.success(f"On this held-out test set, AI MAE is {abs(delta):.3f}°C lower than the persistence baseline.")
                else:
                    st.warning(f"On this held-out test set, AI MAE is {delta:.3f}°C higher than the persistence baseline.")
            validation_scores = hdata.get("validation_scores", {})
            if validation_scores:
                score_rows = [
                    {"Model": name, "Validation MAE (°C)": round(score.get("mae_c", float("nan")), 4), "Validation RMSE (°C)": round(score.get("rmse_c", float("nan")), 4)}
                    for name, score in validation_scores.items()
                ]
                st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)
            plot_path = REPORTS / f"actual_vs_predicted_{horizon}min.png"
            if plot_path.exists():
                st.image(str(plot_path), caption=f"Held-out test period — {horizon}-minute-ahead estimates")
            st.divider()
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download metrics JSON", data=json.dumps(metrics, indent=2), file_name="metrics.json", mime="application/json", use_container_width=True)
        with d2:
            pred_path = REPORTS / "test_predictions.csv"
            if pred_path.exists():
                st.download_button("Download test predictions CSV", data=pred_path.read_bytes(), file_name="test_predictions.csv", mime="text/csv", use_container_width=True)
        st.caption("These scores are not proof of real-world HVAC accuracy when the model was trained on simulated data.")

with dataset_tab:
    st.subheader("Dataset validation and exploration")
    if data is None:
        st.error("There is no valid dataset to explore yet.")
        st.markdown("Required columns are defined in `data/real_data_template.csv` and `data/README.md`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations", f"{len(data):,}")
        c2.metric("Start timestamp", fmt_timestamp(data.timestamp.min()))
        c3.metric("End timestamp", fmt_timestamp(data.timestamp.max()))
        cadence = data.timestamp.diff().dropna().dt.total_seconds() / 60
        typical_cadence = float(cadence.median()) if not cadence.empty else float("nan")
        c4.metric("Median interval", f"{typical_cadence:g} min" if pd.notna(typical_cadence) else "N/A")
        if not cadence.empty and (cadence != 5).any():
            st.warning(f"There are {int((cadence != 5).sum()):,} intervals that are not exactly 5 minutes. Exact lag features require observations at the corresponding timestamps.")
        st.write(f"Loaded file: `{active_file_label}`")
        st.markdown("**Room and outdoor measurements**")
        st.line_chart(data.set_index("timestamp")[["room_temp_c", "outdoor_temp_c", "ac_set_temp_c"]].tail(1000), use_container_width=True)
        with st.expander("View descriptive statistics"):
            st.dataframe(data.describe(include="all").transpose(), use_container_width=True)
        with st.expander("Preview most recent 25 rows"):
            st.dataframe(data.tail(25), hide_index=True, use_container_width=True)
        st.download_button(
            "Download selected dataset CSV",
            data=active_path.read_bytes(),
            file_name=active_file_label,
            mime="text/csv",
        )

with guide_tab:
    st.subheader("Start-to-finish workflow")
    st.markdown(
        """
1. **Install the requirements** in the project's virtual environment using the `START_HERE.md` instructions.
2. **Review the data schema** in `data/README.md`. The CSV requires timestamps, room/outdoor temperatures, humidity, AC setpoint/status/runtime and occupant count.
3. **Start with the bundled sample** to confirm the full site works. It is simulated and only demonstrates the pipeline.
4. **Upload measured data** collected at consistent 5-minute intervals from the same room. Check the Dataset explorer for gaps and validation warnings.
5. **Train/retrain models** using the sidebar. A chronological split is used so later timestamps are held out for validation/testing.
6. **Make a prediction** by entering current conditions and exact historical temperature lags. Download the forecast CSV when needed.
7. **Review model evaluation** and compare AI errors with the persistence baseline before writing any performance claims.

**Required CSV column names:** `timestamp`, `room_temp_c`, `relative_humidity_pct`, `outdoor_temp_c`, `ac_set_temp_c`, `occupants`, `ac_running`, `ac_runtime_min`.

**Important:** This is an educational prototype. Do not connect it directly to HVAC control hardware or use the predictions for safety-critical decisions.
        """
    )
    st.markdown("**Useful commands (run from the project root):**")
    st.code("python -m pip install -r requirements.txt\npython -m unittest discover -s tests -v\nstreamlit run app.py", language="powershell")
