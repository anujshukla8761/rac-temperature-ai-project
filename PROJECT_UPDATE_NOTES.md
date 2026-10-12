# Project update notes — RAC Temperature AI

This folder is a complete website-plus-model project snapshot. The original ZIP has been left unchanged; this is a separately packaged updated copy.

## What is included

- `app.py`: Streamlit dashboard connected directly to both saved model bundles.
- `artifacts/model_30min.joblib`: model for the 30-minute-ahead room-temperature estimate.
- `artifacts/model_60min.joblib`: model for the 60-minute-ahead room-temperature estimate.
- `rac_ai/`: CSV validation, time-aligned feature construction, training/evaluation, command-line prediction and logger code.
- `data/`: simulated demo dataset, CSV template and data notes.
- `reports/`: evaluation metrics, held-out predictions, comparison plots and report template.
- `tests/`: validation, feature alignment and inference checks.

## Dashboard improvements

- Adds an explicit forecast curve and a downloadable forecast CSV.
- Shows model availability, dataset provenance, chronological evaluation metrics, the persistence baseline and stored plots.
- Adds dataset interval summaries, descriptive statistics, row previews and CSV download.
- Stages uploaded CSV files in the operating system temporary directory rather than overwriting the bundled sample CSV.
- Requires an explicit confirmation before the dashboard retrains over existing model/report outputs.
- Continues to allow manual-input prediction if a different CSV is selected; it clearly warns that evaluation metrics belong to the previous training run until the selected dataset is used for retraining.
- Adds stronger CSV checks for infinite numbers, likely temperature/setpoint errors, duplicate timestamps, humidity range, occupancy and AC runtime consistency.
- Adds training-data SHA-256 metadata to models created by a future training run. Existing bundled model files have been preserved, not retrained or replaced during packaging.

## Verification completed for this package

- Python compilation check: passed.
- Unit tests: 8 passed.
- Inference smoke test using both bundled model files: passed.
- Training-pipeline smoke test on a temporary synthetic dataset: trained both horizons and created models/reports successfully; temporary outputs were discarded.
- Dashboard script smoke test with a minimal Streamlit API stub: passed.

A real browser/server launch was **not** verified in the build environment because Streamlit was not installed and network access prevented package installation. Install dependencies on your Windows computer with `python -m pip install -r requirements.txt`, then launch with `streamlit run app.py`.

## Data and accuracy warning

`data/sample_temperature_data.csv` is simulated teaching data. Existing model weights and report metrics are kept from the original archive. They demonstrate the workflow and are not experimental proof of real-room predictive accuracy. Collect representative measurements, retrain, and evaluate on a later held-out time period before reporting real-world performance.
