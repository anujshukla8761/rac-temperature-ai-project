# AI-Based Room Temperature Prediction (RAC) — Beginner Project

A complete starter project for predicting room temperature **30 minutes and 60 minutes ahead** from timestamped room conditions and AC operation. It includes an illustrative simulated dataset, data validation, feature engineering, training/evaluation, a command-line predictor, a manual data logger, a Streamlit dashboard, tests, and a report template.

> **Important:** `data/sample_temperature_data.csv` is simulated teaching data, not measurements from a real room. Any accuracy results from it demonstrate the software workflow only. For real conclusions, collect real data, retrain the models, and evaluate on a later time period that was not used in training.

## What the project predicts

At measurement time *t*, use the current measurement and recent temperature history to predict the room temperature at *t + 30 minutes* and *t + 60 minutes*.

Inputs include room temperature, relative humidity, outdoor temperature, AC set temperature, occupants, AC ON/OFF status, minutes since the current AC run started, and room temperatures from 5, 10, and 15 minutes earlier. Recent-temperature features are calculated from earlier rows in the CSV.

## Project structure

```text
rac_temperature_ai/
├── app.py                         # Interactive Streamlit dashboard
├── requirements.txt
├── README.md
├── data/
│   ├── sample_temperature_data.csv # SIMULATED example data
│   ├── real_data_template.csv      # Header-only template for real readings
│   └── README.md                  # Data collection instructions
├── rac_ai/
│   ├── data.py                    # CSV checks and timestamp handling
│   ├── features.py                # Past-temperature features and future targets
│   ├── synthetic.py               # Re-create simulated practice data
│   ├── modeling.py                # Train/validation/test workflow
│   ├── train.py                   # Training command
│   ├── predict.py                 # Predict from the newest CSV row
│   └── logger.py                  # Manual readings logger
├── artifacts/                     # Bundled 30- and 60-minute trained model files
├── reports/                       # Metrics, plots, and report template
└── tests/                         # Data, feature, and bundled-model inference tests
```

## A. Install on Windows (PowerShell)

1. Extract the ZIP contents directly into `C:\Users\prakh\Desktop\VMs\projects vm for python\rac_temperature_ai_project`. The files `app.py`, `requirements.txt`, `rac_ai`, `data`, `artifacts`, and `reports` must sit directly in that project root.
2. Open PowerShell and move to the project folder:

```powershell
Set-Location -LiteralPath "C:\Users\prakh\Desktop\VMs\projects vm for python\rac_temperature_ai_project"
```

3. Create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the environment interpreter directly instead. From the project root, use ` .\.venv\Scripts\python.exe -m pip install -r requirements.txt` and replace `python` in later commands with ` .\.venv\Scripts\python.exe` (remove the leading spaces inside both commands).

## B. Train and run the first model

Run these commands from the project root, with the virtual environment active:

```powershell
python -m rac_ai.train --data data/sample_temperature_data.csv
python -m rac_ai.predict --data data/sample_temperature_data.csv
```

Training creates model files in `artifacts/`, an evaluation report in `reports/metrics.json`, test-set predictions in `reports/test_predictions.csv`, and actual-vs-predicted plots. Model choice is based on a chronological validation period; test metrics are calculated on a later period. A persistence baseline (predicting that the temperature stays at its current value) is included for comparison.

## C. Open the integrated dashboard

```powershell
streamlit run app.py
```

Streamlit prints a local address (usually `http://localhost:8501`). Open it in your browser. The website directly loads the bundled `artifacts/model_30min.joblib` and `artifacts/model_60min.joblib` files; a separate manual website-to-model integration step is not required. The dashboard includes manual forecasting, the forecast curve, evaluation metrics and plots, CSV upload/validation, dataset exploration, and CSV/JSON downloads.

Uploaded CSV files are staged under the operating system's temporary folder rather than overwriting the bundled `data/sample_temperature_data.csv`. The sidebar requires an explicit confirmation before its training button replaces existing model/report outputs. The command-line training command in Section E still overwrites the output model/report names, so back them up first if you need to preserve them. **The bundled model's predictions are demonstrations only because the bundled dataset is simulated.**

## D. Collect real readings (no sensor hardware required)

The logger asks you for measurements and writes them to a CSV file:

```powershell
python -m rac_ai.logger --output data/real_temperature_data.csv --interval-minutes 5
```

Enter readings at each prompt, then press Enter to save. Keep the logger running; it waits for the next interval. Press `Ctrl+C` when finished. Room temperature and humidity can come from a thermometer/hygrometer; measure outdoor temperature in shade or use a reliable local measurement source. Record the actual AC setpoint and occupancy. Do not guess values when collecting an experiment.

To stop after a single observation (useful for testing the logger):

```powershell
python -m rac_ai.logger --output data/real_temperature_data.csv --once
```

For learning and evaluation, collect data over multiple days, with AC operating in different conditions. Record at a consistent 5-minute interval. The logger's `ac_runtime_min` is the time since the current continuous ON period began, approximated from the logger interval.

## E. Train on real data

After collecting enough observations, train using the real file, not the sample file:

```powershell
python -m rac_ai.train --data data/real_temperature_data.csv
python -m rac_ai.predict --data data/real_temperature_data.csv
```

Training expects the required columns listed below and timestamps at 5-minute increments. Missing timestamps are allowed, but rows lacking an exact 5/10/15-minute history or exact future target are excluded; see the training output for usable sample counts. As a practical starting point, collect several days to a few weeks of data and aim for hundreds to thousands of usable rows. More data does not automatically guarantee a good model; varied, representative conditions and accurate measurements matter.

### Required CSV columns

| Column | Meaning | Example |
|---|---|---:|
| `timestamp` | Local date/time of observation | `2026-10-12 10:00:00` |
| `room_temp_c` | Indoor room air temperature, °C | `28.4` |
| `relative_humidity_pct` | Indoor relative humidity, % | `57` |
| `outdoor_temp_c` | Outdoor air temperature, °C | `35.2` |
| `ac_set_temp_c` | AC thermostat setpoint, °C | `24` |
| `occupants` | Number of people in room | `2` |
| `ac_running` | 1 if AC is running, otherwise 0 | `1` |
| `ac_runtime_min` | Minutes since current AC ON period began; 0 when OFF | `25` |

Do not manually enter a future temperature as an input. The program matches each timestamp to the measured temperature exactly 30 and 60 minutes later to create labels; it never uses these future labels as model inputs.

## F. How to interpret results

- **MAE (°C):** average absolute difference between predicted and measured future temperature. Lower is better.
- **RMSE (°C):** penalizes larger errors more heavily. Lower is better.
- **Persistence baseline:** predicts that the future temperature equals the current temperature. Your AI model should be compared with this baseline.
- **Chronological split:** older observations are used for training/validation, and later observations are held for testing. Do not randomly shuffle time-series rows.

There is no promised accuracy or energy saving. This project predicts temperature only; it does not directly control the AC or prove electricity savings. To study energy savings, add reliable power-consumption measurements and perform controlled comparisons while maintaining thermal comfort.

## G. Run the tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover CSV structure and validation errors, exact timestamp-aligned lag/target construction, single-row feature schema, and actual inference using both bundled model artifacts.

## H. Troubleshooting

- **`python` not found:** use `py -3` or confirm Python is installed and on PATH.
- **Activation blocked:** use `.venv\Scripts\python.exe` directly as shown above.
- **Not enough rows:** collect more readings. The training code needs enough valid rows in training, validation, and test periods after lag and future-target alignment.
- **Invalid/mixed timestamps or duplicates:** fix the CSV timestamps and ensure only one reading per timestamp.
- **Bad predictions:** check sensor placement/calibration, consistent sampling, seasonal/occupancy variation, AC runtime values, and test metrics. Do not use simulated-data metrics as real-world evidence.

## Academic deliverables

See `reports/PROJECT_REPORT_TEMPLATE.md` for a report structure. Include your actual data-collection dates, setup photos, plots, model comparison, limitations, and measured test metrics. Clearly state which data are simulated and which are experimentally measured.
