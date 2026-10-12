# Start Here — RAC Temperature AI

This ZIP is a complete project folder containing the Streamlit website, the two trained ML models, sample data, test code, and reports. You do not need to manually connect a separate website to the model: `app.py` loads `artifacts/model_30min.joblib` and `artifacts/model_60min.joblib`.

## 1. Extract the ZIP into your project folder

Extract the **contents** of this ZIP into:

```text
C:\Users\prakh\Desktop\VMs\projects vm for python\rac_temperature_ai_project
```

After extraction, `app.py`, `requirements.txt`, `rac_ai`, `data`, `artifacts`, and `reports` should be directly inside this folder. Do not extract the project into an additional nested folder with the same name.

## 2. Open PowerShell in the project root

Copy and run these commands one by one:

```powershell
Set-Location -LiteralPath "C:\Users\prakh\Desktop\VMs\projects vm for python\rac_temperature_ai_project"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, run this once in that PowerShell window and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Package installation needs an internet connection.

## 3. Test the project

```powershell
python -m unittest discover -s tests -v
```

The project includes tests for CSV validation, exact time-based features, and inference from both bundled trained models.

## 4. Start the website

```powershell
streamlit run app.py
```

Open the local address Streamlit prints (usually `http://localhost:8501`). Keep the PowerShell window open while using the website. Press `Ctrl+C` there to stop it.

## 5. Use the website

1. Open **Predict temperature** and enter current room conditions plus room temperatures from exactly 5, 10 and 15 minutes earlier.
2. The website loads the saved models and predicts the temperature 30 and 60 minutes ahead. You can download the forecast as CSV.
3. Open **Model evaluation** to review chronological test errors, the simple persistence baseline and available reports.
4. Open **Dataset explorer** to review timestamps, intervals, measurements and download the selected CSV.
5. To use your own measurements, upload the CSV in the sidebar. The upload is staged in the operating system's temporary folder; it does not replace the bundled sample CSV. To train new models on it, tick the explicit replacement confirmation and click the training button.

**Important:** The bundled CSV is simulated teaching data. The bundled model predictions are a demonstration of the working website/model connection, not evidence of real-world HVAC accuracy. Collect real data and retrain/evaluate before making performance claims.

## 6. Optional command-line workflow

From the project root, you can also run:

```powershell
python -m rac_ai.train --data data/sample_temperature_data.csv
python -m rac_ai.predict --data data/sample_temperature_data.csv
```

These command-line training commands replace model/report outputs after successful training. Keep a backup first if you need to preserve the current trained models. See `README.md` for the CSV schema and explanations.
