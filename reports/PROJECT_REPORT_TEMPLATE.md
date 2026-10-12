# AI-Based Room Temperature Prediction Using Machine Learning

## 1. Abstract
This project develops a machine-learning workflow to predict room temperature 30 and 60 minutes in advance using indoor/outdoor conditions and air-conditioner operating information. Replace this template with a short summary of your actual experiment. State whether the dataset is simulated or experimentally measured.

## 2. Problem Statement
Can current room temperature, relative humidity, outdoor temperature, AC setpoint, occupancy, AC runtime, and recent temperature history be used to estimate future room temperature?

## 3. Objectives
1. Understand the relevant RAC concepts, including heat transfer, cooling load, and psychrometrics.
2. Collect and validate timestamped room-condition data.
3. Engineer past-temperature features without using future information as model inputs.
4. Train and compare Linear Regression and Random Forest Regression.
5. Evaluate 30- and 60-minute forecasts on a later chronological test period.

## 4. RAC Theory
Explain sensible and latent heat, indoor heat gains, cooling capacity, the effect of outdoor conditions and occupancy, and how humidity relates to comfort. Explain that air temperature prediction alone is not a complete comfort assessment.

## 5. Equipment and Software
List the actual thermometer/hygrometer, outdoor-temperature measurement source, AC system, laptop, Python version, and libraries used. Add setup photographs if you performed physical measurements.

## 6. Data Collection Method
- Location/room description (avoid publishing private addresses)
- Measurement dates and sampling interval
- Sensor model, placement and stated accuracy
- How occupancy, AC setpoint, ON/OFF, and runtime were recorded
- Missing readings and data-cleaning decisions
- Clearly mark simulated data separately from real measurements

## 7. Dataset Description
Describe columns, units, number of observations, and sample period. Include descriptive statistics and room/outdoor temperature plots.

## 8. Methodology
1. Sort measurements by timestamp and validate required columns.
2. Construct 5-, 10-, and 15-minute temperature lag features using exact timestamps.
3. Create targets using measured room temperature at t+30 and t+60 minutes.
4. Split data chronologically into training, validation and test periods.
5. Select between Linear Regression and Random Forest using validation MAE.
6. Compare final test performance with persistence baseline (future temperature equals current temperature).

## 9. Results
Insert the metrics from `reports/metrics.json` and plots from `reports/actual_vs_predicted_30min.png` and `reports/actual_vs_predicted_60min.png`. Report MAE and RMSE with °C units and include baseline scores. Do not invent or change experimental results.

## 10. Discussion
Explain where the model performs well/poorly, whether it beats the persistence baseline, and possible causes of errors: changing outdoor conditions, open doors/windows, sun exposure, sensor quality, AC cycling, varying occupancy, and limited training data.

## 11. Limitations and Safety
This prototype predicts temperature; it does not control the AC or ensure comfort, safety, or energy savings. Do not connect it to mains voltage or HVAC control hardware as part of this beginner project. Real-use deployment requires reliable data, independent validation, and appropriate engineering review.

## 12. Conclusion and Future Work
Summarize measured results honestly. Future extensions may include power meters for energy prediction, more sensors, improved validation across seasons/rooms, humidity/comfort modelling, and only later controlled energy optimization.

## 13. References
List course textbooks, sensor datasheets, scikit-learn documentation, and sources actually used.
