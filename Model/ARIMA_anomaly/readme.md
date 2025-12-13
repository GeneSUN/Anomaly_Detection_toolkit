# Time Series Anomaly Detection with ARIMA

Prediction-based anomaly detection is a popular technique for time series: train a forecasting model, compare predictions vs. actual observations, and flag points that deviate significantly from expected behavior.

---


## What is prediction-based anomaly detection with ARIMA?

Prediction-based anomaly detection with ARIMA can be used in two common ways:

1. **Out-of-sample (future) forecasting anomaly detection**  
   Forecast future points and flag anomalies if new observations fall outside prediction intervals.

2. **In-sample (training data) reconstruction anomaly detection**  
   Fit the model on existing data, **reconstruct** the training data (in-sample fitted values), and flag points where the reconstructed values differ greatly from the observed values.



---
## 1. In-sample (training data) reconstruction anomaly detection

**Analogy:**  
Imagine a music teacher listening to a student play a familiar song. The teacher knows how the song should sound (the ARIMA model has learned the series pattern). If the student suddenly hits a very wrong note, it stands out — just like an anomaly deviating from the model’s expected behavior.

### Step 1: Create and fit the ARIMA forecasting model

We create a forecasting model using **AutoARIMA** from the **statsforecast** library. AutoARIMA automatically selects the best ARIMA parameters, reducing the need for manual tuning.

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

model = StatsForecast(
    models=[AutoARIMA(season_length=4)],
    freq='Q',
    n_jobs=-1
)
```

Notes:
- `season_length=4` is appropriate for **quarterly** data (4 quarters per year).
- For **daily**, **hourly**, or other frequencies, adjust `season_length` accordingly (e.g., 7 for weekly seasonality in daily data, 24 for daily seasonality in hourly data, etc.).
- `freq='Q'` sets the series frequency for StatsForecast.

---

### Step 2: Generate forecasts and prediction intervals

Once the model is set up, we generate:

- **Forecasts** using `.forecast()`
- **Prediction intervals** by setting `level=[99]` (a 99% confidence interval)
- **In-sample fitted values** by setting `fitted=True` and calling `.forecast_fitted_values()`

```python
forecast_df = model.forecast(df=df_arima, h=8, level=[99], fitted=True)
insample_forecast = model.forecast_fitted_values()
```

Key parameters:
- `h=8`: forecast horizon (8 steps ahead)
- `level=[99]`: 99% prediction interval (wider interval → fewer anomalies flagged)
- `fitted=True`: also compute in-sample fitted values
- `forecast_fitted_values()`: returns fitted values for training points

---

### Step 3: Flag anomalies based on prediction intervals

The core anomaly rule:

- If an observed value falls **outside** the prediction interval → flag as anomaly

```python
insample_forecast["anomaly"] = (
    (insample_forecast["y"] < insample_forecast["AutoARIMA-lo-99"]) |
    (insample_forecast["y"] > insample_forecast["AutoARIMA-hi-99"])
)
```

Notes:
- Using a **99%** interval is stricter (fewer points outside).
- Using a **95%** interval is more sensitive (more points flagged).

---

### Step 4: Visualize the results

Visualization is essential to validate whether the detected anomalies make sense (e.g., spikes, drops, shifts), and to sanity-check false positives caused by model misfit, seasonality mismatch, or non-stationarity.

<img width="1306" height="470" alt="image" src="https://github.com/user-attachments/assets/09e235af-bb57-4087-acc7-eb8fa1848731" />


---

### Complete pipeline (class example)

You can wrap the entire workflow into a class for clean reuse. Example usage (based on your repo structure):

```python
# from ARIMA_anomaly.ARIMAAnomalyDetector import ARIMAAnomalyDetector

detector = ARIMAAnomalyDetector(
    df=my_dataframe,
    time_col="timestamp",
    feature="value",
    season_length=4
)

detector.run()
detector.plot_anomalies()
```

Implementation reference in your repository:
- [ARIMA_anomaly/ARIMAAnomalyDetector.py](https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/ARIMA_anomaly/ARIMAAnomalyDetector.py)
- [colab notebook](https://colab.research.google.com/drive/1Gc7Em68p0ivqWJ98Cne7lyPb5TrTcZ-L)

---

## 2. Out-of-sample (future) forecasting anomaly detection


In addition to in-sample “reconstruction” anomaly detection, ARIMA can also be used for **out-of-sample (future) anomaly detection**. 
- The workflow is essentially the same idea:
   - fit an ARIMA/AutoARIMA model,
   - generate **future forecasts with prediction intervals**,
   - then compare each newly observed value against the interval. If the actual value falls outside the predicted range, it is flagged as an anomaly.

- The key difference is scope: **out-of-sample detection only evaluates future points** (values out of the trained model)
  
The distinction between in-sample and out-of-sample ARIMA anomaly detection mirrors the difference between outlier detection (within known data) and novelty detection (on unseen future data).

<img width="1589" height="490" alt="image" src="https://github.com/user-attachments/assets/b6cfa64d-0503-42a5-8666-2be70a08a093" />


- https://colab.research.google.com/drive/1Gc7Em68p0ivqWJ98Cne7lyPb5TrTcZ-L#scrollTo=2uuvAQgVVxuO

---

