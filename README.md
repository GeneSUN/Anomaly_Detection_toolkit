# Anomaly_Detection_toolkit
this file contains anomaly detection related script/model/automation, and explanation.
<img width="2254" height="278" alt="image" src="https://github.com/user-attachments/assets/5314b404-a9c9-4d00-b9fc-332ff2400c95" />


This diagram is a quick decision map for choosing an anomaly-detection approach based on (1) how many features you have and (2) whether you care about **individual points** or **subsequences/shapes**.

## 1) Univariate (single feature)

### A. Extreme-Value (no time dependence)
When observations are independent and you only care about unusually **large/small** values.

- **When to use:** No strong temporal correlation or seasonality.
- **Typical methods:** Robust z-score/quantiles, **Kernel Density Estimation (KDE)**, **Gaussian Mixture Models (GMM)**.
- **Output:** Point anomalies (outliers by value).

### B. Time-Series (with time dependence)
When the current value depends on recent history and/or there is seasonality.

- **When to use:** Autocorrelation, trends, clear daily/weekly patterns.
- **Typical methods:** **Exponential Smoothing / ETS**, **ARIMA / SARIMA**, STL decomposition, residual-based thresholding.
- **Output:** Point anomalies relative to a time-aware forecast/baseline.

### C. Unusual Shape (subsequence anomalies)
When you care about **segments** that look abnormal (e.g., an entire day that deviates from recent days), not just single points.

- **When to use:** Pattern/shape deviations over a window (e.g., `[xₜ,…,xₜ+W]`).
- **Typical methods:** **Autoencoders** (reconstruction error over windows), distance to shape prototypes, (optionally Matrix Profile/shapelet ideas).
- **Output:** Subsequence anomalies (unusual patterns over time).
