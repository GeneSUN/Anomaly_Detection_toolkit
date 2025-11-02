# Anomaly_Detection_toolkit
This file contains anomaly detection related script/model/automation, and explanation.

## Outline

1. [Univariate Anomaly Detection](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#1-univariate-single-feature)
    - [Extreme-Value (no time dependence)](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#a-extreme-value-no-time-dependence)
    - [Time-Series (with time dependence)](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#b-time-series-with-time-dependence)
    - [Unusual Shape (subsequence anomalies)](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#c-unusual-shape-subsequence-anomalies)
2. [Multivariate Anomaly Detection](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#2-multivariate-multiple-features)
    - [Proximity-Based](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#a-proximity-based-point-anomalies-in-feature-space)
3. [Outlier Ensembles](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#3-outlier-ensembles)
    - [Independent (Parallel) Ensembles](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#1-independent-parallel-ensembles)
    - [Sequential Ensembles](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#2-sequential-ensembles)
4. [Novelty Detection vs Outlier Detection](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#4-novelty-detection-vs-outlier-detection)
    - Outlier Detection
    - Novelty Detection
5. [Multi-Models Distributed Computing](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#5-multi-models-distributed-computing)  
    - Python Environment
    - Distributed Computing of Spark
6. [ML Toolkit](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#6-ml-toolkit)
    - [Preprocess](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#preprocess)
    - [Feature Engineering](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#feature-engineering-and-hyperparameter)
    - [Hyperparameter](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#feature-engineering-and-hyperparameter)
    - [Evaluation](https://github.com/GeneSUN/Anomaly_Detection_toolkit?tab=readme-ov-file#evaluation)
   
<img width="2254" height="278" alt="image" src="https://github.com/user-attachments/assets/5314b404-a9c9-4d00-b9fc-332ff2400c95" />

This diagram is a quick decision map for choosing an anomaly-detection approach based on <br>
- how many features you have  <br>
- whether you care about **individual points** or **subsequences/shapes**.

## 1) Univariate (single feature)

### A. Extreme-Value (no time dependence)
When observations are independent and you only care about unusually **large/small** values.
<img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/dea53d3a-c469-40e0-8f68-33702a91ed85" />

- **When to use:** No strong temporal correlation or seasonality.
- **Typical methods:** Robust z-score/quantiles, **Kernel Density Estimation (KDE)**, **Gaussian Mixture Models (GMM)**.
- **Output:** Point anomalies (outliers by value).

https://medium.com/@injure21/kernel-density-estimation-for-anomaly-detection-715a945bc729 <br>
This article explain kernel density estimation for anomaly detection <p>
https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/KDE/FeaturewiseKDENoveltyDetector.py <br>
This class defined the kernel density estimation model <p>
https://colab.research.google.com/drive/1qC-Gry8py_Icl0V8zNedlIeX3HFEKHuY#scrollTo=WXniSVfCznS_ <br>
This notebook use real-world example for step-by-step explanation of the article.<p>


### B. Time-Series (with time dependence)
When the current value depends on recent history and/or there is seasonality.
<p align="center">
    <img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/df1af6eb-0e9d-4791-8d34-a3ac4d0ae5ea" />
    <img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/e3eda4ce-7afa-415d-a012-6664dc0d6880" />
</p>


- **When to use:** Autocorrelation, trends, clear daily/weekly patterns.
- **Typical methods:** **Exponential Smoothing / ETS**, **ARIMA / SARIMA**, STL decomposition, residual-based thresholding.

[Time Series Anomaly Detection with ARIMA](https://medium.com/@injure21/time-series-anomaly-detection-with-arima-551a91d10fe4 )
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/ARIMA_anomaly/ARIMAAnomalyDetector.py <br>
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/ARIMA_anomaly/ARIMAAnomalyDetectorFuture.py <br>
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/ARIMA_anomaly/EWMAAnomalyDetector.py <br>
- https://colab.research.google.com/drive/1Gc7Em68p0ivqWJ98Cne7lyPb5TrTcZ-L#scrollTo=5CMO3pbLVvTt <br>

### C. Unusual Shape (subsequence anomalies)
When you care about **segments** that look abnormal, not just single points. <br>
<img width="700" height="400" alt="image" src="https://github.com/user-attachments/assets/c5fa911d-c74f-43fc-978f-834335527a0b" />

- **When to use:** Pattern/shape deviations over a window (e.g., `[xₜ,…,xₜ+W]`).
- **Typical methods:** **Autoencoders** (reconstruction error over windows), distance to shape prototypes, (optionally Matrix Profile/shapelet ideas).
- **Output:** Subsequence anomalies (unusual patterns over time).

[Autoencoders for Time Series Anomaly Detection: A Visual and Practical Guide](https://medium.com/@injure21/autoencoder-for-time-series-anomaly-detection-021d4b9c7909)
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/AutoEncoder/AutoencoderAnomalyDetector.py 
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/AutoEncoder/MultiTimeSeriesAutoencoder.py 
- https://colab.research.google.com/drive/174QBd3_2k3e88UyC__jLLG45Ukk-PMBx#scrollTo=DJ8JVhSGc70o 

---

## 2) Multivariate (multiple features)

### A. Proximity-Based (point anomalies in feature space)
When you care about the **overall state** across several features (e.g., 5G SNR, RSRP, RSRQ, …) rather than each feature separately.
<p float="left">
  <img src="https://github.com/user-attachments/assets/ea4c0c0c-b250-4265-9597-10bcce19d50d" width="40%" />
  <img src="https://github.com/user-attachments/assets/bc29ccc1-9ba6-43a5-b464-e2ca03750296" width="40%" />
</p>

- **When to use:** Joint behavior across features matters (correlations, clusters).
- **Typical methods:** **K-Means** (distance to centroid), **k-NN distance**, **LOF**, **DBSCAN**.
- **Output:** Point anomalies in high-dimensional space.
[Multivariate and Proximity-Based Detection](https://medium.com/@injure21/types-of-anomalies-in-data-part-2-value-based-detection-5ad9fabb30a7 )<br>
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/Proximity-based%20/KMeansOutlierDetector.py <br>
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/Proximity-based%20/KNNOutlierDetector.py <br>
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/Proximity-based%20/LOFOutlierDetector.py <br>
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/tree/main/Model/Proximity-based%20/DBSCANOutlierDetector.py <br>
- https://colab.research.google.com/drive/1ot_fdYbEyg8WVg7n_fADoI69TOS9a5P8#scrollTo=KD3jJx5Rh5dx <br>

### B. Multivariate Time-Series (temporal + cross-feature)
Extend proximity ideas with **time dependence** across **all** features.

### C. Multivariate Unusual Shape (subsequence anomalies across features)
Detect **segments** whose joint shape across features is unusual.

While extensions from (Proximity-Based ) A to **multivariate time series** (B) and **multivariate unusual shape detection** (C) provide richer modeling power, they also introduce higher complexity. As the models become more sophisticated, their **interpretability tends to decrease**.

---

## 3) Outlier Ensembles

There are two main ways to build ensembles for anomaly detection:
### 1. **Independent (Parallel) Ensembles**
- Each detector runs separately on the same data.  
- Results are combined at the end (e.g., by score averaging, voting).  
<img width="615" height="207" alt="image" src="https://github.com/user-attachments/assets/cc2e07e6-63d8-43bd-87c8-3c863469c61e" />

[Ensemble Methods for Outlier Detection](https://medium.com/@injure21/ensemble-methods-for-outlier-detection-79f9d9af4af0) <br>
[Independent, Model-Centric Ensemble](https://medium.com/@injure21/ensemble-methods-for-outlier-detection-8b4572a66fe7) <br>
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/Proximity-based%20/EnsembleOutlierDetector.py <br>
- https://colab.research.google.com/drive/1ot_fdYbEyg8WVg7n_fADoI69TOS9a5P8#scrollTo=OpaEvwmVvr5z <br>

### 2. **Sequential Ensembles**
- Detectors are applied one after another.  
- Each stage refines or filters the results from the previous stage.  
[Sequential Ensemble](https://medium.com/@injure21/ensemble-methods-for-outlier-detection-2-sequential-ensemble-abff0fae80bc)
- https://colab.research.google.com/drive/1LqBRw-p1OCP7VJ0Qn4qAA3i2H58_UCky
<img width="1043" height="91" alt="Screenshot 2025-09-22 at 10 50 44 AM" src="https://github.com/user-attachments/assets/2f4b8bcf-9cca-44e1-a5de-fd7d0d959af7" />

---

## 4) Novelty Detection vs Outlier Detection

- **Outlier Detection**  
  - Training and testing use the same dataset.  
  - The data contains outliers, defined as observations that are far from the majority.  
<img width="600" height="405" alt="image" src="https://github.com/user-attachments/assets/3b9cb55c-9d4e-4b57-93a1-3de82ed9c04b" />


- **Novelty Detection**  
  - Training and testing are different.  
  - The training data is *clean* (no outliers) and serves as a reference.  
  - New incoming observations are evaluated against this reference to detect anomalies.  

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/35a8a193-daea-4732-9c95-11e14b8c0f30" />

### This Repository is compatible for both Outlier Detection and Novelty Detection

This repository supports both **Outlier Detection** and **Novelty Detection**.  
You can control the behavior with the parameter `new_idx`:

- `new_idx = "all"` → Perform **Outlier Detection** (entire dataset used).  
- `new_idx = slice(-1, None)` → Perform **Novelty Detection** (detect only the last point).  
- `new_idx = slice(-n, None)` → Detect the last **n** points as novelties.


[Difference between Outlier Detection and Novelty Detection](https://medium.com/@injure21/difference-between-outlier-detection-and-novelty-detection-f21c21ed0962)
- https://colab.research.google.com/drive/1Gc7Em68p0ivqWJ98Cne7lyPb5TrTcZ-L#scrollTo=UKrOIuztVvzw

---

## 5) Multi-Models Distributed Computing

In real-world applications, you often need to run **many anomaly detection models in parallel**.  
For example, monitoring thousands of customers to detect irregular behavior in real time.  

This can be achieved in:  
- Python (single-machine environment): Suitable for smaller datasets or prototyping.  
- **Spark (distributed environment):** Recommended for large-scale scenarios, enabling parallel anomaly detection across millions of records efficiently.  <br>
<img width="1956" height="366" alt="image" src="https://github.com/user-attachments/assets/9e1712f9-6af2-4101-a494-e5049c2bbf97" />

[A practical guide to parallelizing thousands of models — from pandas to PySpark to GPU acceleration](https://medium.com/@injure21/scaling-time-series-modeling-spark-multiprocessing-and-gpu-side-by-sid-e353445ae205)

- https://colab.research.google.com/drive/1OA3EKXqiuMsQ5loQM7MlJVl_OuBpWAAt#scrollTo=32fjpkeS-nYP
- https://colab.research.google.com/drive/1qC-Gry8py_Icl0V8zNedlIeX3HFEKHuY#scrollTo=d82NG7mNxm0c
- https://colab.research.google.com/drive/1qC-Gry8py_Icl0V8zNedlIeX3HFEKHuY#scrollTo=RV6KSqCIP-U-

## 6) ML Toolkit
### Preprocess

- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Preprocess/TimeSeries_Preprocess.ipynb
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Preprocess/TimeSeriesFeatureTransformerPandas.py

### Feature Engineering and Hyperparameter
- [Feature Engineering for Unsupervised Outlier Detection: Practical Strategies and Pitfalls](https://medium.com/@injure21/feature-engineering-for-unsupervised-outlier-detection-practical-strategies-and-pitfalls-30a1155e5853)


### Evaluation
- [Evaluation of Unsupervised Outlier Detection](https://medium.com/@injure21/outlier-detection-evaluation-4e58439f8299)


## 7) Reference
- [**ADTK**](https://adtk.readthedocs.io/en/stable/) — A **versatile** Python toolkit for rule-based and statistical anomaly detection in time series.  
- [**Awesome Time Series Anomaly Detection**](https://github.com/rob-med/awesome-TS-anomaly-detection) — A curated list of resources, libraries, and papers focused on **time series** anomaly detection.  
- [**Anomalib**](https://github.com/open-edge-platform/anomalib) — A **deep learning–based** library offering state-of-the-art algorithms and pipelines for anomaly detection and localization.

## 8) Challenge/Trade-off

1. High Feature Variability complicate model generalization:
    - The 50+ features vary widely in distribution, magnitude, and pattern.
    - A single model with fixed hyperparameters does not generalize well across all features.
    - Customizing hyperparameters or models for each feature is computationally intensive.

2. Large-scale personalized models are computationally heavy:
    - anomaly detection should be tailored per customer per feature using their historical data.
    - This is computationally expensive for million users 50+ feature and infeasible at scale.

3. Lack of Ground Truth:
    - This is an unsupervised learning problem—no labels are available.
    - It’s hard to **evaluate model performance** or **fine-tune hyperparameters**.

**Solution:**
- [Multi-Models Distributed Computing](https://github.com/GeneSUN/Anomaly_Detection_toolkit/edit/main/README.md#5-multi-models-distributed-computing)
- Trade-off Alternatives
