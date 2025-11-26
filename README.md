# Anomaly_Detection_toolkit
This repository contains anomaly detection–related scripts, models, automation utilities, and explanatory documentation.


<img width="2254" height="278" alt="image" src="https://github.com/user-attachments/assets/5314b404-a9c9-4d00-b9fc-332ff2400c95" />

This diagram provides a quick decision guide for selecting an anomaly detection approach based on:  
- **number of features**  
- whether you care about **individual points** or **subsequences/shapes**

## Table of Contents
- [1. Univariate Anomaly Detection](#1-univariate-single-feature)
  - [A. Extreme-Value (no time dependence)](#a-extreme-value-no-time-dependence)
  - [B. Time-Series (with time dependence)](#b-time-series-with-time-dependence)
  - [C. Unusual Shape (subsequence anomalies)](#c-unusual-shape-subsequence-anomalies)
- [2. Multivariate Anomaly Detection](#2-multivariate-multiple-features)
  - [A. Proximity-Based](#a-proximity-based-point-anomalies-in-feature-space)
  - [B. Multivariate Time-Series](#b-multivariate-time-series-temporal--cross-feature)
  - [C. Multivariate Unusual Shape](#c-multivariate-unusual-shape-subsequence-anomalies-across-features)
- [3. Outlier Ensembles](#3-outlier-ensembles)
  - [1. Independent (Parallel) Ensembles](#1-independent-parallel-ensembles)
  - [2. Sequential Ensembles](#2-sequential-ensembles)
- [4. Novelty Detection vs Outlier Detection](#4-novelty-detection-vs-outlier-detection)
- [5. Multi-Models Distributed Computing](#5-multi-models-distributed-computing)
- [6. ML Toolkit](#6-ml-toolkit)
  - [Preprocess](#preprocess)
  - [Feature Selection and Hyperparameter Tuning](#feature-selection-and-hyperparameter)
  - [Evaluation](#evaluation)
- [7. Model Library](#7-model-library)
- [8. Challenge / Trade-off](#8-challengetrade-off)
---

## 1) Univariate (single feature)

### A. Extreme-Value (no time dependence)
Suitable when observations are independent and only extreme **high/low** values matter.
<img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/dea53d3a-c469-40e0-8f68-33702a91ed85" />

- **When to use:** No strong temporal correlation or seasonality  
- **Typical methods:** Robust z-score/quantiles, **KDE**, **Gaussian Mixture Models (GMM)**
- **Output:** Point anomalies (based only on magnitude)

References:  
- Kernel density explanation: https://medium.com/@injure21/kernel-density-estimation-for-anomaly-detection-715a945bc729  
- KDE class: https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/KDE/FeaturewiseKDENoveltyDetector.py  
- Real-world notebook: https://colab.research.google.com/drive/1qC-Gry8py_Icl0V8zNedlIeX3HFEKHuY#scrollTo=WXniSVfCznS_

---

### B. Time-Series (with time dependence)
Used when current values depend on recent history and/or seasonality.
<p align="center">
    <img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/df1af6eb-0e9d-4791-8d34-a3ac4d0ae5ea" />
    <img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/e3eda4ce-7afa-415d-a012-6664dc0d6880" />
</p>

- **When to use:** Autocorrelation, trends, repeating daily/weekly patterns  
- **Typical methods:** **ETS**, **ARIMA/SARIMA**, STL decomposition, residual-based thresholding

References:  
- Article: https://medium.com/@injure21/time-series-anomaly-detection-with-arima-551a91d10fe4  
- ARIMA detectors:  
  - https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/ARIMA_anomaly/ARIMAAnomalyDetector.py  
  - https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/ARIMA_anomaly/ARIMAAnomalyDetectorFuture.py  
  - https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/ARIMA_anomaly/EWMAAnomalyDetector.py  
- End-to-end notebook: https://colab.research.google.com/drive/1Gc7Em68p0ivqWJ98Cne7lyPb5TrTcZ-L#scrollTo=5CMO3pbLVvTt

---

### C. Unusual Shape (subsequence anomalies)
Focuses on detecting **segments** that deviate from typical patterns.
<img width="700" height="400" alt="image" src="https://github.com/user-attachments/assets/c5fa911d-c74f-43fc-978f-834335527a0b" />

- **When to use:** Pattern/shape deviations over a window (e.g., `[xₜ … xₜ+W]`)  
- **Typical methods:** **Autoencoders**, shape distance, matrix profile concepts  
- **Output:** Subsequence anomalies

References:  
- Article: https://medium.com/@injure21/autoencoder-for-time-series-anomaly-detection-021d4b9c7909  
- Code:  
  - https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/AutoEncoder/AutoencoderAnomalyDetector.py  
  - https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/AutoEncoder/MultiTimeSeriesAutoencoder.py  
- Notebook: https://colab.research.google.com/drive/174QBd3_2k3e88UyC__jLLG45Ukk-PMBx#scrollTo=DJ8JVhSGc70o  

---

## 2) Multivariate (multiple features)

### A. Proximity-Based (point anomalies in feature space)
Detects points that are far from the typical cluster/centroid in a high-dimensional feature space.
<p float="left">
  <img src="https://github.com/user-attachments/assets/ea4c0c0c-b250-4265-9597-10bcce19d50d" width="40%" />
  <img src="https://github.com/user-attachments/assets/bc29ccc1-9ba6-43a5-b464-e2ca03750296" width="40%" />
</p>

- **When to use:** Joint feature behavior matters (correlations, clusters)
- **Typical methods:** **K-Means**, **k-NN distance**, **LOF**, **DBSCAN**
- **Output:** High-dimensional point anomalies

References:  
- Article: https://medium.com/@injure21/types-of-anomalies-in-data-part-2-value-based-detection-5ad9fabb30a7  
- Code:  
  - https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/Proximity-based%20/KMeansOutlierDetector.py  
  - https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/Proximity-based%20/KNNOutlierDetector.py  
  - https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/Proximity-based%20/LOFOutlierDetector.py  
  - https://github.com/GeneSUN/Anomaly_Detection_toolkit/tree/main/Model/Proximity-based%20/DBSCANOutlierDetector.py  
- Notebook: https://colab.research.google.com/drive/1ot_fdYbEyg8WVg7n_fADoI69TOS9a5P8#scrollTo=KD3jJx5Rh5dx  

### B. Multivariate Time-Series (temporal + cross-feature)
Extends proximity-based methods to include temporal dependence across all features.

### C. Multivariate Unusual Shape (subsequence anomalies across features)
Detects segments whose joint shape across features is abnormal.

As models progress from (A) to (B) and (C), power increases but interpretability decreases.

---

## 3) Outlier Ensembles

Two major ensemble strategies:

### 1. **Independent (Parallel) Ensembles**
- Each detector runs on the same data independently  
- Final anomaly score is aggregated (averaging, voting)
<img width="615" height="207" alt="image" src="https://github.com/user-attachments/assets/cc2e07e6-63d8-43bd-87c8-3c863469c61e" />

References:  
- Articles:  
  - https://medium.com/@injure21/ensemble-methods-for-outlier-detection-79f9d9af4af0  
  - https://medium.com/@injure21/ensemble-methods-for-outlier-detection-8b4572a66fe7  
- Code: https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Model/Proximity-based%20/EnsembleOutlierDetector.py  
- Notebook: https://colab.research.google.com/drive/1ot_fdYbEyg8WVg7n_fADoI69TOS9a5P8#scrollTo=OpaEvwmVvr5z  

### 2. **Sequential Ensembles**
- Detectors are applied in sequence  
- Each stage refines output of the previous
<img width="1043" height="91" alt="Screenshot 2025-09-22 at 10 50 44 AM" src="https://github.com/user-attachments/assets/2f4b8bcf-9cca-44e1-a5de-fd7d0d959af7" />

References:  
- Article: https://medium.com/@injure21/ensemble-methods-for-outlier-detection-2-sequential-ensemble-abff0fae80bc  
- Notebook: https://colab.research.google.com/drive/1LqBRw-p1OCP7VJ0Qn4qAA3i2H58_UCky

---

## 4) Novelty Detection vs Outlier Detection

- **Outlier Detection**  
  - Training and testing use the same dataset  
  - Dataset contains anomalous samples  
  <img width="600" height="405" alt="image" src="https://github.com/user-attachments/assets/3b9cb55c-9d4e-4b57-93a1-3de82ed9c04b" />

- **Novelty Detection**  
  - Training uses clean data  
  - New incoming observations are compared against this reference  
  <img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/35a8a193-daea-4732-9c95-11e14b8c0f30" />

### This Repository is Compatible with Both

You can control mode using `new_idx`:
- `new_idx = "all"` → **Outlier Detection**  
- `new_idx = slice(-1, None)` → **Novelty Detection (last point)**  
- `new_idx = slice(-n, None)` → Detect last **n** points as novelties

References:  
- Article: https://medium.com/@injure21/difference-between-outlier-detection-and-novelty-detection-f21c21ed0962  
- Notebook: https://colab.research.google.com/drive/1Gc7Em68p0ivqWJ98Cne7lyPb5TrTcZ-L#scrollTo=UKrOIuztVvzw  

---

## 5) Multi-Models Distributed Computing

In production, anomaly detection may need to run across **thousands of entities** in parallel (e.g., customers, devices, cells).

Two environments:
- Python (single machine)
- **Spark** (distributed, scalable to millions of records)
<img width="1956" height="366" alt="image" src="https://github.com/user-attachments/assets/9e1712f9-6af2-4101-a494-e5049c2bbf97" />

Reference article:  
- https://medium.com/@injure21/scaling-time-series-modeling-spark-multiprocessing-and-gpu-side-by-sid-e353445ae205
- https://github.com/GeneSUN/spark-performance-toolbox/blob/main/Distributed_Computing/Spark%20Packaging%20&%20Deploying%20Custom%20Module.md
  
Notebooks:  
- Spark: https://colab.research.google.com/drive/1OA3EKXqiuMsQ5loQM7MlJVl_OuBpWAAt
- Python: https://colab.research.google.com/drive/1qC-Gry8py_Icl0V8zNedlIeX3HFEKHuY


---

## 6) ML Toolkit

### Preprocess

Real-time anomaly detection needs clean, consistent, and model-ready data, this [document](https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Preprocess/feature_pipeline_for_realtime_anomaly_detection.md
) explains the feature engineering workflow used in our real-time novelty detection pipeline, following a simple Bronze → Silver → Gold structure.


- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Preprocess/TimeSeries_Preprocess.ipynb
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Preprocess/TimeSeriesFeatureTransformerPandas.py

### Feature Selection and Hyperparameter
- https://medium.com/@injure21/feature-engineering-for-unsupervised-outlier-detection-practical-strategies-and-pitfalls-30a1155e5853

### Evaluation
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Evaluation/Evaluation%20of%20Unsupervised%20Outlier%20Detection.md

---

## 7) Model Library
- [**ADTK**](https://adtk.readthedocs.io/en/stable/) — versatile toolkit for rule-based/statistical time-series anomaly detection  
- [**Awesome Time Series Anomaly Detection**](https://github.com/rob-med/awesome-TS-anomaly-detection) — curated list of libraries and papers  
- [**Anomalib**](https://github.com/open-edge-platform/anomalib) — deep-learning based SOTA models for anomaly detection and localization  
- [**PyOD**](https://pyod.readthedocs.io/en/latest/)/[**PyGOD**](https://docs.pygod.org/en/latest/) — traditional ML/DL/graph outlier detection tools  

---

## 8) Challenge/Trade-off

1. **High feature variability complicates model generalization**
    - 50+ features differ in distribution, magnitude, and pattern  
    - One fixed model/hyperparameter set rarely works for all  
    - Fully customized models per feature are expensive

2. **Large-scale personalized models are computationally heavy**
    - Ideally each user and each feature should have a tailored model  
    - Doing this for millions of users × 50+ features is computationally prohibitive

3. **No ground truth labels**
    - Unsupervised problem
    - Hard to tune or evaluate performance

<img width="2686" height="1296" alt="image" src="https://github.com/user-attachments/assets/4cc2acee-f03c-47ac-83ee-29482b310c64" />

**Solution:**  
- [Multi-Models Distributed Computing](https://github.com/GeneSUN/Anomaly_Detection_toolkit/edit/main/README.md#5-multi-models-distributed-computing)  
- Practical trade-offs are discussed in the article and notebooks

## License
This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.
