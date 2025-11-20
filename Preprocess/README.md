# 📘 Feature Engineering for Real-Time ML Anomaly Detection



Real-time anomaly detection needs clean and model-ready data — especially when the upstream source is raw  device logs.  
This document explains the feature engineering workflow used in our 5G Home real-time novelty detection pipeline, following a simple Bronze → Silver → Gold structure.

- **Bronze:** Raw logs pulled from S3 into HDFS by Data Engineering  
- **Silver:** Cleaned, normalized, and reshaped device metrics  
- **Gold:** Final model features ready for real-time detection

<img width="800" height="1100" alt="image" src="https://github.com/user-attachments/assets/d709b2f7-29c6-4082-99d3-2783f2a9c323" />

The goal is to turn noisy device logs into reliable input features that support fast and accurate anomaly detection for millions of 5G Home customers.



## 🥉 1. Bronze Layer — Raw Data Ingestion (Data Engineering Layer)

The raw data is ingested by Data Engineering from S3 → HDFS, using scheduled ingestion jobs.

At this stage, no feature engineering occurs.
- Preserve data exactly as received
- Provide a stable foundation for downstream ML pipeline
- Timestamp + partition checks


## 🥈 2. Silver Layer — Preprocessing & Data Quality Enforcement

✔ Key Goals

1. Convert Strings to Numerical Types
2. Forward Fill (ffill): Used when instrumentation gaps or network delays produce missing values:
3. zeros are treated as nulls: like age, zero is not impossible; some machine use fill null with zero
  <img width="489" height="389" alt="image" src="https://github.com/user-attachments/assets/6e728555-4ad7-49dd-b7bc-58f36ed2a609" />

4. Aggregation (agg): Aggregate raw events into a fixed frequency (default: hourly):
  <img width="463" height="547" alt="image" src="https://github.com/user-attachments/assets/73c478e5-2b23-4247-8bfa-573918a2fd35" />

5. Differencing (diff): Compute smoothed increments
  
  <img width="405" height="547" alt="image" src="https://github.com/user-attachments/assets/5cb20652-7db1-4de7-a959-ef4108fa3edc" />

6. Log Transform (log)

  <img width="489" height="490" alt="image" src="https://github.com/user-attachments/assets/4da92a80-8c3a-4178-9bd6-bcaa081f4a6c" />

  



The full preprocessing implementation is available in the repository:

- **[TimeSeriesFeatureTransformerPandas.py](https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Preprocess/TimeSeriesFeatureTransformerPandas.py#L6
)**  
  → Core preprocessing logic (aggregation, diff, log1p, ffill, zero→null handling, etc.)  
  → [Source code](https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Preprocess/TimeSeriesFeatureTransformerPandas.py#L6
):  
  
To see how the transformer is used end-to-end, refer to the demo notebook:

- **[Colab Usage Example](https://colab.research.google.com/drive/1z6PtK9nOo6h2e05E_UZAjFHDD-esBA-T)**  
  → Walks through loading raw data, running the transformer, visualizing outputs, and preparing series for anomaly detection  
  → [Notebook](https://colab.research.google.com/drive/1z6PtK9nOo6h2e05E_UZAjFHDD-esBA-T):  
  



## 🟡 3. Gold Layer — Model-Ready Feature Tensor Construction

The Gold layer prepares data for real-time anomaly detection models, 

### sequence-based models (LSTM AE, GRU, CNN, ARIMA, seasonal decomposition, etc.).
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/Preprocess/TimeSeriesFeatureTransformerPandas.py#L177
- https://colab.research.google.com/drive/1z6PtK9nOo6h2e05E_UZAjFHDD-esBA-T#scrollTo=DNx790E7D097
  
<img width="503" height="990" alt="image" src="https://github.com/user-attachments/assets/d571d506-620f-4253-8c01-b016e80c5584" />

### unpivot transformation for distributed computing

```python

def unpivot_wide_to_long(df, time_col, feature_cols):
    # Build a stack(expr) for unpivot: (feature, value)
    n = len(feature_cols)
    expr = "stack({n}, {pairs}) as (feature, value)".format(
        n=n,
        pairs=", ".join([f"'{c}', `{c}`" for c in feature_cols])
    )
    return df.select("sn", time_col, F.expr(expr))
```

**❌ Original Wide Dataset (Not Suitable for Distributed Processing)**

| Serial Number | feature_1 | feature_2 | feature_3 |
|--------------|-----------|-----------|-----------|
| 1            | x1        | y1        | z1        |
| 2            | x2        | y2        | z2        |
| 3            | x3        | y3        | z3        |
| 4            | x4        | y4        | z4        |
| 5            | x5        | y5        | z5        |

Problems with this format: PySpark cannot **parallelize feature columns** — each feature becomes a separate column, not a separate row.

**✔ Unpivoted (Long Format) Dataset — Ready for Distributed Computing**

| Serial Number | feature     | value |
|--------------|-------------|-------|
| 1            | feature_1   | x1    |
| 1            | feature_2   | y1    |
| 1            | feature_3   | z1    |
| 2            | feature_1   | x2    |
| 2            | feature_2   | y2    |
| 2            | feature_3   | z2    |
| ...          | ...         | ...   |


Once unpivoted, PySpark can split the work, Executors can now process every feature in parallel:

- Executor A → all rows of `feature_1`

| Serial Number | feature   | value |
|--------------|-----------|-------|
| 1            | feature_1 | x1    |
| 5            | feature_1 | x2    |
| 10           | feature_1 | x3    |
  
- Executor B → all rows of `feature_2`
- Executor C → all rows of `feature_3`
- ...
- Executor N → all rows of `feature_k`



