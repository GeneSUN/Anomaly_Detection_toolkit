# http://njbbvmaspd13:18080/#/notebook/2KW77G1JD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union

from pyspark.sql.functions import sum, lag, col, split, concat_ws, lit ,udf,count, max,lit,avg, when,concat_ws,to_date,explode
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pyod.models.auto_encoder_torch import AutoEncoder
#spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")


class AutoencoderAnomalyDetector:
    def __init__(self, 
                 df: pd.DataFrame, 
                 time_col: str, 
                 feature: str, 
                 window_size: int = 24,
                 overlap: float = 0.5,
                 model_params: Optional[dict] = None,
                 model: Optional[object] = None,
                 scaler: Union[str, object, None] = "standard",
                 threshold_percentile: float = 99
                 ):
        """
        Autoencoder-based anomaly detector for univariate time series.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        time_col : str
            Column containing timestamps.
        feature : str
            Column containing the target feature (numeric).
        window_size : int
            Length of each sub-series window.
        overlap : float
            Fractional overlap between windows (0=no overlap, 0.5=50% overlap, etc.).
        model_params : dict, optional
            Parameters for the default autoencoder.
        model : object, optional
            Custom model (must implement fit + decision_function).
        scaler : {'standard','minmax',object,None}
            Scaler type. Can be a string, custom scaler, or None.
        threshold_percentile : float
            Percentile cutoff for anomaly threshold.
        """
        self.df_raw = df.copy()
        self.time_col = time_col
        self.feature = feature
        self.window_size = window_size
        self.overlap = overlap
        self.model_params = model_params
        self.external_model = model
        self.scaler_type = scaler
        self.scaler = None
        self.model = None
        self.threshold_percentile = threshold_percentile

        self.df = None
        self.input_data = None
        self.input_data_scaled = None

        self.anomaly_scores = None
        self.threshold_scores = None
        self.window_end_idx = None  # alignment indices

    def _format_time_series(self):
        df = self.df_raw[[self.time_col, self.feature]].copy()
        df = df.rename(columns={self.time_col: "ds", self.feature: "y"})
        df["unique_id"] = "series_1"
        return df

    def _segment_time_series(self, series: pd.Series) -> np.ndarray:
        """
        Split series into overlapping windows.
        """
        n = len(series)
        w = self.window_size
        if n < w:
            self.window_end_idx = []
            return np.empty((0, w))

        if not (0 <= self.overlap < 1):
            raise ValueError("overlap must be between 0 and 1 (non-inclusive of 1).")

        step = max(1, int(np.round(w * (1 - self.overlap))))
        windows, end_idx = [], []

        for start in range(0, n - w + 1, step):
            end = start + w
            windows.append(series.iloc[start:end].values)
            end_idx.append(end - 1)

        self.window_end_idx = end_idx
        return np.asarray(windows)

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_type is None:
            return X
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        else:
            self.scaler = self.scaler_type
        return self.scaler.fit_transform(X)

    def prepare(self):
        self.df = self._format_time_series()
        self.input_data = self._segment_time_series(self.df["y"])
        if self.input_data.size == 0:
            raise ValueError(f"Not enough data for one window of size {self.window_size}.")
        self.input_data_scaled = self._apply_scaler(self.input_data)

    def _init_model(self):
        if self.external_model is not None:
            return self.external_model
        default_params = {
            "hidden_neurons": [self.window_size, 4, 4, self.window_size],
            "hidden_activation": "relu",
            "epochs": 10,
            "batch_norm": True,
            "learning_rate": 0.001,
            "batch_size": 32,
            "dropout_rate": 0.2,
        }
        if self.model_params:
            default_params.update(self.model_params)
        return AutoEncoder(**default_params)

    def fit(self, threshold_percentile=None):
        if self.input_data_scaled is None:
            raise ValueError("Call prepare() before fit().")
        if threshold_percentile is None:
            threshold_percentile = self.threshold_percentile

        self.model = self._init_model()
        self.model.fit(self.input_data_scaled)

        self.anomaly_scores = self.model.decision_scores_
        self.threshold_scores = np.percentile(self.anomaly_scores, threshold_percentile)

    def predict(self, input_series: pd.Series) -> np.ndarray:
        if self.model is None:
            raise ValueError("Call fit() before predict().")

        n = len(input_series)
        w = self.window_size
        step = max(1, int(round(w * (1 - self.overlap))))

        windows = []
        for start in range(0, n - w + 1, step):
            end = start + w
            windows.append(input_series.iloc[start:end].values)
        X = np.asarray(windows)

        if X.size == 0:
            return np.array([])

        if self.scaler:
            X = self.scaler.transform(X)

        return self.model.decision_function(X)

    def plot_score_distribution(self, title_id):
        if self.anomaly_scores is None:
            raise ValueError("Model not trained. Call fit() first.")
        plt.figure(figsize=(10, 4))
        plt.hist(self.anomaly_scores, bins=20, edgecolor='black')
        plt.title(f"Histogram of Anomaly Scores at {title_id}")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_series_with_anomalies(self, title_id):
        if self.anomaly_scores is None:
            raise ValueError("Model not trained. Call fit() first.")
        if not self.window_end_idx:
            raise ValueError("Window indices missing. Did you call prepare()?")

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot raw series
        ax1.plot(self.df['ds'], self.df['y'], label="Original Time Series", color="blue")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Original Value", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")

        # Plot anomaly scores at window end positions
        x_scores = self.df['ds'].iloc[self.window_end_idx].values
        ax2 = ax1.twinx()
        ax2.plot(x_scores, self.anomaly_scores, color="orange", label="Anomaly Score", linewidth=2)
        ax2.set_ylabel("Anomaly Score", color="orange")
        ax2.tick_params(axis='y', labelcolor="orange")

        plt.title(f"Time Series and Anomaly Scores at {title_id}")
        fig.tight_layout()
        plt.grid(True)
        plt.show()

    def get_anomaly_stats(self):
        if self.anomaly_scores is None:
            raise ValueError("Model not trained. Call fit() first.")
        if not self.window_end_idx:
            raise ValueError("Window indices missing. Did you call prepare()?")

        is_outlier = self.anomaly_scores > self.threshold_scores
        base_df = self.df_raw.iloc[self.window_end_idx].copy()
        base_df["anomaly_score"] = self.anomaly_scores
        base_df["is_outlier"] = is_outlier

        cols = ["sn", self.time_col, self.feature, "anomaly_score", "is_outlier"]
        cols = [c for c in cols if c in base_df.columns]
        return base_df[base_df["is_outlier"]][cols]

