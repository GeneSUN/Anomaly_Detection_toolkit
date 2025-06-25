import pandas as pd
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class EWMAAnomalyDetector:
    """
    Applies EWMA smoothing, calculates control limits, detects anomalies, and summarizes recent results.

    Parameters:
        df (pd.DataFrame): Input time series data.
        feature (str): Target feature to detect anomalies on.
        recent_window_size (int): Number of recent points to evaluate in scoring.
        window (int): Span for EWMA and rolling std.
        no_of_stds (float): Control limit multiplier.
        n_shift (int): Shift to prevent leakage.
        anomaly_direction (str): One of {'both', 'high', 'low'}.

    Usage:
    -------
        np.random.seed(0)
        n = 100
        df_example = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="D"),
            "Close": np.random.normal(loc=10, scale=2, size=n)
        })
        detector = EWMAAnomalyDetector(df_example, feature="Close", window=48, anomaly_direction="high")
        result = detector.fit()
        detector.plot()
    """

    def __init__(self, df, feature, recent_window_size=24, window=10, no_of_stds=2.0, n_shift=1, anomaly_direction="low"):
        assert anomaly_direction in {"both", "high", "low"}, \
            "anomaly_direction must be one of {'both', 'high', 'low'}"
        self.df_original = df.copy()
        self.feature = feature
        self.window = window
        self.no_of_stds = no_of_stds
        self.n_shift = n_shift
        self.recent_window_size = recent_window_size
        self.anomaly_direction = anomaly_direction
        self.df_ = None

    def _add_ewma(self):
        df = self.df_original.copy()
        target = df[self.feature].shift(self.n_shift)
        df['EMA'] = target.ewm(span=self.window, adjust=False).mean()
        df['rolling_std'] = target.rolling(window=self.window).std()
        df['UCL'] = df['EMA'] + self.no_of_stds * df['rolling_std']
        df['LCL'] = df['EMA'] - self.no_of_stds * df['rolling_std']
        return df

    def _detect_anomalies(self, df):
        if self.anomaly_direction == "high":
            df['anomaly'] = df[self.feature] > df['UCL']
        elif self.anomaly_direction == "low":
            df['anomaly'] = df[self.feature] < df['LCL']
        else:  # both
            df['anomaly'] = (df[self.feature] > df['UCL']) | (df[self.feature] < df['LCL'])
        return df

    def fit(self):
        df = self._add_ewma()
        df = self._detect_anomalies(df)
        self.df_ = df
        recent_df = self.df_.tail(self.recent_window_size)
        outliers = recent_df[recent_df['anomaly']]
        return {
            "outlier_count": len(outliers),
            "total_new_points": len(recent_df),
            "outlier_indices": outliers.index.tolist()
        }


    def plot(self, timestamp_col="timestamp", figsize=(12, 6)):
        if self.df_ is None:
            raise ValueError("Run `.run()` before plotting.")
        df = self.df_

        plt.figure(figsize=figsize)
        plt.plot(df[timestamp_col], df[self.feature], label='Original', color='blue', alpha=0.6)
        plt.plot(df[timestamp_col], df['EMA'], label='EWMA', color='orange')
        plt.plot(df[timestamp_col], df['UCL'], label='UCL', color='green', linestyle='--')
        plt.plot(df[timestamp_col], df['LCL'], label='LCL', color='red', linestyle='--')

        anomalies = df[df['anomaly']]
        plt.scatter(anomalies[timestamp_col], anomalies[self.feature], color='red', label='Anomalies', zorder=5)

        plt.title(f"EWMA Anomaly Detection ({self.anomaly_direction})")
        plt.xlabel('Time')
        plt.ylabel(self.feature)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

