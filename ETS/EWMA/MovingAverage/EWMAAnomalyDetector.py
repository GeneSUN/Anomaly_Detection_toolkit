from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
        scaler (str or object): Optional scaler: 'standard', 'minmax', or custom scaler with fit_transform and inverse_transform.
    """

    def __init__(
        self,
        df,
        feature,
        recent_window_size=600,
        window=10,
        no_of_stds=2.0,
        n_shift=1,
        anomaly_direction="low",
        scaler=None
    ):
        assert anomaly_direction in {"both", "high", "low"}, \
            "anomaly_direction must be one of {'both', 'high', 'low'}"
        assert scaler in {None, "standard", "minmax"} or hasattr(scaler, "fit_transform"), \
            "scaler must be 'standard', 'minmax', None, or a custom scaler with fit_transform"

        self.df_original = df.copy()
        self.feature = feature
        self.window = window
        self.no_of_stds = no_of_stds
        self.n_shift = n_shift
        self.recent_window_size = recent_window_size
        self.anomaly_direction = anomaly_direction
        self.df_ = None
        self.scaler_type = scaler
        self._scaler = None  # Actual scaler object

    def _apply_scaler(self, df):
        df = df.copy()
        self._scaler = None

        if self.scaler_type is None:
            df['feature_scaled'] = df[self.feature]
        else:
            if self.scaler_type == "standard":
                self._scaler = StandardScaler()
            elif self.scaler_type == "minmax":
                self._scaler = MinMaxScaler()
            else:
                self._scaler = self.scaler_type  # custom

            df['feature_scaled'] = self._scaler.fit_transform(df[[self.feature]])
        return df

    def _inverse_scaler(self, series):
        if self._scaler is None:
            return series
        return self._scaler.inverse_transform(series.values.reshape(-1, 1)).flatten()

    def _add_ewma(self):
        df = self._apply_scaler(self.df_original)
        target = df['feature_scaled'].shift(self.n_shift)
        df['EMA'] = target.ewm(span=self.window, adjust=False).mean()
        df['rolling_std'] = target.rolling(window=self.window).std()
        df['UCL'] = df['EMA'] + self.no_of_stds * df['rolling_std']
        df['LCL'] = df['EMA'] - self.no_of_stds * df['rolling_std']
        return df

    def _detect_anomalies(self, df):
        if self.anomaly_direction == "high":
            df['anomaly'] = df['feature_scaled'] > df['UCL']
        elif self.anomaly_direction == "low":
            df['anomaly'] = df['feature_scaled'] < df['LCL']
        else:  # both
            df['anomaly'] = (df['feature_scaled'] > df['UCL']) | (df['feature_scaled'] < df['LCL'])
        return df

    def fit(self):
        df = self._add_ewma()
        df = self._detect_anomalies(df)
        self.df_ = df
        recent_df = df.tail(self.recent_window_size)
        outliers = recent_df[recent_df['anomaly']]
        return {
            "outlier_count": len(outliers),
            "total_new_points": len(recent_df),
            "outlier_indices": outliers.index.tolist()
        }

    def plot(self, timestamp_col="timestamp", figsize=(12, 6)):
        if self.df_ is None:
            raise ValueError("Run `.fit()` before plotting.")
        df = self.df_

        plt.figure(figsize=figsize)
        plt.plot(df[timestamp_col], df[self.feature], label='Original', color='blue', alpha=0.6)

        # Inverse-transform if scaled
        ema = self._inverse_scaler(df['EMA'])
        ucl = self._inverse_scaler(df['UCL'])
        lcl = self._inverse_scaler(df['LCL'])

        plt.plot(df[timestamp_col], ema, label='EWMA', color='orange')
        plt.plot(df[timestamp_col], ucl, label='UCL', color='green', linestyle='--')
        plt.plot(df[timestamp_col], lcl, label='LCL', color='red', linestyle='--')

        anomalies = df[df['anomaly']]
        plt.scatter(anomalies[timestamp_col], anomalies[self.feature], color='red', label='Anomalies', zorder=5)

        plt.title(f"EWMA Anomaly Detection ({self.anomaly_direction})")
        plt.xlabel('Time')
        plt.ylabel(self.feature)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()