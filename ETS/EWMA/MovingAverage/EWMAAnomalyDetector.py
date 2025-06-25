import pandas as pd
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class EWMAAnomalyDetector:
    """
    EWMAAnomalyDetector applies Exponentially Weighted Moving Average (EWMA) smoothing,
    calculates control limits (UCL/LCL), and detects anomalies in time series data.

    Attributes:
        window (int): The span for EWMA and rolling standard deviation.
        no_of_stds (float): Number of standard deviations for control limits.
        n_shift (int): Shift applied before calculating EWMA to prevent leakage.
        df_ema (pd.DataFrame): DataFrame after applying EWMA and limits.

    Notes:
        - If the input DataFrame contains multiple independent time series (e.g., multiple sensors or devices),
          the `ts_id` parameter should be specified to compute EWMA and statistics separately for each group.
        - If `ts_id` is None, the algorithm assumes a single time series.
    
    Usage
    -----
    np.random.seed(0)
    n = 100
    df_example = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="D"),
        "Close": np.random.normal(loc=10, scale=2, size=n)
    })

    # Apply the detector
    detector = EWMAAnomalyDetector(df_example,"Close" ,window=48, no_of_stds=2.0)
    detector.run()
    detector.plot()
    res = detector.score_recent()
    """

    def __init__(self, df, column, n_recent = 24 ,window: int = 10, no_of_stds: float = 2.0, n_shift: int = 1):
        self.df = df.copy()
        self.column = column
        self.window = window
        self.no_of_stds = no_of_stds
        self.n_shift = n_shift


    def add_ewma(self, df: pd.DataFrame = None, column: str = None, ts_id: str = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        if column is None:
            column = self.column

        target = df[column].shift(self.n_shift)

        if ts_id:
            df['EMA'] = df.groupby(ts_id)[column].shift(self.n_shift).ewm(span=self.window, adjust=False).mean()
            df['rolling_std'] = (
                df.groupby(ts_id)[column].shift(self.n_shift)
                .rolling(window=self.window).std()
                .reset_index(level=0, drop=True)
            )
        else:
            df['EMA'] = target.ewm(span=self.window, adjust=False).mean()
            df['rolling_std'] = target.rolling(window=self.window).std()

        df['UCL'] = df['EMA'] + self.no_of_stds * df['rolling_std']
        df['LCL'] = df['EMA'] - self.no_of_stds * df['rolling_std']
        return df

    def detect_anomalies(self, df: pd.DataFrame = None, column: str = None) -> pd.DataFrame:
        if df is None:
            df = self.df_ema
        if column is None:
            column = self.column
        df['anomaly'] = (df[column] > df['UCL']) | (df[column] < df['LCL'])
        return df


    def score_recent(self,df_anomaly = None, n_recent: int = 24) -> dict:
        """
        Evaluate only the most recent n_recent points for anomalies.

        Args:
            n_recent (int): Number of most recent data points to evaluate.

        Returns:
            dict: Summary of anomaly detection on recent data.
        """
        if df_anomaly is None:
            df_anomaly = self.df_anomaly
        
        recent_df = df_anomaly.tail(n_recent)
        outliers = recent_df[recent_df['anomaly']]

        return {
            "outlier_count": len(outliers),
            "total_new_points": len(recent_df),
            "outlier_indices": outliers.index.tolist()
        }

    def plot(self, timestamp_col: str = "timestamp", figsize=(12, 6)):
        """
        Plot the time series, EWMA, control limits, and mark anomalies.

        Args:
            timestamp_col (str): Name of the timestamp column for x-axis.
            figsize (tuple): Figure size for the plot.
        """
        df = self.df_anomaly.copy()

        plt.figure(figsize=figsize)
        plt.plot(df[timestamp_col], df[self.column], label='Original', color='blue', alpha=0.6)
        plt.plot(df[timestamp_col], df['EMA'], label='EWMA', color='orange')
        plt.plot(df[timestamp_col], df['UCL'], label='UCL', color='green', linestyle='--')
        plt.plot(df[timestamp_col], df['LCL'], label='LCL', color='red', linestyle='--')

        # Plot anomalies
        anomalies = df[df['anomaly']]
        plt.scatter(anomalies[timestamp_col], anomalies[self.column], color='red', label='Anomalies', zorder=5)

        plt.title('EWMA Anomaly Detection')
        plt.xlabel('Time')
        plt.ylabel(self.column)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def run(self):
        self.df_ema = self.add_ewma()
        self.df_anomaly = self.detect_anomalies()

        


