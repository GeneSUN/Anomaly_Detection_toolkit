import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from datetime import datetime, timedelta, date
from pyspark.sql.window import Window
from pyspark.sql.functions import sum, lag, col, split, concat_ws, lit ,udf,count, max,lit,avg, when,concat_ws,to_date,explode
from pyspark.sql.types import *

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from functools import reduce
from pyspark.sql import DataFrame


from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd
import matplotlib.pyplot as plt

class ARIMAAnomalyDetector:
    """
    Anomaly detection for univariate time series using Nixtla's AutoARIMA with prediction intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data.
    time_col : str
        Timestamp column.
    feature : str
        Feature to analyze.
    season_length : int
        Periodicity (e.g., 24 for daily pattern in hourly data).
    confidence_level : int
        Confidence interval percentage (default: 99).
    freq : str
        Time series frequency string (e.g., 'h').
    anomaly_direction : str
        Which anomalies to detect: 'both', 'upper', or 'lower'.
    """

    def __init__(self, df, time_col, feature, season_length=24, confidence_level=99,
                 freq='h', anomaly_direction='both'):
        self.df = df
        self.time_col = time_col
        self.feature = feature
        self.season_length = season_length
        self.confidence_level = confidence_level
        self.freq = freq
        self.anomaly_direction = anomaly_direction  # NEW
        self.model = StatsForecast(
            models=[AutoARIMA(season_length=season_length)],
            freq=freq,
            n_jobs=-1
        )
        self.df_arima = None
        self.forecast_df = None
        self.insample_forecast = None

    def prepare_data(self, df=None):
        if df is None:
            df = self.df
        df_arima = df[[self.time_col, self.feature]].copy()
        df_arima = df_arima.rename(columns={self.time_col: "ds", self.feature: "y"})
        df_arima["unique_id"] = "series_1"
        self.df_arima = df_arima

    def fit_forecast(self, df_arima=None, horizon=24):
        if df_arima is None:
            df_arima = self.df_arima
        self.forecast_df = self.model.forecast(
            df=df_arima, h=horizon, level=[self.confidence_level], fitted=True
        ).reset_index()
        self.insample_forecast = self.model.forecast_fitted_values().reset_index()

    def detect_anomalies(self, insample_forecast=None):
        if insample_forecast is None:
            insample_forecast = self.insample_forecast

        lo_col = f'AutoARIMA-lo-{self.confidence_level}'
        hi_col = f'AutoARIMA-hi-{self.confidence_level}'

        # NEW: Add anomaly type column and flag based on direction
        if self.anomaly_direction == 'lower':
            insample_forecast['anomaly'] = insample_forecast['y'] < insample_forecast[lo_col]
            insample_forecast['anomaly_type'] = insample_forecast['anomaly'].apply(lambda x: 'low' if x else None)
        elif self.anomaly_direction == 'upper':
            insample_forecast['anomaly'] = insample_forecast['y'] > insample_forecast[hi_col]
            insample_forecast['anomaly_type'] = insample_forecast['anomaly'].apply(lambda x: 'high' if x else None)
        else:  # both
            is_low = insample_forecast['y'] < insample_forecast[lo_col]
            is_high = insample_forecast['y'] > insample_forecast[hi_col]
            insample_forecast['anomaly'] = is_low | is_high
            insample_forecast['anomaly_type'] = is_low.map({True: 'low'}).combine_first(is_high.map({True: 'high'}))

        self.insample_forecast = insample_forecast

    def plot_anomalies(self, insample_forecast=None, date_filter=None, confidence_level=None):
        if insample_forecast is None:
            insample_forecast = self.insample_forecast
        if confidence_level is None:
            confidence_level = self.confidence_level

        if date_filter is not None:
            start_date = pd.to_datetime(date_filter[0])
            end_date = pd.to_datetime(date_filter[1])
            insample_forecast = insample_forecast[
                (insample_forecast["ds"] >= start_date) & (insample_forecast["ds"] <= end_date)
            ]

        lo_col = f'AutoARIMA-lo-{confidence_level}'
        hi_col = f'AutoARIMA-hi-{confidence_level}'

        plt.figure(figsize=(16, 5))
        plt.plot(insample_forecast['ds'], insample_forecast['y'], label='Actual')
        plt.plot(insample_forecast['ds'], insample_forecast['AutoARIMA'], label='Forecast')
        plt.fill_between(insample_forecast['ds'], insample_forecast[lo_col], insample_forecast[hi_col],
                         color='gray', alpha=0.2, label=f'{confidence_level}% Prediction Interval')

        # Color code anomalies
        if 'anomaly_type' in insample_forecast.columns:
            low_anomalies = insample_forecast[insample_forecast['anomaly_type'] == 'low']
            high_anomalies = insample_forecast[insample_forecast['anomaly_type'] == 'high']
            plt.scatter(low_anomalies['ds'], low_anomalies['y'], color='blue', label='Low Anomalies')
            plt.scatter(high_anomalies['ds'], high_anomalies['y'], color='red', label='High Anomalies')
        else:
            anomalies = insample_forecast[insample_forecast['anomaly']]
            plt.scatter(anomalies['ds'], anomalies['y'], color='red', label='Anomalies')

        plt.legend()
        plt.title(f"ARIMA-based Anomaly Detection ({self.anomaly_direction})")
        plt.xlabel("Time")
        plt.ylabel(self.feature)
        plt.show()
    def get_recent_anomaly_stats(self, num_recent_points=24):
        if self.insample_forecast is None or 'anomaly' not in self.insample_forecast.columns:
            raise ValueError("Anomaly detection has not been run yet. Please call run() first.")
        recent_data = self.insample_forecast[-num_recent_points:].copy()
        outliers = recent_data[recent_data['anomaly']]
        return {
            "outlier_count": outliers.shape[0],
            "total_new_points": num_recent_points,
            "outlier_indices": outliers.index.tolist()
        }

    def run(self):
        self.prepare_data()
        self.fit_forecast()
        self.detect_anomalies()

class FeaturewiseKDENoveltyDetector:
    def __init__(self, df, feature_col="avg_4gsnr", time_col="hour", bandwidth=0.5,
                 train_idx=None, new_idx=None, train_percentile=100,
                 anomaly_direction="low"):
        """
        Parameters:
            df (pd.DataFrame): Input data.
            feature_col (str): Column containing values to evaluate.
            time_col (str): Time column for plotting.
            bandwidth (float): Bandwidth for KDE.
            train_idx (slice): Slice for training data.
            new_idx (slice): Slice for new (test) data.
            train_percentile (float): Percentile for filtering out high-end outliers in training set.
            anomaly_direction (str): One of {"both", "high", "low"} to detect direction of anomaly.
        """
        self.df = df
        self.feature_col = feature_col
        self.time_col = time_col
        self.bandwidth = bandwidth
        self.train_idx = train_idx
        self.new_idx = new_idx
        self.train_percentile = train_percentile
        self.anomaly_direction = anomaly_direction
        self.kde = None
        self.threshold = None

    def _filter_train_df(self, train_df):
        if self.train_percentile < 100:
            upper = np.percentile(train_df[self.feature_col], self.train_percentile)
            train_df = train_df[train_df[self.feature_col] <= upper]
        return train_df

    def fit(self):
        # Slice training and new data
        train_df = self.df.iloc[self.train_idx] if self.train_idx is not None else self.df.iloc[:-1]
        train_df = self._filter_train_df(train_df)
        new_df = self.df.iloc[self.new_idx] if self.new_idx is not None else self.df.iloc[-1:]

        # Fit KDE on training data
        X_train = train_df[self.feature_col].values.reshape(-1, 1)
        X_new = new_df[self.feature_col].values.reshape(-1, 1)

        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.kde.fit(X_train)

        # Compute densities
        dens_train = np.exp(self.kde.score_samples(X_train))
        self.threshold = np.quantile(dens_train, 0.01)

        dens_new = np.exp(self.kde.score_samples(X_new))
        outlier_mask_kde = dens_new < self.threshold

        # Directional anomaly logic based on percentiles
        new_values = new_df[self.feature_col].values
        lower_threshold = np.percentile(train_df[self.feature_col], 100 - self.train_percentile)
        upper_threshold = np.percentile(train_df[self.feature_col], self.train_percentile)

        if self.anomaly_direction == "low":
            direction_mask = new_values < lower_threshold
        elif self.anomaly_direction == "high":
            direction_mask = new_values > upper_threshold
        else:  # both
            direction_mask = (new_values < lower_threshold) | (new_values > upper_threshold)

        final_outlier_mask = outlier_mask_kde & direction_mask

        self.dens_new = dens_new
        self.outlier_mask = final_outlier_mask

        return {
            "new_densities": dens_new,
            "threshold (1% quantile)": self.threshold,
            "outlier_count": final_outlier_mask.sum(),
            "total_new_points": len(dens_new),
            "outlier_indices": list(np.where(final_outlier_mask)[0]),
        }

    def plot_line(self, sn_num=None):
        plt.figure(figsize=(10, 4))
        plt.plot(self.df[self.time_col], self.df[self.feature_col], marker='o', label=self.feature_col)
        if self.new_idx is not None:
            if isinstance(self.new_idx, int):
                idxs = [self.new_idx]
            elif isinstance(self.new_idx, slice):
                idxs = list(range(*self.new_idx.indices(len(self.df))))
            else:
                idxs = self.new_idx
            for idx in idxs:
                plt.axvline(self.df.iloc[idx][self.time_col], color='red', linestyle='-', alpha=0.1)
        title = f"Line Plot: {self.feature_col} Over Time"
        if sn_num:
            title += f" ({sn_num})"
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel(self.feature_col)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_kde(self):
        train_df = self.df.iloc[self.train_idx] if self.train_idx is not None else self.df.iloc[:-1]
        train_df = self._filter_train_df(train_df)
        new_df = self.df.iloc[self.new_idx] if self.new_idx is not None else self.df.iloc[-1:]

        X_train = train_df[self.feature_col].values.reshape(-1, 1)
        X_new = new_df[self.feature_col].values.reshape(-1, 1)

        x_vals = np.linspace(X_train.min() - 1, X_train.max() + 1, 1000).reshape(-1, 1)
        log_dens = self.kde.score_samples(x_vals)
        dens = np.exp(log_dens)

        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, dens, label="KDE Density Curve")
        plt.axhline(self.threshold, color='red', linestyle='-.', label=f"Threshold (1%)")

        label_added = False
        outlier_label_added = False
        for i, val in enumerate(X_new.flatten()):
            is_outlier = self.outlier_mask[i]
            color = 'green' if not is_outlier else 'orange'
            if is_outlier and not outlier_label_added:
                plt.axvline(val, color=color, linestyle='--', alpha=0.7, label="New Data (outlier)")
                outlier_label_added = True
            elif not is_outlier and not label_added:
                plt.axvline(val, color=color, linestyle='--', alpha=0.7, label="New Data")
                label_added = True
            else:
                plt.axvline(val, color=color, linestyle='--', alpha=0.7)
        plt.title(f"KDE Novelty Detection: {self.feature_col}")
        plt.xlabel(self.feature_col)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    spark = SparkSession.builder\
            .appName('KDENoveltyDetector')\
            .config("spark.ui.port","24041")\
            .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    hdfs_pd = "hdfs://njbbvmaspd11.nss.vzwnet.com:9000/"
    hdfs_pa =  'hdfs://njbbepapa1.nss.vzwnet.com:9000'


    snr_data_path = "/user/ZheS//owl_anomally/capacity_records/"
    feature_col = "avg_5gsnr"
    time_col = "hour"
    columns = ["sn", time_col, feature_col]

    df_snr_all = spark.read.parquet(snr_data_path).select(columns)
    df_snr = df_snr_all.toPandas()

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import pandas as pd

    def detect_anomaly_group(args):
        sn_val, group, feature_col, time_col = args
        group_sorted = group.sort_values(by=time_col).reset_index(drop=True)
        """
        detector = FeaturewiseKDENoveltyDetector(
            df=group_sorted,
            feature_col=feature_col,
            time_col=time_col,
            train_idx=slice(0, 1068),
            new_idx=slice(-26, None),
            train_percentile=99
        )
        output = detector.fit()
        """

        detector = ARIMAAnomalyDetector(df=group_sorted, time_col='hour', feature='avg_5gsnr', season_length=1)
        detector.run()
        output = detector.get_recent_anomaly_stats(num_recent_points = 26)
        return {
            "sn": sn_val,
            "outlier_count": output["outlier_count"],
            "total_new_points": output["total_new_points"]
        }

    def detect_anomalies_parallel(df, feature_col="avg_5gsnr", time_col="hour", n_workers=200):
        args = [(sn_val, group, feature_col, time_col) for sn_val, group in df.groupby("sn")]
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(detect_anomaly_group, arg) for arg in args]
            for future in as_completed(futures):
                results.append(future.result())
        return results

    workers = 30
    result = detect_anomalies_parallel(df_snr, n_workers=workers)
    df_result = pd.DataFrame(result)

    df_result_spark = spark.createDataFrame(df_result)
    output_path = f"/user/ZheS//owl_anomally//anomally_result/kde_{workers}"
    df_result_spark.write.mode("overwrite").parquet(output_path)
