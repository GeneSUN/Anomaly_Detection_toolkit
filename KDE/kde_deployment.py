
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from datetime import datetime, timedelta, date
from pyspark.sql.window import Window
from pyspark.sql.functions import sum, lag, col, split, concat_ws, lit ,udf,count, max,lit,avg, when,concat_ws,to_date,explode
from pyspark.sql.types import *
from pyspark.sql.types import FloatType
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys 
import traceback
from pyspark.sql.functions import from_unixtime 
import argparse 

from functools import reduce
from pyspark.sql import DataFrame

sys.path.append('/usr/apps/vmas/scripts/ZS') 
from MailSender import MailSender


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
    email_sender = MailSender()
    spark = SparkSession.builder\
            .appName('KDENoveltyDetector')\
            .config("spark.ui.port","24041")\
            .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    import time
    start_time = time.time()  # ⏱ Start timing




    hdfs_pd = "hdfs://njbbvmaspd11.nss.vzwnet.com:9000/"
    hdfs_pa =  'hdfs://njbbepapa1.nss.vzwnet.com:9000'
    import random

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
        detector = FeaturewiseKDENoveltyDetector(
            df=group_sorted,
            feature_col=feature_col,
            time_col=time_col,
            train_idx=slice(0, 1068),
            new_idx=slice(-26, None),
            train_percentile=99
        )
        output = detector.fit()
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

    # Main experiment loop
    worker_list = [200, 150, 100, 64, 32, 16, 8, 4, 1]
    timing_results = []

    for workers in worker_list:
        print(f"\nRunning with {workers} workers...")
        start = time.time()
        result = detect_anomalies_parallel(df_snr, n_workers=workers)
        duration = time.time() - start
        df_result = pd.DataFrame(result)

        # Print the DataFrame for this worker setting (optional)
        print(df_result)

        # Save to HDFS using Spark
        df_result_spark = spark.createDataFrame(df_result)
        output_path = f"/user/ZheS//owl_anomally//anomally_result/kde_{workers}"
        df_result_spark.write.mode("overwrite").parquet(output_path)

        timing_results.append({
            "workers": workers,
            "duration_sec": duration,
            "num_serial_numbers": len(result)
        })
        print(f"Completed in {duration:.2f} seconds with {len(result)} serial numbers.")

    # Print timing summary table
    print("\n=== Summary of All Runs ===")
    timing_df = pd.DataFrame(timing_results)
    print(timing_df)

