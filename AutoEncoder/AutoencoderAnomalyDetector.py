# http://njbbvmaspd13:18080/#/notebook/2KW77G1JD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    """
    Autoencoder-based anomaly detection for univariate time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series.
    time_col : str
        Name of the column containing timestamps.
    feature : str
        Name of the feature column to analyze.
    n_lags : int
        Number of lag observations to use as input.
    model_params : dict, optional
        Dictionary to override the default AutoEncoder settings.
        Template:
            model_params = {
                "hidden_neurons": [144, 4, 4, 144],
                "hidden_activation": "relu",     # activation function ('relu', 'sigmoid', etc.)
                "epochs": 20,                    # number of training epochs
                "batch_norm": True,              # whether to apply batch normalization
                "learning_rate": 0.001,          # learning rate for optimizer
                "batch_size": 32,                # batch size for training
                "dropout_rate": 0.2              # dropout rate between layers
            }
    """

    def __init__(self, df, time_col, feature, n_lags=24, model_params=None):
        self.df_raw = df.copy()
        self.time_col = time_col
        self.feature = feature
        self.n_lags = n_lags

        self.df = self._prepare_df()
        self.input_data = self._generate_lagged_input()
        self.scaler = StandardScaler()
        self.input_data_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.input_data)
        ).reset_index(drop=True)

        self.model = self._init_model(model_params)
        self.anomaly_scores = None

    def _prepare_df(self):
        df = self.df_raw[[self.time_col, self.feature]].copy()
        df = df.rename(columns={self.time_col: "ds", self.feature: "y"})
        df["unique_id"] = "series_1"
        return df

    def _generate_lagged_input(self):
        series = self.df["y"]
        input_data = [
            series.iloc[i - self.n_lags:i].values
            for i in range(self.n_lags, len(series))
        ]
        return np.array(input_data)

    def _init_model(self, custom_params=None):
        default_params = {
            "hidden_neurons": [self.n_lags, 4, 4, self.n_lags],
            "hidden_activation": "relu",
            "epochs": 20,
            "batch_norm": True,
            "learning_rate": 0.001,
            "batch_size": 32,
            "dropout_rate": 0.2,
        }
        if custom_params:
            default_params.update(custom_params)
        return AutoEncoder(**default_params)

    def fit(self):
        """Train the autoencoder model."""
        self.model.fit(self.input_data_scaled)
        self.anomaly_scores = self.model.decision_scores_

    def plot_score_distribution(self):
        """Plot histogram of anomaly scores."""
        if self.anomaly_scores is None:
            raise ValueError("Model not trained. Call fit() first.")
        plt.figure(figsize=(10, 4))
        plt.hist(self.anomaly_scores, bins=20, edgecolor='black')
        plt.title("Histogram of Anomaly Scores")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_series_with_anomalies(self):
        """Plot original time series with anomaly scores overlaid."""
        if self.anomaly_scores is None:
            raise ValueError("Model not trained. Call fit() first.")
        plt.figure(figsize=(16, 6))
        plt.plot(self.df['ds'], self.df['y'], label="Original Time Series", color="blue")
        plt.plot(
            self.df['ds'][self.n_lags:].values,
            self.anomaly_scores,
            color="orange",
            label="Anomaly Score (Reconstruction Error)",
            linewidth=2
        )
        plt.xlabel("Time")
        plt.ylabel("Value / Anomaly Score")
        plt.title("Time Series and Anomaly Scores")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def get_anomaly_stats(self, threshold=None, percentile=95):
        """
        Return a dictionary summarizing detected anomalies.
    
        Parameters
        ----------
        threshold : float, optional
            Manual anomaly score threshold. If None, use percentile cutoff.
        percentile : int
            Percentile to define threshold if not manually provided (default is 95).
    
        Returns
        -------
        dict
            {
                "total_new_points": int,
                "outlier_count": int,
                "anomaly_indices": list,
                "anomaly_timestamps": list,
                "anomaly_values": list
            }
        """
        if self.anomaly_scores is None:
            raise ValueError("Model not trained. Call fit() first.")
    
        if threshold is None:
            threshold = np.percentile(self.anomaly_scores, percentile)
    
        anomaly_flags = self.anomaly_scores > threshold
        indices = np.where(anomaly_flags)[0]
        
        anomaly_df = self.df.iloc[self.n_lags:].iloc[indices]
    
        return {
            "total_new_points": len(self.anomaly_scores),
            "outlier_count": len(indices),
            "anomaly_indices": indices.tolist(),
            "anomaly_timestamps": anomaly_df['ds'].tolist(),
            "anomaly_values": anomaly_df['y'].tolist()
        }


if __name__ == "__main__":

    spark = SparkSession.builder\
            .appName('AutoencoderAnomalyDetector')\
            .config("spark.ui.port","24041")\
            .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    import time
    start_time = time.time()  # ⏱ Start timing



    N_LAGS = 144
    hdfs_pd = "hdfs://njbbvmaspd11.nss.vzwnet.com:9000/"
    hdfs_pa =  'hdfs://njbbepapa1.nss.vzwnet.com:9000'
    import random

    snr_data_path = "/user/ZheS//owl_anomally/capacity_records/"
    feature_col = "avg_5gsnr"; time_col = "hour"; columns = ["sn", time_col, feature_col]

    df_snr_all = spark.read.parquet(snr_data_path).select(columns)
    sn_counts = df_snr_all.groupBy("sn").count()
    sn_valid = sn_counts.filter(F.col("count") >= 100).select("sn")
    df_snr = df_snr_all.join(sn_valid, on="sn", how="inner").toPandas()

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

        detector = AutoencoderAnomalyDetector(df=group_sorted, time_col=time_col, feature=feature_col)
        detector.fit()
        output = detector.get_anomaly_stats(num_recent_points = N_LAGS)
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

        # Save to HDFS using Spark
        df_result_spark = spark.createDataFrame(df_result)
        output_path = f"/user/ZheS//owl_anomally//anomally_result/arima_{workers}"
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
