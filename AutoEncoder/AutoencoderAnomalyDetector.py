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
    def __init__(self, 
                 df: pd.DataFrame, 
                 time_col: str, 
                 feature: str, 
                 window_type: str = "sliding",
                 n_lags: int = 24,
                 model_params: Optional[dict] = None,
                 model: Optional[object] = None,
                 scaler: Union[str, object, None] = "standard",
                 threshold_percentile = 99
                 ):
        """
        Initialization.

        Parameters
        ----------
        df : pd.DataFrame
        time_col : str
        feature : str
        n_lags : int
        model_params : dict, optional
        model : object, optional
            If provided, this custom model will be used instead of the default autoencoder.
        scaler : {'standard', 'minmax', object, None}
            'standard' for StandardScaler, 'minmax' for MinMaxScaler,
            a custom scaler instance (must implement fit_transform), or None.
        """
        self.df_raw = df.copy()
        self.time_col = time_col
        self.feature = feature
        self.window_type = window_type
        self.n_lags = n_lags
        self.model_params = model_params
        self.external_model = None
        self.scaler_type = scaler
        self.scaler = None
        self.model = None
        self.threshold_percentile = threshold_percentile
        
        self.df = None
        self.input_data = None
        self.input_data_scaled = None
        
        self.anomaly_scores = None
        self.threshold_scores = None
        
    def _format_time_series(self):
        df = self.df_raw[[self.time_col, self.feature]].copy()
        df = df.rename(columns={self.time_col: "ds", self.feature: "y"})
        df["unique_id"] = "series_1"
        return df

    def _segment_time_series(self, series: pd.Series) -> np.ndarray:
        """
        Generate lagged input sequences from a univariate time series.
    
        Parameters
        ----------
        series : pd.Series
            Input univariate time series.
        window_type : str
            Type of windowing. Options:
                - 'sliding': overlapping windows (default)
                - 'block': non-overlapping segments
    
        Returns
        -------
        np.ndarray
            2D array where each row is a lagged input sequence.
        """
        if self.window_type == "sliding":
            return np.array([
                series.iloc[i - self.n_lags:i].values
                for i in range(self.n_lags, len(series))
            ])
        
        elif self.window_type == "block":
            num_blocks = len(series) // self.n_lags
            return np.array([
                series.iloc[i * self.n_lags : (i + 1) * self.n_lags].values
                for i in range(num_blocks)
            ])
    
        else:
            raise ValueError("Invalid window_type. Choose 'sliding' or 'block'.")


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
        self.input_data_scaled = self._apply_scaler(self.input_data)

    def _init_model(self):
        if self.external_model is not None:
            return self.external_model

        default_params = {
            "hidden_neurons": [self.n_lags, 4, 4, self.n_lags],
            "hidden_activation": "relu",
            "epochs": 20,
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
            
        input_matrix = self._segment_time_series(input_series)
        
        if self.scaler:
            input_matrix = self.scaler.transform(input_matrix)
        
        return self.model.decision_function(input_matrix)

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

    def plot_series_with_anomalies(self,title_id):
        
        if self.anomaly_scores is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        plt.figure(figsize=(16, 6))
        plt.plot(self.df['ds'], self.df['y'], label="Original Time Series", color="blue")
        plt.plot(
            self.df['ds'][self.n_lags:].values,
            self.anomaly_scores,
            color="orange",
            label="Anomaly Score",
            linewidth=2
        )
        plt.xlabel("Time")
        plt.ylabel("Value / Anomaly Score")
        plt.title(f"Time Series and Anomaly Scores at {title_id}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_anomaly_stats(self):
        """
        Return anomaly records and scores.
        """
        
        if self.anomaly_scores is None:
            raise ValueError("Model not trained. Call fit() first.")
    
    
        is_outlier = self.anomaly_scores > self.threshold_scores
    
        # Create mask for valid rows depending on windowing type
        if self.window_type == "sliding":
            base_df = self.df_raw.iloc[self.n_lags:].copy()
        else:  # "block"
            total_windows = len(self.anomaly_scores)
            base_df = self.df_raw.iloc[:total_windows * self.n_lags].copy()
            base_df = base_df.groupby(np.arange(len(base_df)) // self.n_lags).last().reset_index(drop=True)
    
        base_df["anomaly_score"] = self.anomaly_scores
        base_df["is_outlier"] = is_outlier
    
        anomaly_df = base_df[base_df["is_outlier"]][["sn", self.time_col, self.feature, "is_outlier"]]
    
        return anomaly_df


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
