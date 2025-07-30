from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import sum, lag, col, split, concat_ws, lit ,udf,count, max,lit,avg, when,concat_ws,to_date,explode,last
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, FloatType, BooleanType

from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

class FeaturewiseKDENoveltyDetector:
    def __init__(self, 
                 df, 
                 feature_col="avg_4gsnr", 
                 time_col="hour", 
                 bandwidth=0.5,
                 train_idx="all", 
                 new_idx="all", 
                 filter_percentile=100, 
                 threshold_percentile=99,
                 anomaly_direction="low"):
        """
        Parameters:
            df (pd.DataFrame): Input data.
            feature_col (str): Column containing values to evaluate.
            time_col (str): Time column for plotting.
            bandwidth (float): Bandwidth for KDE.
            train_idx (slice, list, int, or "all"): Indices for training data. "all" uses the entire DataFrame.
            new_idx (slice, list, int, or "all"): Indices for test data. "all" uses the entire DataFrame.
            filter_percentile (float): Percentile for filtering out high-end outliers in training set.
            threshold_percentile (float): Percentile to apply directional outlier threshold.
            anomaly_direction (str): One of {"both", "high", "low"} to control direction of anomaly detection.
        
        Example Usage:
        detector = FeaturewiseKDENoveltyDetector(
                                                df=your_df,
                                                feature_col="avg_5gsnr",
                                                time_col="hour",
                                                train_idx=slice(0, 1068),
                                                new_idx=slice(-26, None),
                                                filter_percentile = 100,
                                                threshold_percentile=95,
                                                anomaly_direction="both"  # can be "low", "high", or "both"
                                                )
        result = detector.fit()
        """
        self.df = df
        self.feature_col = feature_col
        self.time_col = time_col
        self.bandwidth = bandwidth
        self.train_idx = train_idx
        self.new_idx = new_idx
        self.filter_percentile = filter_percentile
        self.threshold_percentile = threshold_percentile
        self.anomaly_direction = anomaly_direction
        self.kde = None
        self.threshold = None
        self.outlier_mask = None

    def _filter_train_df(self, train_df):
        if self.filter_percentile < 100:
            upper = np.percentile(train_df[self.feature_col], self.filter_percentile)
            train_df = train_df[train_df[self.feature_col] <= upper]
        return train_df

    def fit(self):
        # Handle "all" option for training and testing index
        if self.train_idx == "all":
            train_df = self.df.copy()
        else:
            train_df = self.df.iloc[self.train_idx]
        train_df = self._filter_train_df(train_df)

        if self.new_idx == "all":
            new_df = self.df.copy()
            new_indices = self.df.index
        else:
            new_df = self.df.iloc[self.new_idx]
            new_indices = self.df.iloc[self.new_idx].index

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
        lower_threshold = np.percentile(train_df[self.feature_col], 100 - self.threshold_percentile)
        upper_threshold = np.percentile(train_df[self.feature_col], self.threshold_percentile)

        if self.anomaly_direction == "low":
            direction_mask = new_values < lower_threshold
        elif self.anomaly_direction == "high":
            direction_mask = new_values > upper_threshold
        else:  # both
            direction_mask = (new_values < lower_threshold) | (new_values > upper_threshold)

        # Final anomaly mask
        final_outlier_mask = outlier_mask_kde & direction_mask
        self.outlier_mask = final_outlier_mask

        is_outlier_col = pd.Series(False, index=self.df.index)
        is_outlier_col.loc[new_indices] = final_outlier_mask
        self.df["is_outlier"] = is_outlier_col

        return self.df[self.df["is_outlier"]][["sn", self.time_col, self.feature_col, "is_outlier"]]

    def plot_line(self, sn_num=None):
        plt.figure(figsize=(10, 4))
    
        # Plot the full time series
        plt.plot(self.df[self.time_col], self.df[self.feature_col], marker='o',
                 label=self.feature_col, color='blue', alpha=0.6)
    
        # Overlay outlier points
        if "is_outlier" in self.df.columns:
            df_outliers = self.df[self.df["is_outlier"]]
            if not df_outliers.empty:
                plt.scatter(df_outliers[self.time_col], df_outliers[self.feature_col],
                            color='red', label='Outlier', zorder=5, s=60, marker='X')
    
        # Optionally highlight vertical bars for test indices
        if self.new_idx != "all":
            if isinstance(self.new_idx, int):
                idxs = [self.new_idx]
            elif isinstance(self.new_idx, slice):
                idxs = list(range(*self.new_idx.indices(len(self.df))))
            else:
                idxs = self.new_idx
            for idx in idxs:
                plt.axvline(self.df.iloc[idx][self.time_col], color='gray', linestyle='--', alpha=0.2)
    
        # Plot formatting
        title = f"Line Plot: {self.feature_col} Over Time"
        if sn_num:
            title += f" ({sn_num})"
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel(self.feature_col)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def plot_kde(self):
        if self.train_idx == "all":
            train_df = self.df.copy()
        else:
            train_df = self.df.iloc[self.train_idx]
        train_df = self._filter_train_df(train_df)

        if self.new_idx == "all":
            new_df = self.df.copy()
        else:
            new_df = self.df.iloc[self.new_idx]

        X_train = train_df[self.feature_col].values.reshape(-1, 1)
        X_new = new_df[self.feature_col].values.reshape(-1, 1)

        x_vals = np.linspace(X_train.min() - 1, X_train.max() + 1, 1000).reshape(-1, 1)
        log_dens = self.kde.score_samples(x_vals)
        dens = np.exp(log_dens)

        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, dens, label="KDE Density Curve")
        plt.axhline(self.threshold, color='red', linestyle='-.', label=f"Threshold ({100-self.threshold_percentile}%)")

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


def convert_string_numerical(df, String_typeCols_List): 
    df = df.select([F.col(column).cast('double') if column in String_typeCols_List else F.col(column) for column in df.columns]) 
    return df

if __name__ == "__main__":
    spark = SparkSession.builder.appName('Zhe_FeaturewiseKDENoveltyDetector')\
                        .config("spark.ui.port", "24041")\
                        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    FEATURE_COL = "4GRSRP"
    TIME_COL = "time"
    
    # 1. Read and preprocess data from date list
    start_date = datetime.strptime("2025-07-07", "%Y-%m-%d")
    end_date = datetime.strptime("2025-07-13", "%Y-%m-%d")
    date_list = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date - start_date).days + 1)]
    heartbeat_base = "/user/ZheS//owl_anomally/df_adhoc_heartbeat/"
    paths = [heartbeat_base + date_str for date_str in date_list]

    df_raw = spark.read.parquet(*paths)
    df_converted = convert_string_numerical(df_raw, [FEATURE_COL])
    df_filtered = df_converted.select("sn", TIME_COL, FEATURE_COL)

    # 2. Define output schema
    schema = StructType([
                        StructField("sn", StringType(), True),
                        StructField(TIME_COL, TimestampType(), True),
                        StructField(FEATURE_COL, FloatType(), True),
                        StructField("is_outlier", BooleanType(), True)
                    ])

    # 3. UDF
    def detect_kde_outliers(group_df: pd.DataFrame) -> pd.DataFrame:
        if len(group_df) < 10:
            return pd.DataFrame([], columns=schema.fieldNames())
        try:
            group_df = group_df.sort_values(TIME_COL)  # ✅ Ensure time ordering
            detector = FeaturewiseKDENoveltyDetector(
                df=group_df,
                feature_col=FEATURE_COL,
                time_col=TIME_COL,
                train_idx="all",
                new_idx="all",
                filter_percentile=100,
                threshold_percentile=95,
                anomaly_direction="low"
            )
            return detector.fit()
        except Exception:
            return pd.DataFrame([], columns=schema.fieldNames())

    # 4. Run anomaly detection in parallel
    df_anomaly_result = df_filtered.groupBy("sn").applyInPandas(detect_kde_outliers, schema=schema)

    # 5. Write to HDFS
    df_anomaly_result.write.mode("overwrite").parquet(f"/user/ZheS/owl_anomally/dailyrawreboot/outlier_{FEATURE_COL}/kde")



