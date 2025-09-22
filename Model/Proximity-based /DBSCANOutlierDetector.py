import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, FloatType, BooleanType
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, lag, col, split, concat_ws, lit ,udf,count, max,lit,avg, when,concat_ws,to_date,explode,last
from datetime import datetime, timedelta

class DBSCANOutlierDetector:
    """
    DBSCAN-based outlier detection for time series with train/test window flexibility.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the data.
    features : List[str]
        List of feature column names to be used in DBSCAN.
    eps : float
        Neighborhood radius parameter for DBSCAN.
    min_samples : int
        Minimum number of points to form a core point in DBSCAN.
    train_idx : int or str
        Number of initial rows to use for training, or "all".
    test_idx : int or str
        Number of most recent rows to use for testing, or "all".
    time_col : str
        Name of timestamp column.
    feature_col : str
        Name of the primary feature column for output.
    scale : bool
        Whether to standardize features.
    filter_percentile : float
        Percentile filter for training set. 100 means no filtering.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        eps: float = 0.5,
        min_samples: int = 5,
        train_idx: Union[int, str] = "all",
        recent_window_size: Union[int, str] = 24,
        time_col: str = "time",

        scale: bool = False,
        filter_percentile: float = 100,
    ):
        self.df = df.copy()
        self.features = features
        self.eps = eps
        self.min_samples = min_samples
        self.train_idx = train_idx
        self.test_idx = recent_window_size
        self.time_col = time_col

        self.scale = scale
        self.filter_percentile = filter_percentile

        self.train_X = None
        self.test_X = None
        self.test_indices = None
        self.labels_ = None
        self.outlier_mask = None
        self.scaler = None
        self.dbscan_model = None

    def _apply_percentile_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.filter_percentile >= 100:
            return df
        for col in self.features:
            lower = np.percentile(df[col], (100 - self.filter_percentile) / 2)
            upper = np.percentile(df[col], 100 - (100 - self.filter_percentile) / 2)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        return df

    def _apply_scaling(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.scale:
            return X_train, X_test
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X_train), self.scaler.transform(X_test)

    def _split_data(self):
        if self.train_idx == "all":
            df_train = self.df
        else:
            df_train = self.df.iloc[:self.train_idx]

        if self.test_idx == "all":
            df_test = self.df
        else:
            df_test = self.df.iloc[-self.test_idx:]

        df_train = self._apply_percentile_filter(df_train)
        self.test_indices = df_test.index

        X_train = df_train[self.features].values
        X_test = df_test[self.features].values
        self.train_X, self.test_X = self._apply_scaling(X_train, X_test)

    def fit(self) -> pd.DataFrame:
        self._split_data()

        self.dbscan_model = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(self.train_X)
        core_samples = self.train_X[self.dbscan_model.core_sample_indices_]

        self.labels_ = np.array([
                                    -1 if np.min(np.linalg.norm(core_samples - x, axis=1)) > self.eps else 0
                                    for x in self.test_X
                                ])
        self.outlier_mask = self.labels_ == -1

        self.df["is_outlier"] = False
        self.df.loc[self.test_indices[self.outlier_mask], "is_outlier"] = True

        return self.df[self.df["is_outlier"]][["sn", self.time_col] + self.features + ["is_outlier"]]
    
    def plot_scatter(self, use_scaled: bool = False):
        """
        Scatter plot of DBSCAN clustering, differentiating train/test outliers.
    
        Parameters
        ----------
        use_scaled : bool
            If True, plot using standardized features.
            If False, plot using original unscaled feature values.
        """
        if self.dbscan_model is None:
            raise ValueError("Call .fit() before plotting.")
    
        train_labels = self.dbscan_model.labels_
        train_outliers = np.where(train_labels == -1)[0]
    
        if use_scaled:
            X_train_plot = self.train_X
            X_test_plot = self.test_X
        else:
            if self.test_idx == "all":
                X_train_plot = np.empty((0, len(self.features)))  # No train set
                X_test_plot = self.df[self.features].values
            else:
                X_train_plot = self.df.iloc[:-self.test_idx][self.features].values
                X_test_plot = self.df.iloc[-self.test_idx:][self.features].values
    
        plt.figure(figsize=(8, 6))
    
        if X_train_plot.shape[0] > 0:
            plt.scatter(X_train_plot[:, 0], X_train_plot[:, 1], c='blue', label='Train Inliers')
            plt.scatter(X_train_plot[train_outliers, 0], X_train_plot[train_outliers, 1],
                        c='gray', label='Train Outliers', alpha=0.3)
    
        plt.scatter(X_test_plot[~self.outlier_mask, 0], X_test_plot[~self.outlier_mask, 1],
                    c='orange', label='Test Inliers')
        plt.scatter(X_test_plot[self.outlier_mask, 0], X_test_plot[self.outlier_mask, 1],
                    c='red', label='Test Outliers')
    
        plt.xlabel(self.features[0])
        if len(self.features) > 1:
            plt.ylabel(self.features[1])
        test_size_display = self.test_idx if isinstance(self.test_idx, int) else "All"
        plt.title(f"DBSCAN Outlier Detection (Test: {test_size_display})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    
    def plot_timeseries(self, time_col: str = None, feature: str = None):
        """
        Plot time series with DBSCAN-detected outliers from the test window.
        
        Parameters
        ----------
        time_col : str
            Name of the timestamp column for the X-axis.
        feature : str, optional
            Name of the feature to plot on the Y-axis. Defaults to self.features[0].
        """
        if self.labels_ is None:
            raise ValueError("Call .fit() before plotting.")
    
        if feature is None:
            feature = self.features[0]
        elif feature not in self.df.columns:
            raise ValueError(f"Feature '{feature}' not found in the dataframe.")
    
        if time_col is None:
            time_col = self.time_col
    
        ts_df = self.df.copy()
        ts_df["outlier"] = False
        ts_df.loc[self.test_indices[self.outlier_mask], "outlier"] = True
    
        plt.figure(figsize=(12, 5))
        plt.plot(ts_df[time_col], ts_df[feature], label=feature, alpha=0.7)
        plt.scatter(
            ts_df[time_col][ts_df["outlier"]],
            ts_df[feature][ts_df["outlier"]],
            color='red', label='Outliers'
        )
        plt.xlabel(time_col)
        plt.ylabel(feature)
        plt.title(f"Time Series with DBSCAN Outliers ({feature})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def convert_string_numerical(df, String_typeCols_List):
    from pyspark.sql.functions import col
    return df.select([col(c).cast('double') if c in String_typeCols_List else col(c) for c in df.columns])



if __name__ == "__main__":
    spark = SparkSession.builder.appName('Zhe_DBSCAN_Anomaly_Detection')\
                        .config("spark.ui.port", "24041")\
                        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    FEATURE_COLS = ["RSRQ","4GRSRQ"]
    TIME_COL = "time"

    # 1. Read and preprocess data from date list
    start_date = datetime.strptime("2025-07-07", "%Y-%m-%d")
    end_date = datetime.strptime("2025-07-13", "%Y-%m-%d")
    date_list = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range((end_date - start_date).days + 1)]
    heartbeat_base = "/user/ZheS//owl_anomally/df_adhoc_heartbeat/"
    paths = [heartbeat_base+date_str for date_str in date_list]

    df_raw = spark.read.parquet(*paths)
    df_converted = convert_string_numerical(df_raw, FEATURE_COLS)
    df_filtered = df_converted.select(["sn", TIME_COL] + FEATURE_COLS)\
                                .orderBy( "sn",TIME_COL )
    
    # 2. Define output schema
    schema = StructType([
                        StructField("sn", StringType(), True),
                        StructField(TIME_COL, TimestampType(), True),
                        StructField(FEATURE_COLS[0], FloatType(), True),
                        StructField(FEATURE_COLS[1], FloatType(), True),
                        StructField("is_outlier", BooleanType(), True)
                    ])

    # 3. UDF for applyInPandas
    def udf_detect_DBSCAN_outliers(group_df: pd.DataFrame) -> pd.DataFrame:
        if len(group_df) < 10:
            return pd.DataFrame([], columns=schema.fieldNames())
        try:
            group_df = group_df.sort_values(TIME_COL)  # ✅ Ensure time ordering
            detector = DBSCANOutlierDetector( df = group_df, 
                                    features= FEATURE_COLS, 
                                    eps=2, 
                                    min_samples=2, 
                                    train_idx = "all",
                                    recent_window_size="all", 
                                    time_col = "time",
                                    scale=False, 
                                    filter_percentile=100)
            return detector.fit()
        except Exception:
            return pd.DataFrame([], columns=schema.fieldNames())

    # 4. Run EWMA detection using applyInPandas
    df_anomaly_result = df_filtered.groupBy("sn").applyInPandas(udf_detect_DBSCAN_outliers, schema=schema)
    df_anomaly_result.show()
    # 5. Write results
    df_anomaly_result.write.mode("overwrite").parquet(f"/user/ZheS/owl_anomally/dailyrawreboot/outlier_{FEATURE_COLS[0]}_{FEATURE_COLS[1]}/DBSCAN")
