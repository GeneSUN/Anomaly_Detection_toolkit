from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, FloatType, BooleanType
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
from pyspark.sql.functions import sum, lag, col, split, concat_ws, lit ,udf,count, max,lit,avg, when,concat_ws,to_date,explode,last


class EWMAAnomalyDetector:
    """
    EWMA-based anomaly detector with optional scaling and flexible recent window evaluation.

    Parameters:
        df (pd.DataFrame): Input time series data.
        feature (str): Target feature to detect anomalies on.
        recent_window_size (int or str): 'all' or integer; number of recent points to evaluate in scoring.
        window (int): Span for EWMA and rolling std.
        no_of_stds (float): Control limit multiplier.
        n_shift (int): Shift to prevent leakage.
        anomaly_direction (str): One of {'both', 'high', 'low'}.
        scaler (str or object): Optional scaler: 'standard', 'minmax', or custom scaler with fit_transform and inverse_transform.
        min_std_ratio (float): Minimum rolling std as a ratio of |feature| to avoid near-zero std (default: 0.01).
    """

    def __init__(
        self,
        df,
        feature,
        timestamp_col="time",
        recent_window_size="all",
        window=100,
        no_of_stds=3.0,
        n_shift=1,
        anomaly_direction="low",
        scaler=None,
        min_std_ratio=0.01,
        use_weighted_std=False
    ):
        assert anomaly_direction in {"both", "high", "low"}
        assert scaler in {None, "standard", "minmax"} or hasattr(scaler, "fit_transform")
        assert isinstance(recent_window_size, (int, type(None), str))

        self.df_original = df.copy()
        self.feature = feature
        self.timestamp_col = timestamp_col
        self.window = window
        self.no_of_stds = no_of_stds
        self.n_shift = n_shift
        self.recent_window_size = recent_window_size
        self.anomaly_direction = anomaly_direction
        self.df_ = None
        self.scaler_type = scaler
        self._scaler = None
        self.min_std_ratio = min_std_ratio
        self.use_weighted_std = use_weighted_std

    def _apply_scaler(self, df):
        df = df.copy()
        if self.scaler_type is None:
            df['feature_scaled'] = df[self.feature]
        else:
            if self.scaler_type == "standard":
                self._scaler = StandardScaler()
            elif self.scaler_type == "minmax":
                self._scaler = MinMaxScaler()
            else:
                self._scaler = self.scaler_type
            df['feature_scaled'] = self._scaler.fit_transform(df[[self.feature]])
        return df

    def _inverse_scaler(self, series):
        if self._scaler is None:
            return series
        return self._scaler.inverse_transform(series.values.reshape(-1, 1)).flatten()

    def _weighted_std_ewm(self, series, span):
        """
        Calculate exponentially weighted standard deviation.

        Formula:
            σ_w = sqrt( Σ wᵢ (xᵢ - μ_w)² / Σ wᵢ )

        Where:
            - xᵢ: input values in the rolling window
            - wᵢ: exponential weights (more recent points have higher weight)
            - μ_w: weighted mean = Σ wᵢ xᵢ / Σ wᵢ

        Parameters:
            series (pd.Series): Input series to compute weighted std on
            span (int): EWMA span (same as for EMA)

        Returns:
            pd.Series: weighted std aligned with EMA
        """
        import numpy as np
        alpha = 2 / (span + 1)
        weights = np.array([(1 - alpha) ** i for i in reversed(range(span))])
        weights /= weights.sum()

        x = series.values
        stds = []
        for i in range(len(x)):
            if i < span:
                stds.append(np.nan)
            else:
                window = x[i - span + 1:i + 1]
                mu_w = np.sum(weights * window)
                var_w = np.sum(weights * (window - mu_w) ** 2)
                stds.append(np.sqrt(var_w))
        return pd.Series(stds, index=series.index)


    def _add_ewma(self):
        
        df = self._apply_scaler(self.df_original)

        target = df['feature_scaled'].shift(self.n_shift)
        
        df['EMA'] = target.ewm(span=self.window, adjust=False).mean()
        if self.use_weighted_std:
            df['rolling_std'] = self._weighted_std_ewm(target, span=self.window)
        else:
            df['rolling_std'] = target.rolling(window=self.window).std()
        
        # Impose a lower bound on std to avoid degenerate control limits
        min_std = self.min_std_ratio * df['feature_scaled'].abs()
        df['rolling_std'] = df['rolling_std'].where(df['rolling_std'] >= min_std, min_std)

        df['UCL'] = df['EMA'] + self.no_of_stds * df['rolling_std']
        df['LCL'] = df['EMA'] - self.no_of_stds * df['rolling_std']
        
        return df

    def _detect_anomalies(self, df):
        if self.anomaly_direction == "high":
            df['is_outlier'] = df['feature_scaled'] > df['UCL']
        elif self.anomaly_direction == "low":
            df['is_outlier'] = df['feature_scaled'] < df['LCL']
        else:
            df['is_outlier'] = (df['feature_scaled'] > df['UCL']) | (df['feature_scaled'] < df['LCL'])
        return df

    def fit(self):
        df = self._add_ewma()
        df = self._detect_anomalies(df)
        df_clean = df.dropna(subset=["EMA", "UCL", "LCL", "feature_scaled"])

        if self.recent_window_size in [None, "all"]:
            recent_df = df_clean
        else:
            recent_df = df_clean.tail(self.recent_window_size)

        self.df_ = df
        return recent_df[recent_df["is_outlier"]][["sn", self.timestamp_col, self.feature, "is_outlier"]]


    def plot(self, timestamp_col=None, figsize=(12, 6), title=None, start_time=None, end_time=None):
        if timestamp_col is None:
            timestamp_col = self.timestamp_col
        if self.df_ is None:
            raise ValueError("Run `.fit()` before plotting.")

        # Filter the DataFrame based on the specified time range
        df = self.df_
        if start_time and end_time:
            df = df[(df[timestamp_col] >= start_time) & (df[timestamp_col] <= end_time)]
        elif start_time:
            df = df[df[timestamp_col] >= start_time]
        elif end_time:
            df = df[df[timestamp_col] <= end_time]

        plt.figure(figsize=figsize)
        plt.plot(df[timestamp_col], df[self.feature], label='Original', color='blue', alpha=0.6)

        ema = self._inverse_scaler(df['EMA'])
        ucl = self._inverse_scaler(df['UCL'])
        lcl = self._inverse_scaler(df['LCL'])

        plt.plot(df[timestamp_col], ema, label='EWMA', color='orange')
        plt.plot(df[timestamp_col], ucl, label='UCL', color='green', linestyle='--')
        plt.plot(df[timestamp_col], lcl, label='LCL', color='red', linestyle='--')

        anomalies = df[df['is_outlier']]
        plt.scatter(anomalies[timestamp_col], anomalies[self.feature], color='red', label='Anomalies', zorder=5)

        plt.title(f"EWMA Anomaly Detection ({self.anomaly_direction}) {self.feature} {title}")
        plt.xlabel('Time')
        plt.ylabel(self.feature)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
def convert_string_numerical(df, String_typeCols_List):
    from pyspark.sql.functions import col
    return df.select([col(c).cast('double') if c in String_typeCols_List else col(c) for c in df.columns])

class HourlyIncrementProcessor:
    def __init__(self, df, columns):
        self.df = df
        self.columns = columns
        self.acc_columns = [f"{col}" for col in self.columns]
        self.incr_columns = [f"{col}_incr" for col in self.acc_columns]
        self.log_columns = [f"log_{col}" for col in self.incr_columns]

        self.df_hourly = None


    def compute_hourly_average(self):
        agg_exprs = [F.round(F.avg(c), 2).alias(f"{c}") for c in self.columns]
        self.df_hourly = self.df.groupBy("sn", "time").agg(*agg_exprs)

    def compute_increments(self, partition_col="sn", order_col="time"):
        window_spec = Window.partitionBy(partition_col).orderBy(order_col)
        for col_name in self.acc_columns:
            prev = lag(col(col_name), 1).over(window_spec)
            raw_incr = col(col_name) - prev
            prev_incr = lag(raw_incr, 1).over(window_spec)

            incr = when(col(col_name) < prev,
                        when(prev_incr.isNotNull(), prev_incr).otherwise(lit(0)))\
                   .otherwise(raw_incr)

            self.df_hourly = self.df_hourly.withColumn(f"{col_name}_incr", incr)

        for col_name in self.incr_columns:
            prev = lag(col(col_name), 1).over(window_spec)
            smoothed = when(col(col_name) < 0, prev).otherwise(col(col_name))
            self.df_hourly = self.df_hourly.withColumn(col_name, F.round(smoothed, 2))

    def apply_log_transform(self):
        for incr_col in self.incr_columns:
            self.df_hourly = self.df_hourly.withColumn(
                f"log_{incr_col}",
                F.round(F.log1p(F.when(F.col(incr_col) < 0, 0).otherwise(F.col(incr_col))), 2)
            )

    def fill_zero_with_previous(self, partition_col="sn", order_col="time"):
        window_spec = Window.partitionBy(partition_col).orderBy(order_col)\
            .rowsBetween(Window.unboundedPreceding, 0)

        for c in self.incr_columns:
            log_col = f"log_{c}"
            filled_col = f"{log_col}_nonzero_filled"
            self.df_hourly = self.df_hourly.withColumn(
                filled_col, when(col(log_col) != 0, col(log_col)).otherwise(None)
            )
            self.df_hourly = self.df_hourly.withColumn(
                log_col,
                last(filled_col, ignorenulls=True).over(window_spec)
            ).drop(filled_col)

    def run_all(self):
        self.compute_hourly_average()
        self.compute_increments()
        self.apply_log_transform()
        self.fill_zero_with_previous()

    def get_selected_pandas_df(self):
        return self.df_hourly.select(
            "time",
            "sn", 
            *self.acc_columns,
            *self.incr_columns,
            *self.log_columns
        ).na.drop().toPandas()


# ------------------------------
# Main Entry Point for PySpark
# ------------------------------
def convert_string_numerical(df, String_typeCols_List):
    from pyspark.sql.functions import col
    return df.select([col(c).cast('double') if c in String_typeCols_List else col(c) for c in df.columns])


if __name__ == "__main__":
    spark = SparkSession.builder.appName('Zhe_EWMA_Anomaly_Detection')\
                        .config("spark.ui.port", "24041")\
                        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    FEATURE_COL = "4GRSRP"
    TIME_COL = "time"

    # 1. Read and preprocess data from date list
    start_date = datetime.strptime("2025-07-07", "%Y-%m-%d")
    end_date = datetime.strptime("2025-07-13", "%Y-%m-%d")
    date_list = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range((end_date - start_date).days + 1)]
    heartbeat_base = "/user/ZheS//owl_anomally/df_adhoc_heartbeat/"
    paths = [heartbeat_base+date_str for date_str in date_list]

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

    # 3. UDF for applyInPandas
    def detect_ewma_outliers(group_df: pd.DataFrame) -> pd.DataFrame:
        if len(group_df) < 10:
            return pd.DataFrame([], columns=schema.fieldNames())
        try:
            group_df = group_df.sort_values(TIME_COL)  # ✅ Ensure time ordering
            detector = EWMAAnomalyDetector(df = group_df, 
                                            feature=FEATURE_COL, 
                                            timestamp_col = TIME_COL,
                                            recent_window_size="all",
                                            window=72,
                                            no_of_stds=3.0,
                                            n_shift=1,
                                            anomaly_direction="low",
                                            scaler="standard"
                                                            )
            return detector.fit()
        except Exception:
            return pd.DataFrame([], columns=schema.fieldNames())

    # 4. Run EWMA detection using applyInPandas
    df_anomaly_result = df_filtered.groupBy("sn").applyInPandas(detect_ewma_outliers, schema=schema)
    df_anomaly_result.show()
    # 5. Write results
    df_anomaly_result.write.mode("overwrite").parquet(f"/user/ZheS/owl_anomally/dailyrawreboot/outlier_{FEATURE_COL}/EWMA")
