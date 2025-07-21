# Anomaly_Display http://njbbvmaspd13:18080/#/notebook/2M1FCBHCY
# anomaly_model_comparison http://njbbvmaspd13:18080/#/notebook/2KZ8H5GJA

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, date

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys 
import time

sys.path.append('/usr/apps/vmas/scripts/ZS') 
from MailSender import MailSender

sys.path.append('/usr/apps/vmas/scripts/ZS/owl_anomaly') 

from FeaturewiseKDENoveltyDetector import FeaturewiseKDENoveltyDetector
from DBSCANOutlierDetector import DBSCANOutlierDetector
from ARIMAAnomalyDetector import ARIMAAnomalyDetector
from EWMAAnomalyDetector import EWMAAnomalyDetector


def prepare_data(df, min_rows=100):


    sn_counts = df.groupBy("sn").count()
    sn_valid = sn_counts.filter(F.col("count") >= min_rows).select("sn")
    df_filtered = df.join(sn_valid, on="sn", how="inner").toPandas()
    return df_filtered

def detect_anomaly_group(sn_val, group, method, time_col, feature_col, feature_cols, num_recent_points):
    group_sorted = group.sort_values(by=time_col).reset_index(drop=True)
    
    if method == "KDE":
        detector = FeaturewiseKDENoveltyDetector(
            df=group_sorted,
            feature_col=feature_col,
            time_col=time_col,
            train_idx=slice(0, 168),
            new_idx=slice(-num_recent_points, None),
            filter_percentile = 99,
            threshold_percentile=99.5,
            anomaly_direction="low"
        )
        output = detector.fit()

    elif method == "ARIMA":
        detector = ARIMAAnomalyDetector(
            df=group_sorted,
            time_col=time_col,
            feature=feature_col,
            season_length=1,
            confidence_level=99.5,
            anomaly_direction='lower'
        )
        detector.run()
        output = detector.get_recent_anomaly_stats(num_recent_points=num_recent_points)

    elif method == "DBSCAN":
        detector = DBSCANOutlierDetector(group_sorted, 
                                features=feature_cols, 
                                eps=2, 
                                min_samples=2, 
                                recent_window_size=num_recent_points, 
                                scale=True, 
                                filter_percentile=98)
        output = detector.fit()

    elif method == "EWMA":
        detector = EWMAAnomalyDetector(group_sorted, 
                                        feature=feature_col, 
                                        recent_window_size = "all",
                                        window=72,
                                        no_of_stds=2.6, 
                                        n_shift=1, 
                                        anomaly_direction="low")
        output = detector.fit()

    else:
        raise ValueError(f"Unsupported method: {method}")

    return output

def detect_anomalies_parallel(df, method, feature_col, time_col, feature_cols, n_workers, num_recent_points):
    args = [(sn, group, method, time_col, feature_col, feature_cols, num_recent_points)
            for sn, group in df.groupby("sn")]
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(detect_anomaly_group, *arg) for arg in args]
        for future in as_completed(futures):
            result_df = future.result()  # a DataFrame from each group
            if result_df is not None and not result_df.empty:
                results.append(result_df)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def run_detection_experiment(df, methods, feature_col, feature_cols, time_col, worker_list, result_base_path, num_recent_points):
    summary = []
    for method in methods:
        for workers in worker_list:
            print(f"\nRunning {method} with {workers} workers...")
            start = time.time()

            final_df = detect_anomalies_parallel(df, method, feature_col, time_col, feature_cols, workers, num_recent_points)
            duration = time.time() - start

            if not final_df.empty:
                spark_df = spark.createDataFrame(final_df)
                output_path = f"{result_base_path}/{feature_col}/{method}/{method}_{workers}"
                spark_df.write.mode("overwrite").parquet(output_path)

            summary.append({
                "method": method,
                "workers": workers,
                "duration_sec": duration,
                "num_outliers": len(final_df),
                "num_serial_numbers": df["sn"].nunique()
            })
            print(f"Completed in {duration:.2f}s, Outliers: {len(final_df)}")

    return pd.DataFrame(summary)

if __name__ == "__main__":
    spark = SparkSession.builder.appName('Zhe_AnomalyDetection').config("spark.ui.port", "24041").getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    TIME_COL = "hour"
    FEATURE_COLS = ["TotalBytesReceived", "TotalBytesSent"]
    FEATURE_COL = "TotalBytesReceived"
    NUM_RECENT_POINTS = 1
    METHODS = ["KDE", "EWMA","DBSCAN","ARIMA"]
    METHODS = ["EWMA",]
    WORKER_LIST = [25]
    result_path = "/user/ZheS//owl_anomally//zanomally_result_sliced/"

    #input_path = "/user/ZheS//owl_anomally/capacity_pplan50127_sliced/"
    input_path = "/user/ZheS//owl_anomally/throughput_pplan50127_sliced/"
    df_spark = spark.read.parquet(input_path).drop("sn").withColumnRenamed("slice_id", "sn")
    df_spark = df_spark.select(["sn", TIME_COL] + FEATURE_COLS)
    df_filtered = prepare_data(df_spark)

    summary_df = run_detection_experiment(
        df_filtered,
        methods=METHODS,
        feature_col=FEATURE_COL,
        feature_cols=FEATURE_COLS,
        time_col=TIME_COL,
        worker_list=WORKER_LIST,
        result_base_path=result_path,
        num_recent_points=NUM_RECENT_POINTS
    )

    print("\n=== Summary of All Runs ===")
    print(summary_df)
