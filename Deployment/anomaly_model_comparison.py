import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, date
from pyspark.sql.window import Window
from pyspark.sql.functions import sum, lag, col, split, concat_ws, lit ,udf,count, max,lit,avg, when,concat_ws,to_date,explode
from pyspark.sql.types import *
from pyspark.sql.types import FloatType
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys 

sys.path.append('/usr/apps/vmas/scripts/ZS') 

from MailSender import MailSender
# http://njbbvmaspd13:18080/next/#/notebook/2KW1NYKEF
sys.path.append('/usr/apps/vmas/scripts/ZS/owl_anomaly') 

import time


from OutlierDetector import (
    FeaturewiseKDENoveltyDetector,
    DBSCANOutlierDetector,
    ARIMAAnomalyDetector,
    EWMAAnomalyDetector
)

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
                                scale=False, 
                                filter_percentile=98)
        output = detector.fit()

    elif method == "EWMA":
        detector = EWMAAnomalyDetector(group_sorted, 
                                        feature=feature_col, 
                                        recent_window_size=num_recent_points, 
                                        window=72,
                                        no_of_stds=2.6, 
                                        n_shift=1, 
                                        anomaly_direction="low")
        output = detector.fit()

    else:
        raise ValueError(f"Unsupported method: {method}")

    return {
        "sn": sn_val,
        "outlier_count": output["outlier_count"],
        "total_new_points": output["total_new_points"]
    }

def detect_anomalies_parallel(df, method, feature_col, time_col, feature_cols, n_workers, num_recent_points):
    args = [(sn, group, method, time_col, feature_col, feature_cols, num_recent_points) for sn, group in df.groupby("sn")]
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(detect_anomaly_group, *arg) for arg in args]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def run_detection_experiment(df, methods, feature_col, feature_cols, time_col, worker_list, result_base_path, num_recent_points):
    summary = []
    for method in methods:
        for workers in worker_list:
            print(f"\nRunning {method} with {workers} workers...")
            start = time.time()

            results = detect_anomalies_parallel(df, method, feature_col, time_col, feature_cols, workers, num_recent_points)
            duration = time.time() - start

            df_result = pd.DataFrame(results)
            spark_df = spark.createDataFrame(df_result)
            output_path = f"{result_base_path}/{feature_col}/{method}/{method}_{workers}"
            spark_df.write.mode("overwrite").parquet(output_path)

            summary.append({"method": method, "workers": workers, "duration_sec": duration, "num_serial_numbers": len(results)})
            print(f"Completed in {duration:.2f}s, SNs: {len(results)}")

    return pd.DataFrame(summary)

if __name__ == "__main__":
    spark = SparkSession.builder.appName('AnomalyDetection').config("spark.ui.port", "24041").getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    TIME_COL = "hour"
    FEATURE_COLS = ["avg_4gsnr", "avg_5gsnr"]
    FEATURE_COL = "avg_4gsnr"
    NUM_RECENT_POINTS = 1
    METHODS = ["KDE", "EWMA","DBSCAN"]
    WORKER_LIST = [50]

    input_path = "/user/ZheS//owl_anomally/capacity_pplan50127_sliced/"
    result_path = "/user/ZheS//owl_anomally//zanomally_result_sliced/"

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
