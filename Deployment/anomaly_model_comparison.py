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
import traceback
from pyspark.sql.functions import from_unixtime 
import argparse 

from functools import reduce
from pyspark.sql import DataFrame

sys.path.append('/usr/apps/vmas/scripts/ZS') 

from MailSender import MailSender
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd
import matplotlib.pyplot as plt
# http://njbbvmaspd13:18080/next/#/notebook/2KW1NYKEF
sys.path.append('/usr/apps/vmas/scripts/ZS/owl_anomaly') 
from OutlierDetector import FeaturewiseKDENoveltyDetector, DBSCANOutlierDetector, ARIMAAnomalyDetector,EWMAAnomalyDetector

def prepare_data(df, min_rows=100):


    sn_counts = df.groupBy("sn").count()
    sn_valid = sn_counts.filter(F.col("count") >= min_rows).select("sn")
    df_filtered = df.join(sn_valid, on="sn", how="inner").toPandas()
    return df_filtered

def detect_anomaly_group(args, method, feature_col = None, feature_cols = None, num_recent_points= 1):
    sn_val, group, time_col = args
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
        """            """
        detector = DBSCANOutlierDetector(group_sorted, 
                                        features=FEATURE_COLS, 
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

def detect_anomalies_parallel(df, method, feature_col="avg_4gsnr", time_col=None, n_workers=None):
    assert time_col is not None, "time_col must be provided"
    args = [(sn_val, group, time_col) for sn_val, group in df.groupby("sn")]
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(detect_anomaly_group, arg, method, feature_col) for arg in args]
        for future in as_completed(futures):
            results.append(future.result())
    return results

def run_experiments(method, df, worker_list, feature_col, time_col):
    timing_results = []

    for workers in worker_list:
        print(f"\nRunning {method} with {workers} workers...")
        start = time.time()
        result = detect_anomalies_parallel(df, method=method, feature_col=feature_col, time_col=time_col, n_workers=workers)        
        duration = time.time() - start

        df_result = pd.DataFrame(result)
        df_result_spark = spark.createDataFrame(df_result)
        output_path = f"{result_base_path}/{feature_col}/{method}/{method}_{workers}"
        df_result_spark.write.mode("overwrite").parquet(output_path)

        timing_results.append({
            "method": method,
            "workers": workers,
            "duration_sec": duration,
            "num_serial_numbers": len(result)
        })
        print(f"Completed in {duration:.2f} seconds with {len(result)} serial numbers.")
    
    return timing_results



if __name__ == "__main__":
    email_sender = MailSender()
    spark = SparkSession.builder\
            .appName('KDENoveltyDetector')\
            .config("spark.ui.port","24041")\
            .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    import time
    start_time = time.time()  # ⏱ Start timing
    TIME_COL = "hour"
    # Constants
    input_path = "/user/ZheS//owl_anomally/capacity_pplan50127_sliced/"
    result_base_path = "/user/ZheS//owl_anomally//zanomally_result_sliced/"
    FEATURE_COLS = ["avg_4gsnr","avg_5gsnr"]
    num_recent_points = 1

    columns = ["sn", TIME_COL] + FEATURE_COLS
    df = spark.read.parquet(input_path)\
                .drop("sn")\
                .withColumnRenamed("slice_id", "sn")\
                .select(columns)


    hdfs_pd = "hdfs://njbbvmaspd11.nss.vzwnet.com:9000/"
    hdfs_pa =  'hdfs://njbbepapa1.nss.vzwnet.com:9000'

    df = prepare_data(df)
    worker_list = [50]
    #worker_list = [100]
    all_results = []

    #for method in ["EWMA","ARIMA", "DBSCAN", "KDE"]:
    for method in ["KDE","EWMA"]:
        feature_col = "avg_4gsnr"  # you can switch this depending on the use case

        results = run_experiments(method, df, worker_list, feature_col, time_col=TIME_COL)
        all_results.extend(results)

    # Final timing summary
    summary_df = pd.DataFrame(all_results)
    print("\n=== Summary of All Runs ===")
    print(summary_df)

