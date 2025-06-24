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
from OutlierDetector import FeaturewiseKDENoveltyDetector, DBSCANOutlierDetector, ARIMAAnomalyDetector


if __name__ == "__main__":
    email_sender = MailSender()
    spark = SparkSession.builder\
            .appName('KDENoveltyDetector')\
            .config("spark.ui.port","24041")\
            .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    import time
    start_time = time.time()  # ⏱ Start timing

    # Constants
    snr_data_path = "/user/ZheS//owl_anomally/capacity_records/"
    result_base_path = "/user/ZheS//owl_anomally//anomally_result/"
    feature_cols = ["avg_5gsnr", "avg_4gsnr"]
    time_col = "hour"



    hdfs_pd = "hdfs://njbbvmaspd11.nss.vzwnet.com:9000/"
    hdfs_pa =  'hdfs://njbbepapa1.nss.vzwnet.com:9000'

    def prepare_data(min_rows=100):
        columns = ["sn", time_col] + feature_cols
        df = spark.read.parquet(snr_data_path).select(columns)
        sn_counts = df.groupBy("sn").count()
        sn_valid = sn_counts.filter(F.col("count") >= min_rows).select("sn")
        df_filtered = df.join(sn_valid, on="sn", how="inner").toPandas()
        return df_filtered

    def detect_anomaly_group(args, method, feature_col = None, feature_cols = None):
        sn_val, group, time_col = args
        group_sorted = group.sort_values(by=time_col).reset_index(drop=True)

        if method == "KDE":
            detector = FeaturewiseKDENoveltyDetector(
                df=group_sorted,
                feature_col=feature_col,
                time_col=time_col,
                train_idx=slice(0, 1068),
                new_idx=slice(-26, None),
                train_percentile=99
            )
            output = detector.fit()

        elif method == "ARIMA":
            detector = ARIMAAnomalyDetector(
                df=group_sorted,
                time_col=time_col,
                feature=feature_col,
                season_length=1
            )
            detector.run()
            output = detector.get_recent_anomaly_stats(num_recent_points=26)

        elif method == "DBSCAN":
            detector = DBSCANOutlierDetector(group_sorted, 
                                            features=["avg_4gsnr", "avg_5gsnr"], 
                                            eps=3, 
                                            min_samples=2,
                                            scale = False,
                                            recent_window_size=24)
            output = detector.fit()

        else:
            raise ValueError(f"Unsupported method: {method}")

        return {
            "sn": sn_val,
            "outlier_count": output["outlier_count"],
            "total_new_points": output["total_new_points"]
        }

    def detect_anomalies_parallel(df, method, feature_col="avg_5gsnr", time_col="hour", n_workers=16):
        args = [(sn_val, group, time_col) for sn_val, group in df.groupby("sn")]
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(detect_anomaly_group, arg, method, feature_col) for arg in args]
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def run_experiments(method, df, worker_list, feature_col):
        timing_results = []

        for workers in worker_list:
            print(f"\nRunning {method} with {workers} workers...")
            start = time.time()
            result = detect_anomalies_parallel(df, method=method, feature_col=feature_col, n_workers=workers)
            duration = time.time() - start

            df_result = pd.DataFrame(result)
            df_result_spark = spark.createDataFrame(df_result)
            output_path = f"{result_base_path}/{method}/{method}_{workers}"
            df_result_spark.write.mode("overwrite").parquet(output_path)

            timing_results.append({
                "method": method,
                "workers": workers,
                "duration_sec": duration,
                "num_serial_numbers": len(result)
            })
            print(f"Completed in {duration:.2f} seconds with {len(result)} serial numbers.")
        
        return timing_results


    df = prepare_data()
    #worker_list = [200, 150, 100, 64, 32, 16, 8]
    worker_list = [200]
    all_results = []

    for method in ["ARIMA", "DBSCAN", "KDE"]:
    #for method in ["DBSCAN"]:
        feature_col = "avg_5gsnr"  # you can switch this depending on the use case

        results = run_experiments(method, df, worker_list, feature_col)
        all_results.extend(results)

    # Final timing summary
    summary_df = pd.DataFrame(all_results)
    print("\n=== Summary of All Runs ===")
    print(summary_df)

