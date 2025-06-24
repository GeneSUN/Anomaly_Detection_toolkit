
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

sys.path.append('/usr/apps/vmas/scripts/ZS/owl_anomaly') 
from OutlierDetector import FeaturewiseKDENoveltyDetector, DBSCANOutlierDetector, ARIMAAnomalyDetector



if __name__ == "__main__":

    spark = SparkSession.builder\
            .appName('HourlyScoreProcessing')\
            .config("spark.sql.adapative.enabled","true")\
            .config("spark.ui.port","24041")\
            .enableHiveSupport().getOrCreate()
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

    
    def detect_anomalies_for_all_sn(df, feature_col="avg_5gsnr", time_col="hour"):
        results = []
        for sn_val, group in df.groupby("sn"):
            group_sorted = group.sort_values(by=time_col).reset_index(drop=True)
            """
            detector = FeaturewiseKDENoveltyDetector(
                df=group_sorted,
                feature_col=feature_col,
                time_col=time_col,
                train_idx=slice(0, 1068),
                new_idx=slice(-26, None),
                train_percentile=99  # or 100 if you don't want to filter
            )
            output = detector.fit()
            """
            detector = DBSCANOutlierDetector(df, features=["x1", "x2"], eps=3, min_samples=2)
            output = detector.fit()            

            results.append({
                "sn": sn_val,
                "outlier_count": output["outlier_count"],
                "total_new_points": output["total_new_points"]
            })
        return results
    result = detect_anomalies_for_all_sn(df_snr)

    df_result = spark.createDataFrame(pd.DataFrame(result))

    df_result.show(truncate=False)
    df_result.printSchema()

    df_result.write.mode("overwrite")\
           .parquet( hdfs_pd + "/user/ZheS//owl_anomally//anomally_result/kde") 
                    
    #

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds for {df_result.count()} customer")