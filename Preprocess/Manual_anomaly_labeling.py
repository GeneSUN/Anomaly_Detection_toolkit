#%matplotlib inline

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import sum, lag, col, split, concat_ws, lit ,udf,count, max,lit,avg, when,concat_ws,to_date,explode,last
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

from datetime import datetime
import matplotlib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class RandomSNRFetcher:
    def __init__(
        self,
        base_anomaly_path,
        snr_data_path,
        features=["avg_4gsnr"],
        sample_fraction=0.1,
        limit=1
    ):
        self.base_anomaly_path = base_anomaly_path
        self.snr_data_path = snr_data_path
        self.features = features if isinstance(features, list) else [features]
        self.sample_fraction = sample_fraction
        self.limit = limit
        self.sn = None
        self.df_snr_filtered = None

    def get_random_sn(self):
        random_list = (
            spark.read.option("recursiveFileLookup", "true")
            .parquet(self.base_anomaly_path)
            .select("sn", "pplan_desc", "cpe_model_name")
            .distinct()
            .sample(self.sample_fraction)
            .limit(self.limit)
            .collect()
        )
        if not random_list:
            raise ValueError("No SN found in the sampled data.")
        self.sn = random_list[0]["sn"]
        return self.sn

    def filter_snr_data(self):
        if not self.sn:
            raise ValueError("Call get_random_sn() before filtering SNR data.")

        columns = ["sn", "hour"] + self.features
        df_snr = (
            spark.read.parquet(self.snr_data_path)
            .select(*columns)
            .orderBy("sn", "hour")
            .toPandas()
        )
        self.df_snr_filtered = df_snr[df_snr["sn"] == self.sn]
        return self.df_snr_filtered

fetcher = RandomSNRFetcher(
    base_anomaly_path="/user/ZheS/5g_home_anomally/",
    snr_data_path="/user/ZheS//owl_anomally/capacity_records/",
    features=["avg_4gsnr","avg_5gsnr"],
)

sn_value = fetcher.get_random_sn()
df_cap_hour_pd = fetcher.filter_snr_data()
print(df_cap_hour_pd.head())