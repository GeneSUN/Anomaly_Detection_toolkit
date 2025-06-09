
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import sum, lag, col, split, concat_ws, lit ,udf,count, max,lit,avg, when,concat_ws,to_date,explode,last
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")



columns = [ "sn", "time","hour", "day", "_4gsnr", "_5gsnr",
            "4GPccBand", "4GScc1Band", "4GScc2Band", "4GScc3Band",
            "5GPccBand", "5GScc1Band", "5GEARFCN_DL",
            "_lte_band", "_nwbandwidth", "_cbandbandwidths",
            "lte_capacity", "nw_capacity", "c_band_capacity"
        ]



df_heartbeat = spark.read.option("recursiveFileLookup", "true")\
                    .parquet("/user/ZheS/5g_home_anomally/")\
                    .withColumn(
                        "day",
                        F.from_unixtime((F.col("ts") / 1000).cast("long")).cast("date")
                    )\
                    .withColumn("hour", F.date_trunc("hour", col("time")))\
                    .withColumn("time", F.to_timestamp("time"))\
                    .withColumnRenamed("SNR", "_4gsnr").withColumnRenamed("5GSNR", "_5gsnr")\
                    .filter(
                                (F.col("_4gsnr").between(-10, 40)) & (F.col("_4gsnr") != 0) & 
                                (F.col("_5gsnr").between(-10, 40)) & (F.col("_5gsnr") != 0)
                            )\
                    .withColumn(
                        "_lte_band",
                        (F.when(F.col("4GPccBand").cast("bigint") > 0, 20).otherwise(0) +
                        F.when(F.col("4GScc1Band").cast("bigint") > 0, 20).otherwise(0) +
                        F.when(F.col("4GScc2Band").cast("bigint") > 0, 20).otherwise(0) +
                        F.when(F.col("4GScc3Band").cast("bigint") > 0, 20).otherwise(0))
                    ).withColumn(
                        "_nwbandwidth",
                        (F.when((F.col("5GPccBand").cast("bigint") > 0) & (F.col("5GPccBand").cast("bigint") != 77), 20).otherwise(0) +
                        F.when((F.col("5GScc1Band").cast("bigint") > 0) & (F.col("5GScc1Band").cast("bigint") != 77), 20).otherwise(0))
                    ).withColumn(
                        "_cbandbandwidths",
                        F.when(
                            (F.col("5GPccBand").cast("bigint") == 77) & (F.col("5GScc1Band").cast("bigint") == 77), 160
                        ).when(
                            (F.col("5GPccBand").cast("bigint") == 77) & (F.col("5GEARFCN_DL").between(646667, 653329)), 100
                        ).when(
                            (F.col("5GPccBand").cast("bigint") == 77) & (~F.col("5GEARFCN_DL").between(646667, 653329)), 60
                        ).when(
                            (F.col("5GPccBand").cast("bigint") != 77) & (F.col("5GScc1Band").cast("bigint") == 77), 80
                        ).otherwise(0)
                    )\
                    .filter(
                    ( F.col("_lte_band") + F.col("_nwbandwidth") + F.col("_cbandbandwidths")) > 0
                    )\
                    .withColumn(
                        "lte_capacity",
                        F.round(
                            F.when(
                                F.col("_4gsnr") == 0, 0
                            ).otherwise(
                                F.col("_lte_band") * F.least(F.lit(1), (F.col("_4gsnr") + 11) / 41.0)
                            ), 2
                        )
                    ).withColumn(
                        "nw_capacity",
                        F.round(
                            F.when(
                                F.col("_5gsnr") == 0, 0
                            ).otherwise(
                                F.col("_nwbandwidth") * F.least(F.lit(1), (F.col("_5gsnr") + 11) / 41.0)
                            ), 2
                        )
                    ).withColumn(
                        "c_band_capacity",
                        F.round(
                            F.when(
                                F.col("_5gsnr") == 0, 0
                            ).otherwise(
                                F.col("_cbandbandwidths") * 0.8 * F.least(F.lit(1), (F.col("_5gsnr") + 10) / 41.0)
                            ), 2
                        )
                    )\
                    .orderBy("time")\
                    .select(columns)


df_cap_hour = df_heartbeat.groupBy("sn", "hour")\
                            .agg( F.round( F.avg("lte_capacity"),2).alias("avg_lte_capacity"),
                                    F.round( F.avg("_lte_band"),0).alias("avg_lte_band"),
                                    F.round( F.avg("c_band_capacity"),2).alias("avg_c_band_capacity"),
                                    F.round( F.avg("_cbandbandwidths"),0).alias("avg_cbandbandwidths"),
                                    F.round( F.avg("_5gsnr"),0).alias("avg_5gsnr"),
                                    F.round( F.avg("_4gsnr"),0).alias("avg_4gsnr")
                                    )\
                            .orderBy("hour")

df_cap_hour.write.format("parquet")\
        .mode("overwrite")\
        .save(f"/user/ZheS//owl_anomally/5g_home_anomaly_pplan50127/" ) 