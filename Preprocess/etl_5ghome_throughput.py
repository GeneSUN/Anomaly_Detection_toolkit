
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import sum, lag, col, split, concat_ws, lit ,udf,count, max,lit,avg, when,concat_ws,to_date,explode,last
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")


snr_data_path = "/user/ZheS/5g_home_anomally/"

df_owl = spark.read.option("recursiveFileLookup", "true")\
                .parquet(snr_data_path)\
                .withColumn(
                        "day",
                        F.from_unixtime((F.col("ts") / 1000).cast("long")).cast("date")
                    )\
                .withColumn("hour", F.date_trunc("hour", col("time")))\

throughput_data = [
    "LTEPDSCHPeakThroughput", "LTEPDSCHThroughput",
    "LTEPUSCHPeakThroughput", "LTEPUSCHThroughput",
    "TxPDCPBytes", "RxPDCPBytes",
    "TotalBytesReceived", "TotalBytesSent",
    "TotalPacketReceived", "TotalPacketSent",
]

agg_exprs = [F.round(F.avg(col), 2).alias(col) for col in throughput_data]

df_result = (
    df_owl
    .groupBy("sn", "hour")
    .agg(*agg_exprs)
    .orderBy("hour")
)

df_result.write.format("parquet")\
            .mode("overwrite")\
            .save(f"/user/ZheS/owl_anomally/throughput_records/" ) 
