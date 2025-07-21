
from datetime import datetime, timedelta, date
from pyspark.sql.window import Window
from pyspark.sql.functions import sum, lag, col, split, concat_ws, lit ,udf,count, max,lit,avg, when,concat_ws,percentile_approx,explode
from pyspark.sql.functions import udf 
from pyspark.sql.types import FloatType
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import numpy as np
import traceback
import sys 
sys.path.append('/usr/apps/vmas/scripts/ZS') 
from MailSender import MailSender
import argparse 
from functools import reduce
import time


def convert_to_numeric(df, col_name):
    return df.withColumn(f"{col_name}_numeric", F.when(F.col(col_name) == "Poor", 1)
                                                .when(F.col(col_name) == "Fair", 2)
                                                .when(F.col(col_name) == "Good", 3)
                                                .when(F.col(col_name) == "Excellent", 4)
                                                .otherwise(None))
def convert_to_categorical(df, col_name):
    return df.withColumn(col_name, 
                        F.when(F.col(col_name) < 1.5, "Poor")
                        .when((F.col(col_name) >= 1.5) & (F.col(col_name) < 2.5), "Fair")
                        .when((F.col(col_name) >= 2.5) & (F.col(col_name) < 3.5), "Good")
                        .when(F.col(col_name) >= 3.5, "Excellent")
                        .otherwise(None))

class ScoreCalculator: 
    def __init__(self, weights): 
        self.weights = weights 
 
    def calculate_score(self, *args): 
        total_weight = 0 
        score = 0 

        for weight, value in zip(self.weights.values(), args): 
            if value is not None: 
                score += weight * float(value) 
                total_weight += weight 

        return score / total_weight if total_weight != 0 else None 
    
class CellularScore:
    global hdfs_pa, hdfs_pd, count_features

    hdfs_pd = "hdfs://njbbvmaspd11.nss.vzwnet.com:9000/"
    hdfs_pa =  'hdfs://njbbepapa1.nss.vzwnet.com:9000'
    count_features = ["LTERACHFailureCount", "LTEHandOverFailureCount", "NRSCGChangeFailureCount","RRCConnectFailureCount"]
    
    def __init__(self,d,hour,df_heartbeat): 
        self.d = d
        self.hour = hour
        self.df_heartbeat = df_heartbeat
        """
        self.df_heartbeat = spark.read.option("header","true").csv( hdfs_pa + f"/user/kovvuve/owl_history_v3/date={self.d}/hr={self.hour}" )\
                                .dropDuplicates()\
                                .withColumn('time', F.from_unixtime(col('ts') / 1000.0).cast('timestamp'))
        """
        self.custline_path = hdfs_pa + "/user/kovvuve/EDW_SPARK/cust_line/"+ self.d
        self.df_price_cap = self.get_price_plan_df()
        self.df_cust = self.get_customer_df()
        self.df_throughput = self.get_throughput_df()
        self.df_linkCapacity = self.get_linkCapacity_df()
        self.df_ServiceTime = self.get_ServiceTime_df()
        self.df_score = self.get_score_df()

    def get_price_plan_df(self):
        """
        price_plan_data = [
            ('67577', 50, 6), ('50011', 50, 6), ('38365', 50, 6), ('50010', 50, 6), ('75565', 50, 6), 
            ('65655', 50, 6), ('67584', 50, 6), ('65656', 50, 6), ('67571', 100, 10), ('50128', 300, 20), 
            ('50127', 300, 20), ('75561', 300, 20), ('67576', 300, 20), ('50130', 300, 20), ('50129', 300, 20), 
            ('67567', 400, 20), ('50044', 400, 20), ('50116', 1500, 75), ('67568', 1500, 75), ('75560', 1500, 75)
        ]
        """
        price_plan_data = [
                            ('38365', 50, 6), ('39425', 1500, 75), ('39428', 1500, 75), ('46798', 10, 5), ('46799', 25, 5),
                            ('48390', 10, 5), ('48423', 25, 5), ('48445', 50, 6), ('50010', 50, 6), ('50011', 50, 6),
                            ('50044', 300, 20), ('50055', 300, 20), ('50116', 1500, 75), ('50117', 1500, 75), ('50127', 300, 20),
                            ('50128', 300, 20), ('50129', 300, 20), ('50130', 300, 20), ('51219', 150, 10), ('53617', 300, 20),
                            ('65655', 50, 6), ('65656', 50, 6), ('67567', 400, 20), ('67568', 1500, 75), ('67571', 100, 10),
                            ('67576', 300, 20), ('67577', 50, 6), ('67584', 50, 6), ('75560', 1500, 75), ('75561', 300, 20),
                            ('75565', 50, 6)
                            ]

        columns = ['PPLAN_CD', 'DL_CAP', 'UL_CAP']

        df_price_cap = spark.createDataFrame(price_plan_data, columns)
        return df_price_cap

    def get_customer_df(self, custline_path = None):
        if custline_path is None:
            custline_path = self.custline_path

        df_mapping = spark.read.option("header","true").csv(hdfs_pa + "/sha_data/combinedsnmappingv2")\
                    .select("mdn","sn").distinct()\
                    .withColumnRenamed("mdn", "MDN_5G")\
        
        df_cust = spark.read.option("recursiveFileLookup", "true").option("header", "true")\
                        .csv(custline_path)\
                        .withColumnRenamed("VZW_IMSI", "IMSI")\
                        .withColumnRenamed("MTN", "MDN_5G")\
                        .withColumn("IMEI", F.expr("substring(IMEI, 1, length(IMEI)-1)"))\
                        .withColumn("CPE_MODEL_NAME", F.split(F.trim(F.col("DEVICE_PROD_NM")), " "))\
                        .withColumn("CPE_MODEL_NAME", F.col("CPE_MODEL_NAME")[F.size("CPE_MODEL_NAME") - 1])\
                        .select("IMSI", "MDN_5G", "PPLAN_CD", "PPLAN_DESC", "CPE_MODEL_NAME")\
                        .dropDuplicates()\
                        .join( df_mapping, "MDN_5G" )
        
        return df_cust
    
    def get_throughput_df(self, df_cust = None,df_price_cap = None):
        if df_cust is None:
            df_cust = self.df_cust
        if df_price_cap is None:
            df_price_cap = self.df_price_cap

        ultra_schema = StructType([
            StructField("IMSI", StringType(), True),
            StructField("UE_OVERALL_DL_SPEED", DoubleType(), True)
        ])

        # Try to read the CSV and handle the case where it might not exist
        try:
            date_val = datetime.strptime(self.d, '%Y-%m-%d')
            prev_dates = [(date_val - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 4)]

            # Read data from the previous 3 days
            df_ultra = spark.read.option("header", "true") \
                .csv([hdfs_pa + f"/fwa/npp_mdn_agg_insights_rtt/datadate={date}" for date in prev_dates]) \
                .select("IMSI", 'UE_OVERALL_DL_SPEED') \
                .filter(F.col("UE_OVERALL_DL_SPEED").isNotNull()) \
                .filter(F.col("UE_OVERALL_DL_SPEED") != 0) \
                .groupBy('IMSI') \
                .agg(F.avg('UE_OVERALL_DL_SPEED').alias('UE_OVERALL_DL_SPEED'))
            
        except Exception as e:
            email_sender.send(
                    send_from="cellular_Score@verizon.com",
                    subject=f"ultragauge missed at {self.d}",
                    text=e
                )
            df_ultra = spark.createDataFrame([], ultra_schema)  # Create an empty DataFrame if not exists

        # Continue with df_ultra as normal
        df_ultrag_price_cap = df_cust.join(df_ultra, "IMSI", "left")\
            .join(df_price_cap, "PPLAN_CD", "left")\
            .withColumn(
                "ULTRAGAUGE_DL_SCORE",
                F.round(
                    F.when((F.col("UE_OVERALL_DL_SPEED") / F.col("DL_CAP") * 2) > 1, 1)
                    .otherwise(F.col("UE_OVERALL_DL_SPEED") / F.col("DL_CAP") * 2), 4)
            )\
            .withColumn(
                "ULTRAGAUGE_DL_SCORE", col("ULTRAGAUGE_DL_SCORE")*100
            )
        
        return df_ultrag_price_cap

    def get_linkCapacity_df(self, df_heartbeat = None, df_cust = None, df_price_cap = None):
        if df_heartbeat is None:
            df_heartbeat = self.df_heartbeat
        if df_cust is None:
            df_cust = self.df_cust
        if df_price_cap is None:
            df_price_cap = self.df_price_cap

        df_heartbeat = df_heartbeat.join(df_cust, ["sn","IMSI"], "right")\
                                    .join(df_price_cap, "PPLAN_CD", "right")\

        df_with_bandwidths = df_heartbeat.withColumnRenamed("SNR", "_4gsnr").withColumnRenamed("5GSNR", "_5gsnr")\
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
                                        )

        df_linkCapacity = df_with_bandwidths.filter(
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
                                        .withColumn(
                                                "Rate_Plan_Adjustment", 
                                                F.least(F.col("DL_CAP") / 150, F.lit(1.0))
                                            )\
                                        .withColumn(
                                            "_capacity", 
                                            F.round(
                                                100*((F.col("lte_capacity") + F.col("nw_capacity") + F.col("c_band_capacity")) /  (218*col("Rate_Plan_Adjustment") ) ),
                                                2)
                                            )\
                                        .withColumn(
                                            "_capacity", 
                                            F.round(
                                                        F.when((F.col("_capacity")) > 100, 100)
                                                        .otherwise(F.col("_capacity")), 
                                                    4) )\
                                        .groupby("sn", "MDN_5G")\
                                        .agg( 
                                            F.round(F.avg("lte_capacity"),2).alias("lte_capacity"), 
                                            F.round(F.avg("nw_capacity"),2).alias("nw_capacity"), 
                                            F.round(F.avg("c_band_capacity"),2).alias("c_band_capacity"), 
                                            F.round(F.avg("Rate_Plan_Adjustment"),2).alias("Rate_Plan_Adjustment"), 
                                            F.round(F.avg("_capacity"),2).alias("capacity_score") 
                                            )\
                                        .withColumn( "capacity_score_category", 
                                                    when(col("capacity_score").isNull(), None)
                                                    .when(col("capacity_score") >= 80, "Excellent")
                                                    .when(col("capacity_score") >= 50, "Good")
                                                    .when(col("capacity_score") >= 30, "Fair")
                                                    .otherwise("Poor") )

        return df_linkCapacity

    def get_ServiceTime_df(self, df_heartbeat = None):

        if df_heartbeat is None:
            df_heartbeat = self.df_heartbeat
        
        window_spec = Window.partitionBy("sn").orderBy("ServiceUptime") 

        df_heartbeat = df_heartbeat.filter( (col("ServiceDowntime")!="184467440737095")&
                                                (col("ServiceUptime")!="184467440737095")
                                                )\
                                    .withColumn("ServiceDowntime_change", 
                                            when(col("ServiceDowntime") != F.lag("ServiceDowntime").over(window_spec), 1).otherwise(0))\
                                    .withColumn("_ServiceUptime_change", 
                                            when(col("ServiceUptime") == F.lag("ServiceUptime").over(window_spec), 1).otherwise(0))\
                                    .withColumn("ServiceUptime_change", 
                                            when(col("ServiceUptime") != F.lag("ServiceUptime").over(window_spec), 1).otherwise(0))
                                            
        for feature in count_features: 
            # It is tricky of whether | filter( col(feature)!=0 ) |
            df_heartbeat = df_heartbeat\
                                    .withColumn("prev_"+feature, F.lag(feature).over(window_spec))\
                                    .withColumn("pre<cur", 
                                                F.when(F.col("prev_"+feature) <= F.col(feature) , 1).otherwise(0))\
                                    .withColumn("increment_" + feature, 
                                                F.when((F.col("pre<cur") == 1) & (F.col("prev_" + feature).isNotNull()), 
                                                    F.col(feature) - F.col("prev_" + feature)) 
                                                .otherwise(F.coalesce(F.col(feature), F.lit(0) )))
        
        date_str = self.d.replace("-", "")
        df_modem_crash = spark.read.parquet(hdfs_pa + f"/sha_data/OWLHistory/date={date_str}/hour={self.hour}")\
                                .select("rowkey","ts","Tplg_Data_model_name","Owl_Data_modem_event")\
                                .withColumn("sn", F.regexp_extract(F.col("rowkey"), r'-(\w+)_', 1))\
                                .filter( (F.col("Owl_Data_modem_event").isNotNull()) )\
                                .withColumn("datetime", F.from_unixtime(F.col("ts") / 1000).cast("timestamp"))\
                                .withColumn("day", F.to_date("datetime") )\
                                .groupBy("sn") \
                                .agg(F.count("*").alias("num_mdm_crashes"))

        sum_columns = [F.sum("increment_" + feature).alias("sum_" + feature) for feature in count_features] 
        df_count = df_heartbeat.groupby("sn")\
                                .agg( 
                                    *sum_columns,
                                    sum("ServiceDowntime_change").alias("ServiceDowntime_sum"),
                                    sum("ServiceDowntime_change").alias("_ServiceUptime_sum"),
                                    sum("ServiceUptime_change").alias("ServiceUptime_sum"),
                                    )\
                                .join( df_modem_crash, "sn" )\
                                .withColumn("ServicetimePercentage", 100*col("_ServiceUptime_sum")/(col("_ServiceUptime_sum")+col("ServiceUptime_sum") ) )\
                                .withColumn( "assumed_downtime", F.col("num_mdm_crashes") * 90 + F.col("sum_RRCConnectFailureCount") * 1 + F.col("sum_LTERACHFailureCount") * 0.01 + F.col("_ServiceUptime_sum")*300 )\
                                .withColumn(
                                    "not_available_percentage",
                                    (F.col("assumed_downtime") / ( 24*60*60 )) * 100
                                )\
                                .withColumn( "availability_score",
                                            F.when(
                                                (100 - 20 * F.col("not_available_percentage") ) < 0, 
                                                0 
                                            ).otherwise(
                                                F.round(100 - 20 * F.col("not_available_percentage") , 2) 
                                            )
                                        )\
                                .withColumn( "availability_score_category", 
                                                when(col("availability_score").isNull(), None)
                                                .when(col("availability_score") == 100, "Excellent")
                                                .when(col("availability_score") >= 99.77, "Good")
                                                .when(col("availability_score") >= 97.22, "Fair")
                                                .otherwise("Poor") )


        return df_count
    
    def get_score_df(self, df_throughput = None, df_linkCapacity = None, df_ServiceTime = None):
        if df_throughput is None:
            df_throughput = self.df_throughput
        if df_linkCapacity is None:
            df_linkCapacity = self.df_linkCapacity
        if df_ServiceTime is None:
            df_ServiceTime = self.df_ServiceTime

        df_join = df_throughput.join(df_linkCapacity, ["sn","MDN_5G"], "full" )\
                                .join(df_ServiceTime, "sn" ,"full" )

        throughput_score_weights = {
                                    "ultragauge_dl_score": 28,
                                }
        throughput_score_calculator = ScoreCalculator(throughput_score_weights)
        throughput_score_udf = udf(throughput_score_calculator.calculate_score, FloatType())

        from pyspark.sql.functions import sum, lag, col
        df_score = df_join.withColumn(
                                        "throughput_score",
                                        F.round(
                                            throughput_score_udf(*[col(column) for column in throughput_score_weights.keys()]), 
                                            2
                                        )
                                    )\
                            .withColumn(
                                        "throughput_score_category",
                                        when(col("throughput_score").isNull(), None)  # Set NULL if throughput_score is NULL
                                        .when(col("throughput_score") >= 80, "Excellent")
                                        .when(col("throughput_score") >= 60, "Good")
                                        .when(col("throughput_score") >= 30, "Fair")
                                        .otherwise("Poor")
                                    )


        categorical_columns = ["throughput_score_category", "capacity_score_category", "availability_score_category",]

        for col_name in categorical_columns:
            df_score = convert_to_numeric(df_score, col_name) #"{col_name}_numeric"


        score_weights = {
                            "availability_score_category_numeric": 5,
                            "capacity_score_category_numeric": 2,
                            "throughput_score_category_numeric": 1,
                        }
        score_calculator = ScoreCalculator(score_weights)
        score_udf = udf(score_calculator.calculate_score, FloatType())

        df_score = df_score.withColumn(
                                        "score",
                                        F.round(
                                            score_udf(*[col(column) for column in score_weights.keys()]), 
                                            2
                                        )
                                    )
        
        df_score = convert_to_categorical(df_score, "score")

        return df_score





if __name__ == "__main__":
    email_sender = MailSender()
    spark = SparkSession.builder\
            .appName('HourlyScoreProcessing')\
            .config("spark.sql.adapative.enabled","true")\
            .config("spark.ui.port","24041")\
            .enableHiveSupport().getOrCreate()


    from pyspark.sql.functions import col, from_json
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType

    schema = StructType([
        StructField("MDN", StringType(), True),
        StructField("SIMState", IntegerType(), True),
        StructField("IMSI", StringType(), True),
        StructField("IMEI", StringType(), True),
        StructField("SwV", StringType(), True),
        StructField("Status", BooleanType(), True),
        StructField("5GUptimeTimestamp", StringType(), True),
        StructField("5GDowntimeTimestamp", StringType(), True),
        StructField("B1MeasurementConfigurationStatus", BooleanType(), True),
        StructField("B1MeasurementConfigurationBands", StringType(), True),
        StructField("SNR", StringType(), True),
        StructField("CurrentNetwork", StringType(), True),
        StructField("HomeRoam", StringType(), True),
        StructField("MCC", StringType(), True),
        StructField("MNC", StringType(), True),
        StructField("CellID", IntegerType(), True),
        StructField("PCellID", StringType(), True),
        StructField("TotalBytesReceived", IntegerType(), True),
        StructField("TotalBytesSent", IntegerType(), True),
        StructField("TotalPacketReceived", IntegerType(), True),
        StructField("TotalPacketSent", IntegerType(), True),
        StructField("MCS", StringType(), True),
        StructField("PathLoss", IntegerType(), True),
        StructField("BRSRP", DoubleType(), True),
        StructField("EARFCN_DL", IntegerType(), True),
        StructField("EARFCN_UL", IntegerType(), True),
        StructField("5GEARFCN_DL", StringType(), True),
        StructField("5GEARFCN_UL", StringType(), True),
        StructField("PUCCH_TX_PWR", DoubleType(), True),
        StructField("CQI", IntegerType(), True),
        StructField("Rank", IntegerType(), True),
        StructField("MaxMTUSize", IntegerType(), True),
        StructField("LTERadioLinkFailureCount", IntegerType(), True),
        StructField("LTERACHAttemptCount", IntegerType(), True),
        StructField("LTERACHFailureCount", IntegerType(), True),
        StructField("RRCConnectTime", StringType(), True),
        StructField("RRCConnectRequestCount", IntegerType(), True),
        StructField("RRCConnectFailureCount", IntegerType(), True),
        StructField("NRSCGChangeCount", IntegerType(), True),
        StructField("NRSCGChangeFailureCount", IntegerType(), True),
        StructField("LTEHandOverAttemptCount", IntegerType(), True),
        StructField("LTEHandOverFailureCount", IntegerType(), True),
        StructField("LTEPDSCHThroughput", DoubleType(), True),
        StructField("LTEPDSCHPeakThroughput", DoubleType(), True),
        StructField("LTEPUSCHThroughput", DoubleType(), True),
        StructField("LTEPUSCHPeakThroughput", DoubleType(), True),
        StructField("RxPDCPBytes", IntegerType(), True),
        StructField("TxPDCPBytes", IntegerType(), True),
        StructField("4GRSRP", IntegerType(), True),
        StructField("4GRSRQ", IntegerType(), True),
        StructField("4GSignal", IntegerType(), True),
        StructField("5GPCI", StringType(), True),
        StructField("RSRQ", DoubleType(), True),
        StructField("5GSNR", DoubleType(), True),
        StructField("NRPDSCHInitBLER", IntegerType(), True),
        StructField("NRPUSCHInitBLER", IntegerType(), True),
        StructField("GPSEnabled", BooleanType(), True),
        StructField("GPSAltitude", StringType(), True),
        StructField("GPSLatitude", StringType(), True),
        StructField("GPSLongitude", StringType(), True),
        StructField("5GModemTempThreshold", StringType(), True),
        StructField("5GNRSub6AntennaTempThreshold", StringType(), True),
        StructField("4GAntennaTempThreshold", StringType(), True),
        StructField("ModemTemp", StringType(), True),
        StructField("5GNRSub6AntennaTemp", StringType(), True),
        StructField("4GAntennaTemp", StringType(), True),
        StructField("4GTempFallback", BooleanType(), True),
        StructField("4GTempFallbackCause", IntegerType(), True),
        StructField("5GServiceThermalDegradation", BooleanType(), True),
        StructField("5GServiceThermalDegradationCause", IntegerType(), True),
        StructField("ModemLoggingEnabled", BooleanType(), True),

        StructField("4GPccBand", IntegerType(), True),
        StructField("4GScc1Band", IntegerType(), True),
        StructField("4GScc2Band", IntegerType(), True),
        StructField("4GScc3Band", IntegerType(), True),

        StructField("5GPccBand", IntegerType(), True),
        StructField("5GScc1Band", IntegerType(), True),
        StructField("ServiceUptime", StringType(), True),
        StructField("ServiceDowntime", StringType(), True),
        StructField("ServiceUptimeTimestamp", StringType(), True),
        StructField("ServiceDowntimeTimestamp", StringType(), True),
        StructField("5GUW_Allowed", BooleanType(), True),
        StructField("5GNRRadioLinkFailureCount", IntegerType(), True),
        StructField("5GNRRACHAttemptCount", IntegerType(), True),
        StructField("5GNRRACHFailureCount", IntegerType(), True),
        StructField("5GNRRRCConnectTime", StringType(), True),
        StructField("5GNRRRCConnectRequestCount", IntegerType(), True),
        StructField("5GNRRRCConnectFailureCount", IntegerType(), True),
        StructField("5GNRHandOverAttemptCount", IntegerType(), True),
        StructField("5GNRHandOverFailureCount", IntegerType(), True),
        StructField("5GNRPDSCHThroughput", DoubleType(), True),
        StructField("5GNRPUSCHThroughput", DoubleType(), True),
        StructField("5GNRPDSCHPeakThroughput", DoubleType(), True),
        StructField("5GNRPUSCHPeakThroughput", DoubleType(), True),
        StructField("5GNRRxPDCPBytes", IntegerType(), True),
        StructField("5GNRTxPDCPBytes", IntegerType(), True),
        StructField("NRSCGFailureCount", IntegerType(), True),
        StructField("CPUUsage", StringType(), True),
        StructField("Uptime", StringType(), True),
        StructField("RebootCause", StringType(), True),
        StructField("Manufacturer", StringType(), True),
        StructField("ModelName", StringType(), True),
        StructField("FmV", StringType(), True),
        StructField("HwV", StringType(), True),
        StructField("MemoryAvail", StringType(), True),
        StructField("MemoryPercentFree", DoubleType(), True),
        StructField("ipv4_ip", StringType(), True),
        StructField("ipv6_ip", StringType(), True)
    ])


    hdfs_base_path = "/sha_data/OWLHistory/"
    output_base_path = "/user/ZheS/cpe_Score/hourly_score/"


    def process_hourly_data(date_str, hour_str):
        """Function to process data for a given hour."""
        
        hdfs_path = f"{hdfs_base_path}date={date_str}/hour={hour_str}"
        
        df_owl = spark.read.parquet(hdfs_path)\
                    .filter(col("Owl_Data_fwa_cpe_data").isNotNull())\
                    .withColumn("fwa_cpe_data", from_json(col("Owl_Data_fwa_cpe_data"), schema))\
                    .select("rowkey", "ts", "Tplg_Data_model_name", "fwa_cpe_data.*")\
                    .withColumn("SNR", col("SNR").cast("double"))\
                    .dropDuplicates()\
                    .withColumn("sn", F.regexp_extract(F.col("rowkey"), r'-(\w+)_', 1))\
                    .withColumn('time', F.from_unixtime(col('ts') / 1000.0).cast('timestamp'))

        ins = CellularScore(d=date_str, hour=hour_str, df_heartbeat=df_owl)
        
        output_path = f"{output_base_path}{date_str}/hr={hour_str}"
        ins.df_score.write.mode("overwrite").parquet(output_path)


    hadoop_fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    def monitor_and_process():
        """Monitors HDFS and processes hourly files as they arrive."""
        date_str = (datetime.today() ).strftime("%Y-%m-%d")
        
        processed_hours = set()  # Track processed hours
        
        for hour in range(24):
            hour_str = f"{hour:02d}"
            hdfs_path = f"{hdfs_base_path}date={date_str}/hour={hour_str}"
            
            # Wait for the file to appear
            while True:
                if hour_str in processed_hours:
                    break  # Skip if already processed
                if hadoop_fs.exists(spark._jvm.org.apache.hadoop.fs.Path(hdfs_path)):
                    print(f"Found {hdfs_path}, processing...")
                    process_hourly_data(date_str, hour_str)
                    processed_hours.add(hour_str)
                    break
                
                print(f"Waiting for {hdfs_path} to be created...")
                time.sleep(300)  # Check every 5 minutes



    monitor_and_process()

    """
    date_str = (date.today() - timedelta(1) ).strftime("%Y-%m-%d")

    for hour in range(24):
        start_time = datetime.now()
        hour_str = f"{hour:02d}"

        df_owl = spark.read.parquet(hdfs_pa + f"/sha_data/OWLHistory/date=20250317/hour={hour_str}")\
                        .filter( col("Owl_Data_fwa_cpe_data").isNotNull() )\
                        .withColumn("fwa_cpe_data", from_json(col("Owl_Data_fwa_cpe_data"), schema))\
                        .select("rowkey","ts","Tplg_Data_model_name","fwa_cpe_data.*")\
                        .withColumn("SNR", col("SNR").cast("double"))\
                        .dropDuplicates()\
                        .withColumn("sn", F.regexp_extract(F.col("rowkey"), r'-(\w+)_', 1))\
                        .withColumn('time', F.from_unixtime(col('ts') / 1000.0).cast('timestamp'))

        ins = CellularScore(d=date_str, hour=hour_str, df_heartbeat = df_owl)
        output_path = f"/user/ZheS/cpe_Score/hourly_score/{date_str}/hr={hour_str}"
        ins.df_score.write.mode("overwrite").parquet(output_path)
        end_time = datetime.now()
        print(f"Processed hour {hour_str}, Time taken: {end_time - start_time}")

    """