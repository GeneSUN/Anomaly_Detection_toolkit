
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, asc
from typing import List, Optional
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col

def split_time_series(
    df: DataFrame,
    window_size: int,
    order_by: str = "time",
    overlap: float = None,
    shift: int = None
) -> list:
    """
    Split a PySpark time series DataFrame into overlapping or shifting sub-sequences.

    Parameters:
        df (DataFrame): Input time series DataFrame.
        window_size (int): Number of rows in each split.
        order_by (str): Column name to sort chronologically. Default: "time".
        overlap (float): Overlap percentage between windows (0–100). Mutually exclusive with `shift`.
        shift (int): Fixed number of rows to shift between windows. Mutually exclusive with `overlap`.

    Returns:
        List[DataFrame]: List of sliced DataFrames of length `window_size`.
    """
    assert (overlap is not None) ^ (shift is not None), "Specify exactly one of `overlap` or `shift`."
    assert 0 < window_size <= df.count(), "Invalid window size."

    # Step 1: Sort and add row index
    df_sorted = df.orderBy(order_by)
    df_indexed = df_sorted.withColumn("row_idx", row_number().over(Window.orderBy(order_by)) - 1)

    # Step 2: Calculate step size
    if overlap is not None:
        step = int(window_size * (1 - overlap / 100))
        step = max(1, step)
    else:
        step = shift

    # Step 3: Slice into overlapping/shifting windows
    total_rows = df_indexed.count()
    slices = []
    start = 0

    while start + window_size <= total_rows:
        end = start + window_size
        window_df = df_indexed.filter((col("row_idx") >= start) & (col("row_idx") < end)).drop("row_idx")

        slice_id = window_df.agg(min_("timestamp_int")).collect()[0][0]
        window_df = window_df.withColumn("slice_id", lit(slice_id))
        window_df.write.parquet(f"/user/ZheS/owl_anomaly//processed_data/novelt/y{slice_id}")

        slices.append(window_df)
        start += step

    return slices

def slice_time_series_by_sn(df, sequence_length=168):
    """
    Slice each SN's time series into continuous sequences of fixed length.

    Parameters:
    df (DataFrame): Input PySpark DataFrame with 'sn' and 'hour' (already in TimestampType).
    sequence_length (int): Length of each time series slice (e.g., 168 for a week of hourly data).

    Returns:
    DataFrame: Sliced DataFrame with columns 'week_index' and 'slice_id'.
    """
    # Step 1: Assign row numbers per 'sn' ordered by time
    window_spec = Window.partitionBy("sn").orderBy("time")
    df_with_index = df.withColumn("row_num", F.row_number().over(window_spec))

    # Step 2: Compute the slice group (e.g., week index)
    df_with_group = df_with_index.withColumn("slice_id", ((F.col("row_num") - 1) / sequence_length).cast("int"))

    # Step 3: Only retain full sequences (exactly `sequence_length` points)
    valid_groups = (
        df_with_group.groupBy("sn", "slice_id")
        .agg(F.count("*").alias("cnt"))
        .filter(F.col("cnt") == sequence_length)
        .select("sn", "slice_id")
    )
    df_valid = df_with_group.join(valid_groups, on=["sn", "slice_id"], how="inner")

    # Step 4: Create a unique identifier for each slice
    df_sliced = df_valid.withColumn("slice_id", F.concat_ws("_", F.col("sn"), F.col("slice_id")))

    return df_sliced.drop("row_num")
