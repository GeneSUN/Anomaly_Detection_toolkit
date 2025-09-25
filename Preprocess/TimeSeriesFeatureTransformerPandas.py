import pandas as pd
import numpy as np
from typing import List, Union
import matplotlib.pyplot as plt

class TimeSeriesFeatureTransformerPandas:
    """
    A pipeline for transforming time-series features step by step using Pandas.

    Steps:
      - 'agg'   : aggregate to a chosen frequency (default hourly).
      - 'diff'  : replace processed features with smoothed differences (increments).
      - 'log'   : apply log1p scaling to the differences (compress heavy tails).
      - 'ffill' : forward-fill missing values (optionally treat zeros as missing),
                  with configurable fallback for leading/all-missing segments.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 columns: List[str],
                 partition_col: Union[str, List[str]] = "sn",
                 time_col: str = "time"):
        self.df = df.copy()
        self.columns = list(columns)
        self.partition_cols = [partition_col] if isinstance(partition_col, str) else list(partition_col)
        self.time_col = time_col

        # identify carry-through columns
        key_cols = set(self.partition_cols + [self.time_col])
        self.other_cols = [c for c in self.df.columns if c not in self.columns and c not in key_cols]

        self.df_transformed = None
        self._done = set()

    # ------------------------------
    # Aggregation
    # ------------------------------
    def aggregate_by_freq(self, freq: str = "h"):
        """
        Aggregate feature columns to a given frequency (default: hourly).
        Carry-through (non-feature) columns with first(non-null).
        """
        grouped = self.df.groupby(self.partition_cols + [pd.Grouper(key=self.time_col, freq=freq)])
        agg_dict = {c: "mean" for c in self.columns}
        for c in self.other_cols:
            agg_dict[c] = "first"
        self.df_transformed = grouped.agg(agg_dict).reset_index()
        self._done.add("agg")

    # ------------------------------
    # Differences
    # ------------------------------
    def compute_differences(self):
        """Compute smoothed differences (increments) for feature columns."""
        if "agg" not in self._done:
            self.aggregate_by_freq()

        df_out = self.df_transformed.copy()

        for c in self.columns:
            # raw and previous diffs per partition
            rawdiff = df_out.groupby(self.partition_cols)[c].diff()
            prevdiff = rawdiff.groupby(df_out[self.partition_cols].apply(tuple, axis=1)).shift(1) \
                       if len(self.partition_cols) > 1 else rawdiff.groupby(df_out[self.partition_cols[0]]).shift(1)

            # condition: dropouts (current < previous) -> use previous increment; else raw diff
            prev_val = df_out.groupby(self.partition_cols)[c].shift(1)
            cond = df_out[c] < prev_val
            series = np.where(cond, np.where(pd.isna(prevdiff), 0, prevdiff), rawdiff)

            # smooth negatives by carrying forward last positive increment per partition
            s = pd.Series(series, index=df_out.index)
            def _smooth(group):
                g = group.copy()
                # mask negatives then forward-fill within partition
                return g.mask(g < 0).ffill()
            if len(self.partition_cols) == 1:
                s = s.groupby(df_out[self.partition_cols[0]]).apply(_smooth).reset_index(level=0, drop=True)
            else:
                s = s.groupby(df_out[self.partition_cols].apply(tuple, axis=1)).apply(_smooth).reset_index(level=0, drop=True)

            df_out[c] = s.round(2)

        df_out = df_out.dropna(subset=self.columns)
        self.df_transformed = df_out
        self._done.add("diff")

    # ------------------------------
    # Log transform
    # ------------------------------
    def apply_log_scaling(self):
        """Apply log1p scaling to differences (clip negatives to 0 first)."""
        if "diff" not in self._done:
            self.compute_differences()

        for c in self.columns:
            self.df_transformed[c] = np.log1p(self.df_transformed[c].clip(lower=0)).round(2)

        self._done.add("log")

    # ------------------------------
    # Forward fill (flexible)
    # ------------------------------
    def forward_fill(self, treat_zero_as_missing: bool = False,
                     fallback: Union[str, float, None] = "group_mean"):
        """
        Forward-fill missing values per partition, with options:
          - treat_zero_as_missing=False: fill only NaNs, keep zeros as-is
          - treat_zero_as_missing=True : convert zeros to NaN first (so fill zeros & NaNs)
        Fallback for leading/all-missing segments:
          - 'group_mean' (default): fill remaining NaNs with the partition mean
          - 'global_mean'         : fill remaining NaNs with overall mean
          - 'zero'                : fill remaining NaNs with 0
          - None                  : leave remaining NaNs as-is
          - float                 : fill remaining NaNs with this constant
        """
        if self.df_transformed is None:
            self.aggregate_by_freq()

        df_out = self.df_transformed.copy()

        for c in self.columns:
            s = df_out[c].copy()
            if treat_zero_as_missing:
                s = s.replace(0, np.nan)

            # groupwise forward-fill
            if len(self.partition_cols) == 1:
                s = s.groupby(df_out[self.partition_cols[0]]).ffill()
                part_idx = df_out[self.partition_cols[0]]
            else:
                key = df_out[self.partition_cols].apply(tuple, axis=1)
                s = s.groupby(key).ffill()
                part_idx = key

            # fallback for any remaining NaNs
            if isinstance(fallback, (int, float)):
                s = s.fillna(float(fallback))
            elif fallback == "group_mean":
                if len(self.partition_cols) == 1:
                    s = s.fillna(s.groupby(part_idx).transform("mean"))
                else:
                    s = s.fillna(s.groupby(part_idx).transform("mean"))
            elif fallback == "global_mean":
                s = s.fillna(s.mean())
            elif fallback == "zero":
                s = s.fillna(0)
            elif fallback is None:
                pass  # leave NaNs

            df_out[c] = s

        self.df_transformed = df_out
        self._done.add("ffill")

    # ------------------------------
    # Pipeline runner
    # ------------------------------
    def execute_pipeline(self, steps=("agg", "diff", "log", "ffill"), **ffill_kwargs):
        """
        Run selected transformation steps in sequence.
        Pass kwargs to forward_fill via ffill_kwargs (e.g., treat_zero_as_missing=True).
        """
        for step in steps:
            if step == "agg" and "agg" not in self._done:
                self.aggregate_by_freq()
            elif step == "diff" and "diff" not in self._done:
                self.compute_differences()
            elif step == "log" and "log" not in self._done:
                self.apply_log_scaling()
            elif step == "ffill" and "ffill" not in self._done:
                self.forward_fill(**ffill_kwargs)
        return self



def split_time_series(
    df: pd.DataFrame,
    window_size: int,
    step_size: int,
    entity_col: str = "entity",
    time_col: str = "time",
    subseries_col: str = "subseries_id",
    time_fmt: str = "%Y%m%d%H%M%S",
    id_format: str = "time"  # options: "time" → entity_starttime, "index" → entity-index
) -> pd.DataFrame:
    """
    Split each entity's time series into overlapping subseries.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing multiple time series.
    window_size : int
        Number of rows in each subseries (window length).
    step_size : int
        Number of rows to shift between windows.
    entity_col : str
        Column identifying each entity (e.g., customer/device).
    time_col : str
        Timestamp column (must be sortable).
    subseries_col : str
        Name for the generated subseries identifier column.
    time_fmt : str
        Datetime format for labeling subseries start times (if id_format="time").
    id_format : str
        How to name subseries IDs:
        - "time": {entity}_{start_time} (default)
        - "index": {entity}-{window_index}

    Returns
    -------
    pd.DataFrame
        Original rows annotated with a new `subseries_col` giving subseries membership.
    """
    assert window_size > 0 and step_size > 0, "window_size and step_size must be positive"

    # Ensure time is datetime
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])

    results = []

    # Process each entity independently
    for entity, group in df.groupby(entity_col):
        group = group.sort_values(time_col).reset_index(drop=True)
        n = len(group)
        start_indices = range(0, n - window_size + 1, step_size)

        for idx, start in enumerate(start_indices):
            end = start + window_size
            sub_df = group.iloc[start:end].copy()

            if id_format == "time":
                start_time = sub_df[time_col].iloc[0].strftime(time_fmt)
                sub_id = f"{entity}_{start_time}"
            elif id_format == "index":
                sub_id = f"{entity}-{idx}"
            else:
                raise ValueError("id_format must be 'time' or 'index'")

            sub_df[subseries_col] = sub_id
            results.append(sub_df)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=df.columns.tolist() + [subseries_col])


