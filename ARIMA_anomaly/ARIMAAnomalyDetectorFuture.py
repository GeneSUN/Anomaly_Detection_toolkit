from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd
import matplotlib.pyplot as plt

class ARIMAAnomalyDetectorFuture:
    """
    Detect anomalies by forecasting *future* values outside prediction intervals.

    Parameters
    ----------
    df : pandas.DataFrame
    time_col : str
    feature : str
        Column name of the numeric series to model.
    season_length : int, default=24
        Seasonal period passed to `AutoARIMA` (e.g., 24 for hourly data with daily seasonality).
    confidence_level : int, default=99
    freq : str, default='h'
        Pandas/StatsForecast frequency alias (e.g., 'h' for hourly, 'D' for
        daily). Must match the cadence of `time_col`.
    anomaly_direction : {'both', 'upper', 'lower'}, default='both'
        Which side(s) of the interval to treat as anomalous:
        - 'both' : flag values below the lower bound or above the upper bound
        - 'upper': flag values strictly above the upper bound
        - 'lower': flag values strictly below the lower bound
    split_idx : int, default=24
        Number of trailing observations reserved for the test (forecast) horizon.
        The model is trained on all rows except the last `split_idx`, then
        forecasts `h=split_idx` steps ahead for comparison against the held-out
        actuals.

    Attributes
    ----------
    model : StatsForecast
        The StatsForecast object configured with an `AutoARIMA` model.
    train_df : pandas.DataFrame or None
        Prepared training data with columns ['unique_id', 'ds', 'y'].
    test_df : pandas.DataFrame or None
        Prepared test data (last `split_idx` rows) with columns
        ['unique_id', 'ds', 'y'].
    forecast_df : pandas.DataFrame or None
        Forecast output from StatsForecast, including point forecasts in
        'AutoARIMA' and interval columns named
        'AutoARIMA-lo-{confidence_level}' and 'AutoARIMA-hi-{confidence_level}'.
    result_df : pandas.DataFrame or None
        Merge of `forecast_df` and `test_df` with anomaly flags:
        - 'anomaly' (bool)
        - 'anomaly_type' in {'low', 'high', None}

    Methods
    -------
    prepare_data()
        Renames/standardizes columns to ['unique_id', 'ds', 'y'] and splits
        into train/test by `split_idx`.
    fit_forecast()
        Fits AutoARIMA on the training data and produces `h=split_idx`
        forecasts with prediction intervals at `confidence_level`.
    detect_anomalies()
        Compares actuals to forecast intervals and sets 'anomaly' and
        'anomaly_type' columns in `result_df`.
    plot_anomalies()
        Plots actuals, forecasts, prediction interval, and highlights detected
        anomalies.
    run()
        Convenience method: `prepare_data()` → `fit_forecast()` → `detect_anomalies()`.

    Examples
    --------
    >>> detector = ARIMAAnomalyDetectorFuture(
    ...     df=dataframe,
    ...     time_col="timestamp",
    ...     feature="throughput",
    ...     season_length=24,
    ...     confidence_level=95,
    ...     freq="h",
    ...     anomaly_direction="both",
    ...     split_idx=48,
    ... )
    >>> detector.run()
    >>> anomalies = detector.result_df[detector.result_df["anomaly"]]
    >>> detector.plot_anomalies()
    """

    def __init__(self, df, time_col, feature, season_length=24, confidence_level=99,
                 freq='h', anomaly_direction='both', split_idx=1, unique_id = "series_1"):
        self.df = df
        self.time_col = time_col
        self.feature = feature
        self.season_length = season_length
        self.confidence_level = confidence_level
        self.freq = freq
        self.anomaly_direction = anomaly_direction
        self.split_idx = split_idx
        self.model = StatsForecast(
            models=[AutoARIMA(season_length=season_length)],
            freq=freq,
            n_jobs=-1
        )
        self.unique_id = unique_id
        self.train_df = None
        self.test_df = None
        self.df_arima = None
        self.forecast_df = None

    def prepare_data(self):
        df_arima = self.df[[self.time_col, self.feature]].copy()
        df_arima = df_arima.rename(columns={self.time_col: "ds", self.feature: "y"})
        df_arima["unique_id"] = self.unique_id
        self.df_arima = df_arima
        # Split train and test
        self.train_df = df_arima[:-self.split_idx].copy()
        self.test_df = df_arima[-self.split_idx:].copy()

    def fit_forecast(self):
        self.forecast_df = self.model.forecast(
            df=self.train_df,
            h=self.split_idx,
            level=[self.confidence_level]
        ).reset_index()

    def detect_anomalies(self):
        # Merge forecast and actuals
        result = pd.merge(
            self.forecast_df.drop(columns='ds'),
            self.test_df,
            on=["unique_id"],
            how="left"
        )
        lo_col = f"AutoARIMA-lo-{self.confidence_level}"
        hi_col = f"AutoARIMA-hi-{self.confidence_level}"

        if self.anomaly_direction == "lower":
            result["anomaly"] = result["y"] < result[lo_col]
            result["anomaly_type"] = result["anomaly"].apply(lambda x: "low" if x else None)
        elif self.anomaly_direction == "upper":
            result["anomaly"] = result["y"] > result[hi_col]
            result["anomaly_type"] = result["anomaly"].apply(lambda x: "high" if x else None)
        else:
            is_low = result["y"] < result[lo_col]
            is_high = result["y"] > result[hi_col]
            result["anomaly"] = is_low | is_high
            result["anomaly_type"] = is_low.map({True: "low"}).combine_first(is_high.map({True: "high"}))

        self.result_df = result
        
    def plot_anomalies(self):
        """
        Plot:
        - Train actuals (line) from self.train_df
        - Forecast horizon from self.result_df:
            * Forecast (scatter)
            * Actual (scatter, different color)
            * CI bounds (scatter + connected lines)
            * Shaded CI band (handles single-point horizon)
        """
        if self.train_df is None or getattr(self, 'result_df', None) is None:
            raise RuntimeError("Call run() before plotting.")

        lo_col = f"AutoARIMA-lo-{self.confidence_level}"
        hi_col = f"AutoARIMA-hi-{self.confidence_level}"

        # Train (history)
        train = self.train_df[['ds', 'y']].sort_values('ds')

        # Horizon with forecast + CI (already merged in detect_anomalies)
        need = ['ds', 'y', 'AutoARIMA', lo_col, hi_col]
        missing = [c for c in need if c not in self.result_df.columns]
        if missing:
            raise ValueError(f"Missing columns in result_df: {missing}")

        res = (
            self.result_df[need]
            .dropna(subset=['ds'])
            .drop_duplicates('ds')
            .sort_values('ds')
        )

        plt.figure(figsize=(16, 5))

        # 1) Train actuals (line)
        plt.plot(self.df_arima['ds'], self.df_arima['y'], label='Train Actual', linewidth=1.2, alpha=0.85,  color='blue')

        dark_color = 'darkorange'
        light_color = 'orange'  # A lighter shade of blue

        # 2) Forecast vs Actual on horizon (scatter, different colors)
        plt.scatter(res['ds'], res['AutoARIMA'], s=30, label='Forecast',  color=dark_color)
        plt.scatter(res['ds'], res['y'], s=30, label='Actual (Horizon)',  color='blue')

        # 3) Confidence interval bounds (scatter + connected lines), plus shaded band
        if len(res['ds']) > 1:
            plt.plot(res['ds'], res[lo_col], linewidth=1.0, label=f'CI Low ({self.confidence_level}%)',  color=light_color)
            plt.plot(res['ds'], res[hi_col], linewidth=1.0, label=f'CI High ({self.confidence_level}%)',  color=light_color)
            plt.scatter(res['ds'], res[lo_col], s=14,  color=light_color)
            plt.scatter(res['ds'], res[hi_col], s=14,  color=light_color)
            plt.fill_between(res['ds'], res[lo_col], res[hi_col], alpha=1, label=f'{self.confidence_level}% CI')

        else:
            # Plot the two scatter points
            plt.scatter(res['ds'], res[lo_col], s=14, label=f'CI Low ({self.confidence_level}%)',  color=light_color)
            plt.scatter(res['ds'], res[hi_col], s=14, label=f'CI High ({self.confidence_level}%)',  color=light_color)

            # Connect the two points with a transparent line
            plt.plot([res['ds'].iloc[0], res['ds'].iloc[0]], 
                    [res[lo_col].iloc[0], res[hi_col].iloc[0]], 
                    color=light_color, 
                    alpha=0.3, 
                    linestyle='--')


        # 4) Optional: highlight anomalies (horizon)
        if 'anomaly' in self.result_df.columns and self.result_df['anomaly'].fillna(False).any():
            a = self.result_df[self.result_df['anomaly'].fillna(False)]
            plt.scatter(a['ds'], a['y'], s=60, facecolors='none', edgecolors='red', linewidths=1.2, label='Anomaly')

        plt.title(f"ARIMA Anomaly Detection (Train & Horizon, {self.anomaly_direction})")
        plt.xlabel("Time")
        plt.ylabel(self.feature)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


    def run(self):
        self.prepare_data()
        self.fit_forecast()
        self.detect_anomalies()
