from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd
import matplotlib.pyplot as plt
# http://njbbvmaspd13:18080/next/#/notebook/2KW1NYKEF
class ARIMAAnomalyDetector:
    """
    Anomaly detection for univariate time series using Nixtla's AutoARIMA with prediction intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data.
    time_col : str
        Timestamp column.
    feature : str
        Feature to analyze.
    season_length : int
        Periodicity (e.g., 24 for daily pattern in hourly data).
    confidence_level : int
        Confidence interval percentage (default: 99).
    freq : str
        Time series frequency string (e.g., 'h').
    anomaly_direction : str
        Which anomalies to detect: 'both', 'upper', or 'lower'.
    """

    def __init__(self, df, time_col, feature, season_length=24, confidence_level=99,
                 freq='h', anomaly_direction='both'):
        self.df = df
        self.time_col = time_col
        self.feature = feature
        self.season_length = season_length
        self.confidence_level = confidence_level
        self.freq = freq
        self.anomaly_direction = anomaly_direction  # NEW
        self.model = StatsForecast(
            models=[AutoARIMA(season_length=season_length)],
            freq=freq,
            n_jobs=-1
        )
        self.df_arima = None
        self.forecast_df = None
        self.insample_forecast = None

    def prepare_data(self, df=None):
        if df is None:
            df = self.df
        df_arima = df[[self.time_col, self.feature]].copy()
        df_arima = df_arima.rename(columns={self.time_col: "ds", self.feature: "y"})
        df_arima["unique_id"] = "series_1"
        self.df_arima = df_arima

    def fit_forecast(self, df_arima=None, horizon=24):
        if df_arima is None:
            df_arima = self.df_arima
        self.forecast_df = self.model.forecast(
            df=df_arima, h=horizon, level=[self.confidence_level], fitted=True
        ).reset_index()
        self.insample_forecast = self.model.forecast_fitted_values().reset_index()

    def detect_anomalies(self, insample_forecast=None):
        if insample_forecast is None:
            insample_forecast = self.insample_forecast

        lo_col = f'AutoARIMA-lo-{self.confidence_level}'
        hi_col = f'AutoARIMA-hi-{self.confidence_level}'

        # NEW: Add anomaly type column and flag based on direction
        if self.anomaly_direction == 'lower':
            insample_forecast['anomaly'] = insample_forecast['y'] < insample_forecast[lo_col]
            insample_forecast['anomaly_type'] = insample_forecast['anomaly'].apply(lambda x: 'low' if x else None)
        elif self.anomaly_direction == 'upper':
            insample_forecast['anomaly'] = insample_forecast['y'] > insample_forecast[hi_col]
            insample_forecast['anomaly_type'] = insample_forecast['anomaly'].apply(lambda x: 'high' if x else None)
        else:  # both
            is_low = insample_forecast['y'] < insample_forecast[lo_col]
            is_high = insample_forecast['y'] > insample_forecast[hi_col]
            insample_forecast['anomaly'] = is_low | is_high
            insample_forecast['anomaly_type'] = is_low.map({True: 'low'}).combine_first(is_high.map({True: 'high'}))

        self.insample_forecast = insample_forecast

    def plot_anomalies(self, insample_forecast=None, date_filter=None, confidence_level=None):
        if insample_forecast is None:
            insample_forecast = self.insample_forecast
        if confidence_level is None:
            confidence_level = self.confidence_level

        if date_filter is not None:
            start_date = pd.to_datetime(date_filter[0])
            end_date = pd.to_datetime(date_filter[1])
            insample_forecast = insample_forecast[
                (insample_forecast["ds"] >= start_date) & (insample_forecast["ds"] <= end_date)
            ]

        lo_col = f'AutoARIMA-lo-{confidence_level}'
        hi_col = f'AutoARIMA-hi-{confidence_level}'

        plt.figure(figsize=(16, 5))
        plt.plot(insample_forecast['ds'], insample_forecast['y'], label='Actual')
        plt.plot(insample_forecast['ds'], insample_forecast['AutoARIMA'], label='Forecast')
        plt.fill_between(insample_forecast['ds'], insample_forecast[lo_col], insample_forecast[hi_col],
                         color='gray', alpha=0.2, label=f'{confidence_level}% Prediction Interval')

        # Color code anomalies
        if 'anomaly_type' in insample_forecast.columns:
            low_anomalies = insample_forecast[insample_forecast['anomaly_type'] == 'low']
            high_anomalies = insample_forecast[insample_forecast['anomaly_type'] == 'high']
            plt.scatter(low_anomalies['ds'], low_anomalies['y'], color='blue', label='Low Anomalies')
            plt.scatter(high_anomalies['ds'], high_anomalies['y'], color='red', label='High Anomalies')
        else:
            anomalies = insample_forecast[insample_forecast['anomaly']]
            plt.scatter(anomalies['ds'], anomalies['y'], color='red', label='Anomalies')

        plt.legend()
        plt.title(f"ARIMA-based Anomaly Detection ({self.anomaly_direction})")
        plt.xlabel("Time")
        plt.ylabel(self.feature)
        plt.show()
    def get_recent_anomaly_stats(self, num_recent_points=24):
        if self.insample_forecast is None or 'anomaly' not in self.insample_forecast.columns:
            raise ValueError("Anomaly detection has not been run yet. Please call run() first.")
        recent_data = self.insample_forecast[-num_recent_points:].copy()
        outliers = recent_data[recent_data['anomaly']]
        return {
            "outlier_count": outliers.shape[0],
            "total_new_points": num_recent_points,
            "outlier_indices": outliers.index.tolist()
        }

    def run(self):
        self.prepare_data()
        self.fit_forecast()
        self.detect_anomalies()