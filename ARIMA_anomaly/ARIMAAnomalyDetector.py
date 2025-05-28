
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class ARIMAAnomalyDetector:
    def __init__(self, df, time_col, feature, season_length=1, confidence_level=99, freq='h'):
        self.df = df
        self.time_col = time_col
        self.feature = feature
        self.season_length = season_length
        self.confidence_level = confidence_level
        self.freq = freq
        self.model = StatsForecast(models=[AutoARIMA(season_length=season_length)], freq=freq, n_jobs=-1)
        self.forecast_df = None
        self.insample_forecast = None

    def prepare_data(self, df = None):
        if df == None:
            df = self.df
        df_arima = df[[self.time_col, self.feature]].copy()
        df_arima = df_arima.rename(columns={self.time_col: "ds", self.feature: "y"})
        df_arima["unique_id"] = "series_1"
        self.df_arima = df_arima
        return None

    def fit_forecast(self, df_arima = None, horizon=24):
        if df_arima == None:
            df_arima = self.df_arima
        self.forecast_df = self.model.forecast(df=df_arima, h=horizon, level=[self.confidence_level],fitted=True).reset_index()
        self.insample_forecast = self.model.forecast_fitted_values().reset_index()



    def detect_anomalies(self, insample_forecast = None):
        if insample_forecast == None:
            insample_forecast = self.insample_forecast


        insample_forecast['anomaly'] = (
                                (insample_forecast['y'] < insample_forecast[f'AutoARIMA-lo-{self.confidence_level}']) |
                                (insample_forecast['y'] > insample_forecast[f'AutoARIMA-hi-{self.confidence_level}'])
                            )
        self.insample_forecast = insample_forecast


    def plot_anomalies(self, insample_forecast = None, date_filter=None, confidence_level=None):
        if insample_forecast == None:
            insample_forecast = self.insample_forecast
        if confidence_level == None:
            confidence_level = self.confidence_level

        if date_filter is not None:
            start_date = pd.to_datetime(date_filter[0])
            end_date = pd.to_datetime(date_filter[1])
            insample_forecast = insample_forecast[(insample_forecast["ds"] >= start_date) & (insample_forecast["ds"] <= end_date)]


        plt.figure(figsize=(25, 5))
        plt.plot(insample_forecast['ds'], insample_forecast['y'], label='Actual')
        plt.plot(insample_forecast['ds'], insample_forecast['AutoARIMA'], label='Forecast')
        plt.fill_between(insample_forecast['ds'], insample_forecast[f'AutoARIMA-lo-{confidence_level}'], insample_forecast[f'AutoARIMA-hi-{confidence_level}'], color='gray', alpha=0.2, label=f'{confidence_level}% Prediction Interval')
        anomalies = insample_forecast[insample_forecast['anomaly']]
        plt.scatter(anomalies['ds'], anomalies['y'], color='red', label='Anomalies')
        plt.legend()
        plt.show()

detector = ARIMAAnomalyDetector(df = df_cap_hour_pd, time_col = "hour" ,feature="avg_4gsnr")
detector.prepare_data()
detector.fit_forecast()
detector.detect_anomalies()
detector.plot_anomalies()