from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd
import matplotlib.pyplot as plt
# http://njbbvmaspd13:18080/next/#/notebook/2KW1NYKEF
class ARIMAAnomalyDetector:
    """
    Anomaly detection for univariate time series using Nixtla's AutoARIMA with prediction intervals.

    This class fits an AutoARIMA model, computes in-sample forecasts and prediction intervals,
    and flags anomalies as observations falling outside the specified confidence interval.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the time series data.
    time_col : str
        Name of the column containing timestamps.
    feature : str
        Name of the column containing the univariate feature to model.
    season_length : int, optional
        Seasonal periodicity of the time series (default is 1; use e.g. 24 for hourly data with daily seasonality).
    confidence_level : int, optional
        Confidence interval percentage for anomaly detection (default is 99 for 99%).
    freq : str, optional
        Pandas frequency string for the time series (default is 'h' for hourly).

    Example
    -------
    >>> import pandas as pd
    >>> df_cap_hour_pd = pd.DataFrame({
    ...     'sn': ['ABG23405602']*5,
    ...     'hour': pd.date_range('2025-03-27 19:00:00', periods=5, freq='H'),
    ...     'avg_4gsnr': [15.0, 13.0, 12.0, 13.0, 13.0]
    ... })
    >>> detector = ARIMAAnomalyDetector(df=df_cap_hour_pd, time_col='hour', feature='avg_4gsnr')
    >>> detector.run()
    >>> detector.plot_anomalies()
    """

    def __init__(self, df, time_col, feature, season_length=24, confidence_level=99, freq='h'):
        """
        Initialize the ARIMAAnomalyDetector.
        """
        self.df = df
        self.time_col = time_col
        self.feature = feature
        self.season_length = season_length
        self.confidence_level = confidence_level
        self.freq = freq
        self.model = StatsForecast(
            models=[AutoARIMA(season_length=season_length)],
            freq=freq,
            n_jobs=-1
        )
        self.df_arima = None
        self.forecast_df = None
        self.insample_forecast = None

    def prepare_data(self, df=None):
        """
        Prepare the input DataFrame in the format required for StatsForecast.

        Parameters
        ----------
        df : pd.DataFrame or None
            DataFrame to use (default: uses self.df)
        """
        if df is None:
            df = self.df
        df_arima = df[[self.time_col, self.feature]].copy()
        df_arima = df_arima.rename(columns={self.time_col: "ds", self.feature: "y"})
        df_arima["unique_id"] = "series_1"
        self.df_arima = df_arima

    def fit_forecast(self, df_arima=None, horizon=24):
        """
        Fit AutoARIMA model and get in-sample fitted values and prediction intervals.

        Parameters
        ----------
        df_arima : pd.DataFrame or None
            Formatted DataFrame for ARIMA (default: uses self.df_arima)
        horizon : int
            Number of steps to forecast ahead (default: 0, in-sample only)
        """
        if df_arima is None:
            df_arima = self.df_arima

        self.forecast_df = self.model.forecast(df=df_arima, h=horizon, level=[self.confidence_level],fitted=True).reset_index()
        self.insample_forecast = self.model.forecast_fitted_values().reset_index()



    def detect_anomalies(self, insample_forecast=None):
        """
        Flag anomalies as points outside the prediction interval.

        Parameters
        ----------
        insample_forecast : pd.DataFrame or None
            DataFrame with fitted values and intervals (default: uses self.insample_forecast)
        """
        if insample_forecast is None:
            insample_forecast = self.insample_forecast

        insample_forecast['anomaly'] = (
            (insample_forecast['y'] < insample_forecast[f'AutoARIMA-lo-{self.confidence_level}']) |
            (insample_forecast['y'] > insample_forecast[f'AutoARIMA-hi-{self.confidence_level}'])
        )
        self.insample_forecast = insample_forecast

    def plot_anomalies(self, insample_forecast=None, date_filter=None, confidence_level=None):
        """
        Plot the time series, forecast, prediction interval, and anomalies.

        Parameters
        ----------
        insample_forecast : pd.DataFrame or None
            DataFrame to plot (default: uses self.insample_forecast)
        date_filter : tuple or None
            (start_date, end_date) to filter plotted data (default: None, plot all)
        confidence_level : int or None
            Confidence level for prediction interval (default: uses self.confidence_level)
        """
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

        plt.figure(figsize=(16, 5))
        plt.plot(insample_forecast['ds'], insample_forecast['y'], label='Actual')
        plt.plot(insample_forecast['ds'], insample_forecast['AutoARIMA'], label='Forecast')
        plt.fill_between(
            insample_forecast['ds'],
            insample_forecast[f'AutoARIMA-lo-{confidence_level}'],
            insample_forecast[f'AutoARIMA-hi-{confidence_level}'],
            color='gray', alpha=0.2, label=f'{confidence_level}% Prediction Interval'
        )
        anomalies = insample_forecast[insample_forecast['anomaly']]
        plt.scatter(anomalies['ds'], anomalies['y'], color='red', label='Anomalies')
        plt.legend()
        plt.title("ARIMA-based Anomaly Detection")
        plt.xlabel("Time")
        plt.ylabel(self.feature)
        plt.show()
    
    def get_recent_anomaly_stats(self, num_recent_points=24):
        """
        Return a dictionary summarizing anomaly stats from the last n points.
    
        Parameters
        ----------
        num_recent_points : int
            Number of most recent points to analyze.
    
        Returns
        -------
        dict
            {
                "outlier_count": int,
                "total_new_points": int,
                "outlier_indices": list
            }
        """
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
        """
        Run the full anomaly detection pipeline: prepare data, fit model, and flag anomalies.
        """
        self.prepare_data()
        self.fit_forecast()
        self.detect_anomalies()