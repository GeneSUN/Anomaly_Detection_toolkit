import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

class FeaturewiseKDENoveltyDetector:
    def __init__(self, df, feature_col="avg_4gsnr", time_col="hour", bandwidth=0.5):
        """
        Parameters:
            df (pd.DataFrame): Input DataFrame with time and feature columns.
            feature_col (str): Name of the column containing the feature to analyze.
            time_col (str): Name of the column containing the timestamp.
            bandwidth (float): Bandwidth for KDE smoothing.

        """
        self.df = df
        self.feature_col = feature_col
        self.time_col = time_col
        self.bandwidth = bandwidth
        self.kde = None
        self.threshold = None

    def plot_line(self, sn_num=None):
        """Plot the time series line chart of the feature."""
        plt.figure(figsize=(10, 4))
        plt.plot(self.df[self.time_col], self.df[self.feature_col], marker='o', label=self.feature_col)
        plt.axvline(self.df.iloc[-1][self.time_col], color='red', linestyle='--', label="New Observation")
        title = f"Line Plot: {self.feature_col} Over Time"
        if sn_num:
            title += f" ({sn_num})"
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel(self.feature_col)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def fit(self):
        """Fit KDE on training data and detect if the latest point is a novelty."""
        train_df = self.df.iloc[:-1]
        new_point = self.df.iloc[-1]

        X_train = train_df[self.feature_col].values.reshape(-1, 1)
        X_test = np.array([[new_point[self.feature_col]]])

        # Fit KDE
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.kde.fit(X_train)

        # Compute training densities and threshold
        log_dens_train = self.kde.score_samples(X_train)
        dens_train = np.exp(log_dens_train)
        self.threshold  = np.quantile(dens_train, 0.01)  # Bottom 1% of known density
        """
        mu = dens_train.mean()
        sigma = dens_train.std()
        self.threshold = mu - self.threshold_k * sigma
        """
        # Score the new observation
        log_dens_new = self.kde.score_samples(X_test)
        self.dens_new = np.exp(log_dens_new)[0]
        self.prediction = "Novelty" if self.dens_new < self.threshold else "Normal"

        return {
            "new_density": self.dens_new,
            "threshold (mu - k * sigma)": self.threshold,
            "prediction": self.prediction
        }

    def plot_kde(self):
        """Plot the KDE curve and threshold against the new observation."""
        X_train = self.df.iloc[:-1][self.feature_col].values.reshape(-1, 1)
        X_test = np.array([[self.df.iloc[-1][self.feature_col]]])

        x_vals = np.linspace(X_train.min() - 1, X_train.max() + 1, 1000).reshape(-1, 1)
        log_dens = self.kde.score_samples(x_vals)
        dens = np.exp(log_dens)

        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, dens, label="KDE Density Curve")
        plt.axvline(X_test[0][0], color='green', linestyle='--', label="New Observation")
        plt.axhline(self.threshold, color='red', linestyle='-.', label=f"Threshold 1 %")
        plt.title(f"KDE Novelty Detection: {self.feature_col}")
        plt.xlabel(self.feature_col)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

