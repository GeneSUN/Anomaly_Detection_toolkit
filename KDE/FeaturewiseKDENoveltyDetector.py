import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

class FeaturewiseKDENoveltyDetector:
    def __init__(self, df, feature_col="avg_4gsnr", time_col="hour", bandwidth=0.5,
                 train_idx=None, new_idx=None, train_percentile=100,
                 anomaly_direction="low"):
        """
        Parameters:
            df (pd.DataFrame): Input data.
            feature_col (str): Column containing values to evaluate.
            time_col (str): Time column for plotting.
            bandwidth (float): Bandwidth for KDE.
            train_idx (slice): Slice for training data.
            new_idx (slice): Slice for new (test) data.
            train_percentile (float): Percentile for filtering out high-end outliers in training set.
            anomaly_direction (str): One of {"both", "high", "low"} to detect direction of anomaly.
        """
        self.df = df
        self.feature_col = feature_col
        self.time_col = time_col
        self.bandwidth = bandwidth
        self.train_idx = train_idx
        self.new_idx = new_idx
        self.train_percentile = train_percentile
        self.anomaly_direction = anomaly_direction
        self.kde = None
        self.threshold = None

    def _filter_train_df(self, train_df):
        if self.train_percentile < 100:
            upper = np.percentile(train_df[self.feature_col], self.train_percentile)
            train_df = train_df[train_df[self.feature_col] <= upper]
        return train_df

    def fit(self):
        # Slice training and new data
        train_df = self.df.iloc[self.train_idx] if self.train_idx is not None else self.df.iloc[:-1]
        train_df = self._filter_train_df(train_df)
        new_df = self.df.iloc[self.new_idx] if self.new_idx is not None else self.df.iloc[-1:]

        # Fit KDE on training data
        X_train = train_df[self.feature_col].values.reshape(-1, 1)
        X_new = new_df[self.feature_col].values.reshape(-1, 1)

        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.kde.fit(X_train)

        # Compute densities
        dens_train = np.exp(self.kde.score_samples(X_train))
        self.threshold = np.quantile(dens_train, 0.01)

        dens_new = np.exp(self.kde.score_samples(X_new))
        outlier_mask_kde = dens_new < self.threshold

        # Directional anomaly logic based on percentiles
        new_values = new_df[self.feature_col].values
        lower_threshold = np.percentile(train_df[self.feature_col], 100 - self.train_percentile)
        upper_threshold = np.percentile(train_df[self.feature_col], self.train_percentile)

        if self.anomaly_direction == "low":
            direction_mask = new_values < lower_threshold
        elif self.anomaly_direction == "high":
            direction_mask = new_values > upper_threshold
        else:  # both
            direction_mask = (new_values < lower_threshold) | (new_values > upper_threshold)

        final_outlier_mask = outlier_mask_kde & direction_mask

        self.dens_new = dens_new
        self.outlier_mask = final_outlier_mask

        return {
            "new_densities": dens_new,
            "threshold (1% quantile)": self.threshold,
            "outlier_count": final_outlier_mask.sum(),
            "total_new_points": len(dens_new),
            "outlier_indices": list(np.where(final_outlier_mask)[0]),
        }

    def plot_line(self, sn_num=None):
        plt.figure(figsize=(10, 4))
        plt.plot(self.df[self.time_col], self.df[self.feature_col], marker='o', label=self.feature_col)
        if self.new_idx is not None:
            if isinstance(self.new_idx, int):
                idxs = [self.new_idx]
            elif isinstance(self.new_idx, slice):
                idxs = list(range(*self.new_idx.indices(len(self.df))))
            else:
                idxs = self.new_idx
            for idx in idxs:
                plt.axvline(self.df.iloc[idx][self.time_col], color='red', linestyle='-', alpha=0.1)
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

    def plot_kde(self):
        train_df = self.df.iloc[self.train_idx] if self.train_idx is not None else self.df.iloc[:-1]
        train_df = self._filter_train_df(train_df)
        new_df = self.df.iloc[self.new_idx] if self.new_idx is not None else self.df.iloc[-1:]

        X_train = train_df[self.feature_col].values.reshape(-1, 1)
        X_new = new_df[self.feature_col].values.reshape(-1, 1)

        x_vals = np.linspace(X_train.min() - 1, X_train.max() + 1, 1000).reshape(-1, 1)
        log_dens = self.kde.score_samples(x_vals)
        dens = np.exp(log_dens)

        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, dens, label="KDE Density Curve")
        plt.axhline(self.threshold, color='red', linestyle='-.', label=f"Threshold (1%)")

        label_added = False
        outlier_label_added = False
        for i, val in enumerate(X_new.flatten()):
            is_outlier = self.outlier_mask[i]
            color = 'green' if not is_outlier else 'orange'
            if is_outlier and not outlier_label_added:
                plt.axvline(val, color=color, linestyle='--', alpha=0.7, label="New Data (outlier)")
                outlier_label_added = True
            elif not is_outlier and not label_added:
                plt.axvline(val, color=color, linestyle='--', alpha=0.7, label="New Data")
                label_added = True
            else:
                plt.axvline(val, color=color, linestyle='--', alpha=0.7)
        plt.title(f"KDE Novelty Detection: {self.feature_col}")
        plt.xlabel(self.feature_col)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

"""
detector = FeaturewiseKDENoveltyDetector(
    df=your_df,
    feature_col="avg_5gsnr",
    time_col="hour",
    train_idx=slice(0, 1068),
    new_idx=slice(-26, None),
    train_percentile=95,
    anomaly_direction="both"  # can be "low", "high", or "both"
)
result = detector.fit()
"""