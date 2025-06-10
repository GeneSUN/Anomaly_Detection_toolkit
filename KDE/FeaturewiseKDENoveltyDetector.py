import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

class FeaturewiseKDENoveltyDetector:
    def __init__(self, df, feature_col="avg_4gsnr", time_col="hour", bandwidth=0.5,
                 train_idx=None, new_idx=None, train_percentile=100):
        """
        Parameters:
            ...
            train_percentile (float): Upper percentile of training data to include (e.g., 99 means drop top 1%).
        """
        self.df = df
        self.feature_col = feature_col
        self.time_col = time_col
        self.bandwidth = bandwidth
        self.train_idx = train_idx
        self.new_idx = new_idx
        self.train_percentile = train_percentile
        self.kde = None
        self.threshold = None

    def _filter_train_df(self, train_df):
        """Filter out upper outliers in the training set."""
        if self.train_percentile < 100:
            upper = np.percentile(train_df[self.feature_col], self.train_percentile)
            train_df = train_df[train_df[self.feature_col] <= upper]
        return train_df

    def plot_line(self, sn_num=None):
        # [unchanged code, see previous version]
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

    def fit(self):
        train_df = self.df.iloc[self.train_idx] if self.train_idx is not None else self.df.iloc[:-1]
        # --- Filter out upper outliers in train_df ---
        train_df = self._filter_train_df(train_df)
        new_df = self.df.iloc[self.new_idx] if self.new_idx is not None else self.df.iloc[-1:]

        X_train = train_df[self.feature_col].values.reshape(-1, 1)
        X_new = new_df[self.feature_col].values.reshape(-1, 1)

        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.kde.fit(X_train)

        log_dens_train = self.kde.score_samples(X_train)
        dens_train = np.exp(log_dens_train)
        self.threshold = np.quantile(dens_train, 0.01)

        log_dens_new = self.kde.score_samples(X_new)
        dens_new = np.exp(log_dens_new)
        outlier_mask = dens_new < self.threshold
        n_outliers = outlier_mask.sum()

        self.dens_new = dens_new
        self.outlier_mask = outlier_mask

        return {
            "new_densities": dens_new,
            "threshold (1% quantile)": self.threshold,
            "outlier_count": n_outliers,
            "total_new_points": len(dens_new),
            "outlier_indices": list(np.where(outlier_mask)[0]),
        }

    def plot_kde(self):
        train_df = self.df.iloc[self.train_idx] if self.train_idx is not None else self.df.iloc[:-1]
        train_df = self._filter_train_df(train_df)  # filter upper percentile
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