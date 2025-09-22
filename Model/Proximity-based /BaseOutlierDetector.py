import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

# ==============================================================
# Base Detector
# ==============================================================

class BaseOutlierDetector:
    def __init__(self, df, features, time_col="time", scale=True,
                 filter_percentile=None, threshold_percentile=99):
        self.df = df.copy()
        self.features = features
        self.time_col = time_col
        self.scale = scale
        self.filter_percentile = filter_percentile
        self.threshold_percentile = threshold_percentile

        self.scaler = StandardScaler() if scale else None
        self.fitted = False
        self.df_clean = self.preprocess(self.df)

    def preprocess(self, df):
        X = df[self.features].values

        # Optional filter extremes
        if self.filter_percentile is not None:
            center = np.mean(X, axis=0)
            dists = np.linalg.norm(X - center, axis=1)
            lower = np.percentile(dists, self.filter_percentile)
            upper = np.percentile(dists, 100 - self.filter_percentile)
            mask = (dists >= lower) & (dists <= upper)
            df = df.loc[mask]

        if self.scale:
            X = self.scaler.fit_transform(df[self.features].values)
            df[self.features] = X

        return df.reset_index(drop=True)

    def plot(self):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before plotting.")

        X = self.df_clean[self.features].values
        scores = self.df_clean["outlier_score"].values
        is_outlier = self.df_clean["is_outlier"].values

        # Choose raw vs PCA
        if len(self.features) == 2:
            X_plot = X
            title_suffix = "(2D Features)"
        else:
            print(f"[INFO] {len(self.features)} features detected. Using PCA projection to 2D.")
            X_plot = PCA(n_components=2).fit_transform(X)
            title_suffix = "(PCA Reduced)"

        vmin, vmax = scores.min(), scores.max()
        fig, ax = plt.subplots(figsize=(8, 6))

        # Inliers
        ax.scatter(X_plot[~is_outlier, 0], X_plot[~is_outlier, 1],
                   c=scores[~is_outlier], cmap="coolwarm", s=50,
                   edgecolors="none", vmin=vmin, vmax=vmax, label="Inliers")

        # Outliers
        ax.scatter(X_plot[is_outlier, 0], X_plot[is_outlier, 1],
                   c=scores[is_outlier], cmap="coolwarm", s=50,
                   edgecolors="black", linewidths=1, vmin=vmin, vmax=vmax, label="Outliers")

        # Colorbar + labels
        cb = plt.colorbar(ax.collections[0], ax=ax)
        cb.set_label("Outlier Score")
        ax.set_title(f"{self.__class__.__name__} Detection {title_suffix}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_line(self, feature=None):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before plotting.")
        if feature is None:
            feature = self.features[0]
        if self.time_col not in self.df.columns:
            raise ValueError(f"time_col '{self.time_col}' not found in input df.")
        if feature not in self.df.columns:
            raise ValueError(f"Feature '{feature}' not found in df.")

        merged_df = self.df[[self.time_col, feature]].merge(
            self.df_clean[[self.time_col, "is_outlier"]],
            on=self.time_col, how="left"
        ).sort_values(self.time_col)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(merged_df[self.time_col], merged_df[feature], color="blue", label=feature)
        ax.scatter(merged_df[merged_df["is_outlier"]][self.time_col],
                   merged_df[merged_df["is_outlier"]][feature],
                   color="red", label="Outlier", zorder=5)
        ax.set_xlabel(self.time_col)
        ax.set_ylabel(feature)
        ax.set_title(f"{feature} over Time with Outliers ({self.__class__.__name__})")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_radar(self, point_idx=None, agg="median", lower_q=25, upper_q=75):
        """
        Plot a radar chart comparing one outlier point against the majority distribution.

        Parameters
        ----------
        point_idx : int, optional
            Index of the point in df_clean to highlight. If None, the first outlier is used.
        agg : {'mean','median'}, default='median'
            Aggregation for majority profile.
        lower_q : float, default=25
            Lower percentile for variability band.
        upper_q : float, default=75
            Upper percentile for variability band.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before plotting.")

        features = self.features
        n_features = len(features)

        # Majority profile
        normal_df = self.df_clean[~self.df_clean["is_outlier"]][features]
        if agg == "mean":
            majority = normal_df.mean().values
        else:
            majority = normal_df.median().values

        # Variability band
        q_low = normal_df.quantile(lower_q / 100.0).values
        q_high = normal_df.quantile(upper_q / 100.0).values

        # Outlier point
        if point_idx is None:
            outlier_row = self.df_clean[self.df_clean["is_outlier"]].iloc[0]
        else:
            outlier_row = self.df_clean.iloc[point_idx]
        outlier = outlier_row[features].values

        # Radar chart setup
        angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]  # close the circle

        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

        # Majority profile
        majority_vals = majority.tolist() + majority[:1].tolist()
        ax.plot(angles, majority_vals, color="grey", linewidth=2, label=f"Majority ({agg})")
        ax.fill(angles, majority_vals, color="grey", alpha=0.2)

        # Variability band (percentiles)
        q_low_vals = q_low.tolist() + q_low[:1].tolist()
        q_high_vals = q_high.tolist() + q_high[:1].tolist()
        ax.fill_between(angles, q_low_vals, q_high_vals,
                        color="grey", alpha=0.1,
                        label=f"{lower_q}–{upper_q} percentile band")

        # Outlier
        outlier_vals = outlier.tolist() + outlier[:1].tolist()
        ax.plot(angles, outlier_vals, color="red", linewidth=2, label="Outlier")
        ax.fill(angles, outlier_vals, color="red", alpha=0.1)

        # Feature labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features)

        plt.title("Radar Chart: Outlier vs Majority Profile", pad=20)

        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.show()

    def fit(self):
        raise NotImplementedError