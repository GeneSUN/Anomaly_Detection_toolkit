import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class KMeansOutlierDetector:
    def __init__(self, df, features, n_clusters=2, scale=True,
                 filter_percentile=None, threshold_percentile=95,
                 time_col="time", random_state=42):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Input data containing both time and feature columns.
        features : list of str
            List of column names to use for clustering and outlier detection.
        time_col : str
            Column name representing time (used in plot_line).
        """
        self.df = df.copy()
        self.features = features
        self.time_col = time_col
        self.n_clusters = n_clusters
        self.scale = scale
        self.random_state = random_state
        self.filter_percentile = filter_percentile
        self.threshold_percentile = threshold_percentile

        self.scaler = StandardScaler() if scale else None
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.fitted = False

        self.df_clean = self.preprocess(self.df)

    def preprocess(self, df):
        X = df[self.features].values
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

    def fit(self):
        X = self.df_clean[self.features].values
        self.kmeans.fit(X)
        self.df_clean["cluster"] = self.kmeans.labels_
        self.centers = self.kmeans.cluster_centers_

        distances = np.linalg.norm(X - self.centers[self.df_clean["cluster"]], axis=1)
        self.df_clean["outlier_score"] = distances
        threshold = np.percentile(distances, self.threshold_percentile)
        self.df_clean["is_outlier"] = distances >= threshold

        self.fitted = True
        return

    def plot(self):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before plotting.")
        if len(self.features) != 2:
            raise ValueError("plot() only supports 2D feature space. Use plot_pca() for more than 2.")

        distances = self.df_clean["outlier_score"].values
        is_outlier = self.df_clean["is_outlier"].values

        # Get scaled feature matrix
        X_scaled = self.df_clean[self.features].values

        # Inverse transform to get original feature values
        if self.scale:
            X = self.scaler.inverse_transform(X_scaled)
            centers = self.scaler.inverse_transform(self.centers)
        else:
            X = X_scaled
            centers = self.centers

        # Shared color scale
        vmin = distances.min()
        vmax = distances.max()

        fig, ax = plt.subplots(figsize=(8, 6))

        # Inliers (no outline)
        inlier_plot = ax.scatter(
            X[~is_outlier][:, 0], X[~is_outlier][:, 1],
            c=distances[~is_outlier],
            cmap='coolwarm',
            s=50,
            edgecolors='none',
            vmin=vmin,
            vmax=vmax,
            label='Inliers'
        )

        # Outliers (black outline)
        ax.scatter(
            X[is_outlier][:, 0], X[is_outlier][:, 1],
            c=distances[is_outlier],
            cmap='coolwarm',
            s=50,
            edgecolors='black',
            linewidths=1,
            vmin=vmin,
            vmax=vmax,
            label='Outliers'
        )

        # Cluster centers
        ax.scatter(
            centers[:, 0], centers[:, 1],
            c='black', marker='x', s=100, label='Centers'
        )

        ax.set_title("KMeans Outlier Detection (2D, Original Scale)")
        ax.set_xlabel(self.features[0])
        ax.set_ylabel(self.features[1])

        cb = plt.colorbar(inlier_plot, ax=ax)
        cb.set_label("Outlier Score (Distance to Center)")

        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_pca(self):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before plotting.")
        if len(self.features) <= 2:
            raise ValueError("Use plot() instead for 2D input.")

        X = self.df_clean[self.features].values
        distances = self.df_clean["outlier_score"].values
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        centers_pca = pca.transform(self.centers)

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=distances, cmap='coolwarm', edgecolor='k')
        ax.scatter(
            X_pca[self.df_clean["is_outlier"]][:, 0],
            X_pca[self.df_clean["is_outlier"]][:, 1],
            c='red', edgecolor='k', label='Outliers'
        )
        ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='x', s=100, label='Centers')
        ax.set_title("KMeans Outlier Detection (PCA Reduced)")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Outlier Score")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_line(self, feature=None):
        """
        Plot a selected feature over time, highlighting detected outliers.

        Parameters
        ----------
        feature : str or None
            Feature to plot against time. Default is the first in self.features.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before plotting.")
        if feature is None:
            feature = self.features[0]
        if self.time_col not in self.df.columns:
            raise ValueError(f"time_col '{self.time_col}' not found in input dataframe.")
        if feature not in self.df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataframe.")

        merged_df = self.df[[self.time_col, feature]].merge(
            self.df_clean[[self.time_col, "is_outlier"]],
            on=self.time_col,
            how="left"
        ).sort_values(self.time_col)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(merged_df[self.time_col], merged_df[feature], label=feature, color='blue')
        ax.scatter(
            merged_df[merged_df["is_outlier"]][self.time_col],
            merged_df[merged_df["is_outlier"]][feature],
            color='red', label='Outlier', zorder=5
        )
        ax.set_xlabel(self.time_col)
        ax.set_ylabel(feature)
        ax.set_title(f"{feature} over Time with Outliers")
        ax.legend()
        plt.tight_layout()
        plt.show()
