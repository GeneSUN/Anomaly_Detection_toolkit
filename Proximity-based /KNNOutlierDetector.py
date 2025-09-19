import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from BaseOutlierDetector import BaseOutlierDetector

# ==============================================================
# kNN Detector
# ==============================================================

class KNNOutlierDetector(BaseOutlierDetector):
    def __init__(self, df, features, time_col="time", k=5, method="kth",
                 scale=True, filter_percentile=None, threshold_percentile=99):
        super().__init__(df, features, time_col, scale, filter_percentile, threshold_percentile)
        self.k = k
        self.method = method

    def fit(self):
        X = self.df_clean[self.features].values
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        distances, _ = nn.kneighbors(X)
        dists = distances[:, 1:]  # exclude self

        if self.method == "kth":
            scores = dists[:, -1]
        elif self.method == "average":
            scores = dists.mean(axis=1)
        elif self.method == "harmonic":
            scores = self.k / np.sum(1.0 / dists, axis=1)
        else:
            raise ValueError(f"Unknown method {self.method}")

        self.scores = self.df_clean["outlier_score"] = scores
        threshold = np.percentile(scores, self.threshold_percentile)
        self.is_outlier = self.df_clean["is_outlier"] = scores >= threshold
        self.fitted = True
        return self

    def plot(self):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before plotting.")

        # Apply PCA only if dimensionality > 2
        if self.X.shape[1] > 2:
            X_plot = PCA(n_components=2).fit_transform(self.X)
        else:
            X_plot = self.X

        fig, ax = plt.subplots(figsize=(8, 6))

        sc = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=self.scores, cmap='coolwarm', edgecolor='k')
        ax.scatter(X_plot[self.is_outlier][:, 0], X_plot[self.is_outlier][:, 1],
                   c='red', edgecolor='k', label='Outliers')

        ax.set_title(f"{self.method.title()} k-NN Outlier Detection")
        ax.set_xlabel("Component 1" if self.X.shape[1] > 2 else "Feature X")
        ax.set_ylabel("Component 2" if self.X.shape[1] > 2 else "Feature Y")
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Outlier Score (Distance-Based)")
        ax.legend()
        plt.tight_layout()
        plt.show()
