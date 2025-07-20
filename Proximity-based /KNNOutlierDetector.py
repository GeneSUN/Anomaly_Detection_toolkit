import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class KNNOutlierDetector:
    def __init__(self, X, k=5, method='harmonic', scale=True,
                 filter_percentile=None, threshold_percentile=95):
        """
        Parameters:
            X : array-like, dataset
            k : int, number of nearest neighbors
            method : str, 'kth', 'average', or 'harmonic'
            scale : bool, whether to standard scale the data
            filter_percentile : float in (0,100), optional, remove top/bottom percentile during preprocessing
            threshold_percentile : float in (0,100), percentile for defining outliers based on score
        """
        self.X_raw = X.copy()
        self.k = k
        self.method = method
        self.scale = scale
        self.filter_percentile = filter_percentile
        self.threshold_percentile = threshold_percentile

        self.scaler = StandardScaler() if scale else None
        self.fitted = False

        self.X = self.preprocess(self.X_raw)

    def preprocess(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.filter_percentile is not None:
            center = np.mean(X, axis=0)
            dists = np.linalg.norm(X - center, axis=1)
            lower = np.percentile(dists, self.filter_percentile)
            upper = np.percentile(dists, 100 - self.filter_percentile)
            mask = (dists >= lower) & (dists <= upper)
            X = X[mask]

        if self.scale:
            X = self.scaler.fit_transform(X)

        return X

    def fit(self):
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(self.X)
        distances, _ = nn.kneighbors(self.X)
        dists = distances[:, 1:]  # exclude self

        if self.method == 'kth':
            self.scores = dists[:, -1]
        elif self.method == 'average':
            self.scores = dists.mean(axis=1)
        elif self.method == 'harmonic':
            self.scores = self.k / np.sum(1.0 / dists, axis=1)
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from 'kth', 'average', 'harmonic'.")

        threshold_value = np.percentile(self.scores, self.threshold_percentile)
        self.is_outlier = self.scores >= threshold_value
        self.fitted = True
        return self

    def plot(self):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before plotting.")

        fig, ax = plt.subplots(figsize=(8, 6))

        sc = ax.scatter(self.X[:, 0], self.X[:, 1], c=self.scores, cmap='coolwarm', edgecolor='k')
        ax.scatter(self.X[self.is_outlier][:, 0], self.X[self.is_outlier][:, 1],
                   c='red', edgecolor='k', label='Outliers')

        ax.set_title(f"{self.method.title()} k-NN Outlier Detection")
        ax.set_xlabel("Feature X")
        ax.set_ylabel("Feature Y")
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Outlier Score (Distance-Based)")
        ax.legend()
        plt.tight_layout()
        plt.show()

# Test example
np.random.seed(42)
X_normal = np.random.randn(100, 2) * 0.75 + np.array([2, 2])
X_outliers = np.random.uniform(low=-2, high=6, size=(5, 2))
X_all = np.vstack([X_normal, X_outliers])

knn_detector = KNNOutlierDetector(X_all, k=5, method='harmonic', filter_percentile=1, threshold_percentile=95)
knn_detector.fit()
knn_detector.plot()
