import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

class KMeansOutlierDetector:
    def __init__(self, X, n_clusters=2, percentile_threshold=95, scale=True, random_state=42):
        self.X_raw = X.copy()
        self.n_clusters = n_clusters
        self.percentile_threshold = percentile_threshold
        self.scale = scale
        self.random_state = random_state
        self.scaler = StandardScaler() if scale else None
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.fitted = False
        self.X = self.preprocess(self.X_raw)

    def preprocess(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.scale:
            X = self.scaler.fit_transform(X)
        return X

    def fit(self):
        self.kmeans.fit(self.X)
        self.labels = self.kmeans.labels_
        self.centers = self.kmeans.cluster_centers_
        self.distances = np.linalg.norm(self.X - self.centers[self.labels], axis=1)
        threshold_value = np.percentile(self.distances, self.percentile_threshold)
        self.is_outlier = self.distances >= threshold_value
        self.fitted = True
        return self

    def plot(self):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before plotting.")

        fig, ax = plt.subplots(figsize=(8, 6))

        sc = ax.scatter(self.X[:, 0], self.X[:, 1], c=self.distances, cmap='coolwarm', edgecolor='k')
        ax.scatter(self.X[self.is_outlier][:, 0], self.X[self.is_outlier][:, 1],
                   c='red', edgecolor='k', label='Outliers')
        ax.scatter(self.centers[:, 0], self.centers[:, 1], c='black', marker='x', s=100, label='Centers')

        ax.set_title("KMeans Outlier Detection")
        ax.set_xlabel("Feature X")
        ax.set_ylabel("Feature Y")
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Outlier Score (Distance to Center)")
        ax.legend()
        plt.tight_layout()
        plt.show()

np.random.seed(42)
X_clusters, _ = make_blobs(n_samples=300, centers=[[0, 0], [6, 6]], cluster_std=1.0)
outliers = np.array([[10, 10], [-3, -3], [8, 0]])
X_all = np.vstack([X_clusters, outliers])

detector = KMeansOutlierDetector(X_all, n_clusters=2, percentile_threshold=95)
detector.fit()
detector.plot()
