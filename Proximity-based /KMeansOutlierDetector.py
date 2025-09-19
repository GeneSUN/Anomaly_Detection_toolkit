import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from BaseOutlierDetector import BaseOutlierDetector
# ==============================================================
# KMeans Detector
# ==============================================================

class KMeansOutlierDetector(BaseOutlierDetector):
    def __init__(self, df, features, time_col="time", n_clusters=2,
                 distance_metric="euclidean", random_state=42,
                 scale=True, filter_percentile=None, threshold_percentile=99):
        super().__init__(df, features, time_col, scale, filter_percentile, threshold_percentile)
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.centers = None
        self.cov_inv = None

    def _compute_distance(self, X, centers, labels):
        if self.distance_metric == "euclidean":
            return np.linalg.norm(X - centers[labels], axis=1)
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(X - centers[labels]), axis=1)
        elif self.distance_metric == "mahalanobis":
            if self.cov_inv is None:
                cov = np.cov(X.T)
                self.cov_inv = np.linalg.pinv(cov)
            d = []
            for x, center in zip(X, centers[labels]):
                diff = x - center
                dist = np.sqrt(diff.T @ self.cov_inv @ diff)
                d.append(dist)
            return np.array(d)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def fit(self):
        X = self.df_clean[self.features].values
        self.kmeans.fit(X)
        labels = self.kmeans.labels_
        self.centers = self.kmeans.cluster_centers_
        distances = self._compute_distance(X, self.centers, labels)
        self.scores = self.df_clean["outlier_score"] = distances

        threshold = np.percentile(distances, self.threshold_percentile)
        self.is_outlier =  self.df_clean["is_outlier"] = distances >= threshold

        self.fitted = True
