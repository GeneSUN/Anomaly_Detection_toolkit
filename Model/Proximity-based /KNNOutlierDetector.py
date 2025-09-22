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