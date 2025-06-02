
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class DBSCANOutlierDetector:
    def __init__(self, df, features, eps=3, min_samples=2):
        """
        df: pandas DataFrame containing the data
        features: list of feature column names for DBSCAN (e.g., ["avg_4gsnr", "avg_5gsnr"])
        eps: neighborhood radius for DBSCAN
        min_samples: minimum samples for a core point
        """
        self.df = df.copy()
        self.features = features
        self.eps = eps
        self.min_samples = min_samples
        self.outlier_label = None  # to be set after fit

    def fit(self):
        X = self.df[self.features].values
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        self.df['outlier_label'] = (clustering.labels_ == -1).astype(int)
        self.outlier_label = self.df['outlier_label']
        return self.df

    def plot(self):
        if self.outlier_label is None:
            raise ValueError("Call .fit() before plotting.")
        X = self.df[self.features].values
        outlier_label = self.outlier_label.values

        inliers = X[outlier_label == 0]
        outliers = X[outlier_label == 1]

        plt.figure(figsize=(8, 6))
        plt.scatter(inliers[:, 0], inliers[:, 1], c='blue', marker='o', label='Inlier')
        plt.scatter(outliers[:, 0], outliers[:, 1], c='red', marker='x', s=100, label='Outlier')
        plt.xlabel(self.features[0])
        plt.ylabel(self.features[1] if len(self.features) > 1 else '')
        plt.title(f"DBSCAN Outlier Detection (eps={self.eps}, min_samples={self.min_samples})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Assume df_cap_hour_pd is your DataFrame
detector = DBSCANOutlierDetector(df, features=["avg_4gsnr", "avg_5gsnr"], eps=3, min_samples=2)
df_with_label = detector.fit()
detector.plot()
