import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

class KMeansOutlierDetector:
    """
    KMeans-Based Outlier Detector using Distance to Cluster Centroids.

    This class detects outliers by clustering the data using KMeans and computing the Euclidean
    distance from each point to its assigned cluster center. Points that are farthest from their
    cluster centroids are considered potential outliers.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input data to detect outliers from. Should be 2D or higher-dimensional.
    n_clusters : int, default=2
        Number of clusters to form using KMeans.
    scale : bool, default=True
        Whether to standardize features before clustering.
    filter_percentile : float or None, default=None
        Optional. If set, removes both tails of extreme values based on distance from the data center
        before clustering. Helps prevent extreme outliers from distorting cluster centers.
    threshold_percentile : float, default=95
        Percentile of the distance distribution to use as a cutoff for classifying outliers.
        Points with distances >= this threshold are labeled as outliers.

    Attributes
    ----------
    labels : np.ndarray
        Cluster labels assigned to each point.
    centers : np.ndarray
        Coordinates of cluster centroids.
    distances : np.ndarray
        Euclidean distances from each point to its assigned cluster centroid. This is also the Anomalous Score
    is_outlier : np.ndarray of bool
        Boolean array indicating which points are classified as outliers.
    X : np.ndarray
        The preprocessed input data used for clustering.

    Methods
    -------
    fit()
        Fit the KMeans model to the data, assign cluster labels, compute distances, and identify outliers.
    plot()
        Visualize outlier scores (distance to cluster center) and highlight detected outliers.
        If input data has more than 2 dimensions, PCA is used for 2D visualization.

    Example
    -------
    >>> detector = KMeansOutlierDetector(X, n_clusters=2, threshold_percentile=97)
    >>> detector.fit()
    >>> detector.plot()
    """

    def __init__(self, X, n_clusters=2, scale=True,
                 filter_percentile=None, threshold_percentile=95,
                 random_state=42):

        self.X_raw = X.copy()
        self.n_clusters = n_clusters
        self.scale = scale
        self.random_state = random_state
        self.filter_percentile = filter_percentile
        self.threshold_percentile = threshold_percentile

        self.scaler = StandardScaler() if scale else None
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
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
        self.kmeans.fit(self.X)
        self.labels = self.kmeans.labels_
        self.centers = self.kmeans.cluster_centers_
        self.distances = np.linalg.norm(self.X - self.centers[self.labels], axis=1) # Outlier Scores

        threshold_value = np.percentile(self.distances, self.threshold_percentile)
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


