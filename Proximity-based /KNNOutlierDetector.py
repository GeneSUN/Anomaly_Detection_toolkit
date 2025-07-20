import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class KNNOutlierDetector:
    """
    K-Nearest Neighbors Distance-Based Outlier Detector.

    This class implements a simple distance-based outlier detection method using k-nearest neighbors (k-NN).
    It supports multiple scoring strategies and provides tools for preprocessing, fitting, scoring, and visualizing.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input data to detect outliers from. Can be 2D or higher-dimensional.
    k : int, default=5
        Number of neighbors to use in distance calculation.
    method : {'kth', 'average', 'harmonic'}, default='kth'
        Scoring method:
        - 'kth': use distance to the k-th nearest neighbor
        - 'average': use the mean distance to the k nearest neighbors
        - 'harmonic': use the harmonic mean of distances to the k nearest neighbors
    scale : bool, default=True
        Whether to standardize features before fitting.
    filter_percentile : float or None, default=None
        Optional. If set, will remove extreme values (both tails) before fitting based on Euclidean distance
        from the mean. Useful to avoid skewed cluster centers due to strong outliers.
    threshold_percentile : float, default=95
        Percentile threshold to classify outliers after scoring.
        Points with scores >= this percentile will be labeled as outliers.

    Attributes
    ----------
    scores : np.ndarray
        Outlier scores computed based on selected distance method.
    is_outlier : np.ndarray of bool
        Boolean array indicating which points are classified as outliers.
    X : np.ndarray
        The preprocessed input data used in scoring.

    Methods
    -------
    fit()
        Fit the detector, compute outlier scores, and classify outliers.
    plot()
        Visualize outlier scores and highlight detected outliers.
        If input has more than 2 dimensions, PCA is applied for plotting.

    Example
    -------
    >>> detector = KNNOutlierDetector(X, k=5, method='average', filter_percentile=5)
    >>> detector.fit()
    >>> detector.plot()
    """

    def __init__(self, X, k=5, method='kth', scale=True,
                 filter_percentile=None, threshold_percentile=95):
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

# Test example
np.random.seed(42)
X_normal = np.random.randn(100, 2) * 0.75 + np.array([2, 2])
X_outliers = np.random.uniform(low=-2, high=6, size=(5, 2))
X_all = np.vstack([X_normal, X_outliers])

knn_detector = KNNOutlierDetector(X_all, k=5, method='harmonic', filter_percentile=1, threshold_percentile=95)
knn_detector.fit()
knn_detector.plot()

# Test with 5D data
np.random.seed(42)
X_5d_normal = np.random.randn(100, 5) * 0.5 + 1
X_5d_outliers = np.random.uniform(low=-3, high=5, size=(5, 5))
X_5d_all = np.vstack([X_5d_normal, X_5d_outliers])

knn_detector_highdim = KNNOutlierDetector(X_5d_all, k=5, method='average', threshold_percentile=95)
knn_detector_highdim.fit()
knn_detector_highdim.plot()
