import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class DBSCANOutlierDetector:
    """
    DBSCAN-based outlier detection that uses all data for training
    but only tests whether the most recent `n` points are outliers.

    Features can be optionally standardized (zero mean, unit variance) before fitting,
    which is important because DBSCAN is sensitive to feature scales.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the data.
    features : list of str
        Column names to use for DBSCAN clustering (typically 2 features for visualization).
    eps : float, default=3
        The neighborhood radius parameter for DBSCAN.
        Note: If using standardized features, adjust `eps` accordingly (typical values are 0.3 to 1.0).
    min_samples : int, default=2
        Minimum number of points to form a core point in DBSCAN.
    recent_window_size : int
        Number of most recent points to evaluate for outliers.
    scale : bool, default=True
        Whether to scale features before clustering.
    Usage
    -----
    >>> detector = DBSCANOutlierDetector(df, features=["x1", "x2"], eps=3, min_samples=2)
    >>> df_with_label = detector.fit()
    >>> detector.plot()
    Attributes
    ----------
    df : pandas.DataFrame
        The internal dataframe, with outlier labels added after fitting.
    outlier_label : pandas.Series
        Binary label for each row (0 = inlier, 1 = outlier). Set after calling `fit()`.
    scaled : bool
        Whether features have been standardized.
    """

    def __init__(self, df, features, eps=0.5, min_samples=5, recent_window_size=24, scale=True):
        self.df = df.copy()
        self.features = features
        self.eps = eps
        self.min_samples = min_samples
        self.recent_window_size = recent_window_size
        self.scale = scale
        self.scaler = None
        self.full_X = None
        self.labels_ = None
        self.outlier_mask = None

    def _prepare_features(self):
        X = self.df[self.features].values
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        self.full_X = X

    def fit(self):
        """
        Fit DBSCAN on all data, and check outliers only in the recent window.

        Returns
        -------
        dict
            {
                "new_labels": list,
                "outlier_count": int,
                "total_new_points": int,
                "outlier_indices": list
            }
        """
        self._prepare_features()
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = dbscan.fit_predict(self.full_X)

        # define new data segment (last n rows)
        new_idx = np.arange(len(self.full_X) - self.recent_window_size, len(self.full_X))
        new_labels = self.labels_[new_idx]
        outlier_mask = new_labels == -1

        self.outlier_mask = outlier_mask

        return {
            "new_labels": new_labels.tolist(),
            "outlier_count": int(np.sum(outlier_mask)),
            "total_new_points": len(new_labels),
            "outlier_indices": new_idx[outlier_mask].tolist(),
        }

    def plot(self):
        """
        Plot all points with last-N outliers highlighted (works best for 2D features).
        """
        if self.labels_ is None:
            raise ValueError("Call .fit() before plotting.")

        X = self.full_X
        all_labels = self.labels_
        new_start = len(X) - self.recent_window_size

        plt.figure(figsize=(8, 6))
        # plot all inliers
        plt.scatter(X[all_labels != -1, 0], X[all_labels != -1, 1], c='blue', label='Inliers')
        # plot all outliers
        plt.scatter(X[all_labels == -1, 0], X[all_labels == -1, 1], c='gray', alpha=0.3, label='Outliers (overall)')
        # highlight recent window outliers
        recent_outlier_idx = np.where(self.outlier_mask)[0] + new_start
        plt.scatter(X[recent_outlier_idx, 0], X[recent_outlier_idx, 1], c='red', label='Recent Outliers')

        plt.xlabel(self.features[0])
        if len(self.features) > 1:
            plt.ylabel(self.features[1])
        plt.title(f"DBSCAN Outlier Detection (Last {self.recent_window_size} Points)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

