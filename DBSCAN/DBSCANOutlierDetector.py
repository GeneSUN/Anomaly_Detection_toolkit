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
    DBSCAN-based outlier (anomaly) detection for tabular data with optional feature scaling.

    This class applies DBSCAN clustering to identify outliers in a pandas DataFrame. 
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

    Usage
    -----
    >>> detector = DBSCANOutlierDetector(df, features=["x1", "x2"], eps=3, min_samples=2)

    Attributes
    ----------
    df : pandas.DataFrame
        The internal dataframe, with outlier labels added after fitting.
    outlier_label : pandas.Series
        Binary label for each row (0 = inlier, 1 = outlier). Set after calling `fit()`.
    scaled : bool
        Whether features have been standardized.
    """

    def __init__(self, df, features, eps=3, min_samples=2):
        """
        Initialize the outlier detector with data and parameters.
        """
        self.df = df.copy()
        self.features = features
        self.eps = eps
        self.min_samples = min_samples
        self.outlier_label = None  # to be set after fit
        self.scaled = False        # Track if scaling was applied

    def scale_features(self):
        """
        Standardize the selected features using sklearn's StandardScaler (mean=0, std=1).

        This is optional, but strongly recommended if your features are on different scales,
        or if you want to ensure fair distance calculation for DBSCAN.
        Must be called before `fit()`. If called, you may want to reduce `eps`.

        """
        scaler = StandardScaler()
        self.df[self.features] = scaler.fit_transform(self.df[self.features])
        self.scaled = True

    def fit(self):
        """
        Fit DBSCAN to the (optionally scaled) features and assign outlier labels.

        Returns
        -------
        pandas.DataFrame
            The dataframe with an added 'outlier_label' column (0 = inlier, 1 = outlier).
        """
        X = self.df[self.features].values
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        self.df['outlier_label'] = (clustering.labels_ == -1).astype(int)
        self.outlier_label = self.df['outlier_label']
        return self.df

    def plot(self):
        """
        Plot the selected features with inliers and outliers highlighted.

        Raises
        ------
        ValueError
            If called before `fit()`.
        Notes
        -----
        - Works best if exactly 2 features are used for visualization.
        - The plot title indicates whether standardization was applied.
        """
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
        title_suffix = " (Standardized)" if self.scaled else ""
        plt.title(f"DBSCAN Outlier Detection{title_suffix} (eps={self.eps}, min_samples={self.min_samples})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    detector = DBSCANOutlierDetector(df, features=["avg_4gsnr", "avg_5gsnr"], eps=3, min_samples=2)
    df_with_label = detector.fit()
    detector.plot()


    detector = DBSCANOutlierDetector(df, features=["avg_4gsnr", "avg_5gsnr"], eps=0.5, min_samples=2)
    detector.scale_features()   # <- Call before fit()
    df_with_label = detector.fit()
    detector.plot()
