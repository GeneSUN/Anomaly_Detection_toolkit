import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import List, Tuple

class DBSCANOutlierDetector:
    """
    DBSCAN-based outlier detection for time series with recent window focus.
    Supports optional standardization and percentile-based training data filtering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the data.
    features : List[str]
        List of feature column names to be used in DBSCAN.
    eps : float
        Neighborhood radius parameter for DBSCAN.
    min_samples : int
        Minimum number of points to form a core point in DBSCAN.
    recent_window_size : int
        Number of most recent points to evaluate for outliers.
    scale : bool
        Whether to standardize features.
    filter_percentile : float
        Percentile filter for training set. 100 means no filtering.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        eps: float = 0.5,
        min_samples: int = 5,
        recent_window_size: int = 24,
        scale: bool = True,
        filter_percentile: float = 100,
    ):
        self.df = df.copy()
        self.features = features
        self.eps = eps
        self.min_samples = min_samples
        self.recent_window_size = recent_window_size
        self.scale = scale
        self.filter_percentile = filter_percentile

        self.train_X = None
        self.test_X = None
        self.test_indices = None
        self.labels_ = None
        self.outlier_mask = None
        self.scaler = None
        self.dbscan_model = None

    def _apply_percentile_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.filter_percentile >= 100:
            return df
        for col in self.features:
            lower = np.percentile(df[col], (100 - self.filter_percentile) / 2)
            upper = np.percentile(df[col], 100 - (100 - self.filter_percentile) / 2)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        return df

    def _apply_scaling(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.scale:
            return X_train, X_test
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X_train), self.scaler.transform(X_test)

    def _split_data(self):
        df_train = self._apply_percentile_filter(self.df.iloc[:-self.recent_window_size])
        df_test = self.df.iloc[-self.recent_window_size:]
        self.test_indices = df_test.index

        X_train = df_train[self.features].values
        X_test = df_test[self.features].values
        self.train_X, self.test_X = self._apply_scaling(X_train, X_test)

    def fit(self) -> dict:
        self._split_data()

        self.dbscan_model = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(self.train_X)
        core_samples = self.train_X[self.dbscan_model.core_sample_indices_]


        self.labels_ = np.array([
            -1 if np.min(np.linalg.norm(core_samples - x, axis=1)) > self.eps else 0
            for x in self.test_X
        ])
        self.outlier_mask = self.labels_ == -1

        return {
            "new_labels": self.labels_.tolist(),
            "outlier_count": int(np.sum(self.outlier_mask)),
            "total_new_points": len(self.test_X),
            "outlier_indices": self.test_indices[self.outlier_mask].tolist(),
        }
    
    def plot_scatter(self, use_scaled: bool = False):
        """
        Scatter plot of DBSCAN clustering, differentiating train/test outliers.
    
        Parameters
        ----------
        use_scaled : bool
            If True, plot using standardized features.
            If False, plot using original unscaled feature values.
        """
        if self.dbscan_model is None:
            raise ValueError("Call .fit() before plotting.")
    
        train_labels = self.dbscan_model.labels_
        train_outliers = np.where(train_labels == -1)[0]
    
        if use_scaled:
            X_train_plot = self.train_X
            X_test_plot = self.test_X
        else:
            X_train_plot = self.df.iloc[:-self.recent_window_size][self.features].values
            X_test_plot = self.df.iloc[-self.recent_window_size:][self.features].values
    
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train_plot[:, 0], X_train_plot[:, 1], c='blue', label='Train Inliers')
        plt.scatter(X_train_plot[train_outliers, 0], X_train_plot[train_outliers, 1],
                    c='gray', label='Train Outliers', alpha=0.3)
        plt.scatter(X_test_plot[~self.outlier_mask, 0], X_test_plot[~self.outlier_mask, 1],
                    c='green', label='Test Inliers')
        plt.scatter(X_test_plot[self.outlier_mask, 0], X_test_plot[self.outlier_mask, 1],
                    c='red', label='Test Outliers')
    
        plt.xlabel(self.features[0])
        if len(self.features) > 1:
            plt.ylabel(self.features[1])
        plt.title(f"DBSCAN Outlier Detection (Last {self.recent_window_size} Points)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    
    def plot_timeseries(self, time_col: str, feature: str = None):
        """
        Plot time series with DBSCAN-detected outliers from the test window.
        
        Parameters
        ----------
        time_col : str
            Name of the timestamp column for the X-axis.
        feature : str, optional
            Name of the feature to plot on the Y-axis. Defaults to self.features[0].
        """
        if self.labels_ is None:
            raise ValueError("Call .fit() before plotting.")
    
        if feature is None:
            feature = self.features[0]
        elif feature not in self.df.columns:
            raise ValueError(f"Feature '{feature}' not found in the dataframe.")
    
        ts_df = self.df.copy()
        ts_df["outlier"] = False
        ts_df.loc[self.test_indices[self.outlier_mask], "outlier"] = True
    
        plt.figure(figsize=(12, 5))
        plt.plot(ts_df[time_col], ts_df[feature], label=feature, alpha=0.7)
        plt.scatter(
            ts_df[time_col][ts_df["outlier"]],
            ts_df[feature][ts_df["outlier"]],
            color='red', label='Outliers'
        )
        plt.xlabel(time_col)
        plt.ylabel(feature)
        plt.title(f"Time Series with DBSCAN Outliers ({feature})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


