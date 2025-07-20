from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class LOFOutlierDetector:
    """
    LOFOutlierDetector uses the Local Outlier Factor (LOF) algorithm to identify anomalies in a dataset.

    Parameters
    ----------
    X : array-like or DataFrame
        Input dataset with shape (n_samples, n_features). Can be a NumPy array or a pandas DataFrame.

    contamination : float, default=0.1
        The proportion of outliers in the data set. This acts as a filtering mechanism.

    scale : bool, default=True
        Whether to apply standard scaling to the data during preprocessing.

    model : LocalOutlierFactor object, optional
        A fully specified LocalOutlierFactor model. If provided, this model will override the default configuration.
        Useful when you want to control all underlying LOF parameters.
    """
    def __init__(self, X, contamination=0.1, scale=True, model=None):
        self.X_raw = X.values if isinstance(X, pd.DataFrame) else X
        self.contamination = contamination
        self.scale = scale
        self.model = model  # optionally passed by user
        self.is_outlier = None
        self.scores = None

    def preprocess(self):
        if self.scale:
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X_raw)
        else:
            self.X = self.X_raw.copy()

    def fit(self):
        self.preprocess()
        if self.model is None:
            self.model = LocalOutlierFactor(n_neighbors=20, contamination=self.contamination)
        self.is_outlier = self.model.fit_predict(self.X) == -1
        self.scores = -self.model.negative_outlier_factor_

    def plot(self):
        if self.X.shape[1] == 2:
            X_plot = self.X
        else:
            X_plot = PCA(n_components=2).fit_transform(self.X)

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(X_plot[~self.is_outlier, 0], X_plot[~self.is_outlier, 1],
                        c=self.scores[~self.is_outlier], cmap='coolwarm', edgecolor='k', label='Inliers')
        ax.scatter(X_plot[self.is_outlier, 0], X_plot[self.is_outlier, 1],
                   c='red', edgecolor='k', label='Outliers')
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("LOF Outlier Score")
        ax.set_title("Local Outlier Factor (LOF) Detection")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()


# === Example Usage ===
# Generate data
np.random.seed(42)
X_normal = np.random.randn(100, 2) * 0.75 + np.array([2, 2])
X_outliers = np.random.uniform(low=-2, high=6, size=(5, 2))
X_all = np.vstack([X_normal, X_outliers])

# Use the LOFOutlierDetector class
lof_detector = LOFOutlierDetector(X_all, contamination=0.05)
lof_detector.fit()
lof_detector.plot()

custom_model = LocalOutlierFactor(n_neighbors=10, metric='manhattan', contamination=0.1)
lof = LOFOutlierDetector(X_all, model=custom_model)
lof.fit()
lof.plot()
