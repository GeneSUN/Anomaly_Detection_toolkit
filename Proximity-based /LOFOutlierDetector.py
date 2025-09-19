from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from BaseOutlierDetector import BaseOutlierDetector

# ==============================================================
# LOF Detector
# ==============================================================

class LOFOutlierDetector(BaseOutlierDetector):
    def __init__(self, df, features, time_col="time", contamination=0.05,
                 scale=True, filter_percentile=None, threshold_percentile=99):
        super().__init__(df, features, time_col, scale, filter_percentile, threshold_percentile)
        self.contamination = contamination
        self.model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)

    def fit(self):
        X = self.df_clean[self.features].values
        y_pred = self.model.fit_predict(X)
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
