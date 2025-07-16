import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

class PCAOutlierDetector:
    def __init__(self, n_components=1, scale_data=True):
        self.n_components = n_components
        self.scale_data = scale_data
        self.scaler = StandardScaler() if scale_data else None
        self.pca = PCA(n_components=self.n_components)

        # Internal state
        self.X_raw = None        # original unscaled data
        self.X_scaled = None     # scaled version of data
        self.X_proj = None       # back-projected data
        self.residuals = None
        self.outlier_mask = None

    def fit(self, X):
        """
        Fit PCA on the (optionally standardized) data and compute residuals.
        """
        self.X_raw = X
        if self.scale_data:
            self.X_scaled = self.scaler.fit_transform(X)
        else:
            self.X_scaled = X.copy()

        X_pca = self.pca.fit_transform(self.X_scaled)
        self.X_proj = self.pca.inverse_transform(X_pca)
        self.residuals = np.linalg.norm(self.X_scaled - self.X_proj, axis=1)
        return self

    def get_outlier_scores(self):
        if self.residuals is None:
            raise RuntimeError("You must call fit() before getting scores.")
        return self.residuals

    def get_outliers(self, percentile=95):
        if self.residuals is None:
            raise RuntimeError("You must call fit() before identifying outliers.")
        threshold = np.percentile(self.residuals, percentile)
        self.outlier_mask = self.residuals > threshold
        return self.outlier_mask

    def visualize(self, percentile=95):
        """
        Only supports 2D original data for visualization.
        """
        if self.X_raw.shape[1] != 2:
            raise ValueError("Visualization only supported for 2D input data.")

        self.get_outliers(percentile)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_raw[:, 0], self.X_raw[:, 1], c=self.residuals, cmap='coolwarm', edgecolor='k', label='Data Points')

        # PC1 direction (in scaled space, then map back)
        mean_scaled = np.mean(self.X_scaled, axis=0)
        vector_scaled = self.pca.components_[0] * 3  # scaled for plotting
        if self.scale_data:
            mean = self.scaler.inverse_transform([mean_scaled])[0]
            vec_end = self.scaler.inverse_transform([mean_scaled + vector_scaled])[0]
        else:
            mean = mean_scaled
            vec_end = mean_scaled + vector_scaled

        dx, dy = vec_end - mean
        plt.arrow(mean[0], mean[1], dx, dy, color='green', width=0.05, head_width=0.3, label='PC1 direction')

        # Projection line (from scaled to original space)
        if self.scale_data:
            X_proj_original = self.scaler.inverse_transform(self.X_proj)
        else:
            X_proj_original = self.X_proj

        plt.plot(X_proj_original[:, 0], X_proj_original[:, 1], 'k.', alpha=0.5, label='Projection onto PC1')

        for i in range(len(self.X_raw)):
            plt.plot([self.X_raw[i, 0], X_proj_original[i, 0]],
                     [self.X_raw[i, 1], X_proj_original[i, 1]], 'gray', alpha=0.3)

        plt.scatter(self.X_raw[self.outlier_mask, 0], self.X_raw[self.outlier_mask, 1],
                    edgecolor='black', facecolor='none', s=120, linewidth=2, label='Outliers (Top 5%)')

        plt.colorbar(label='Residual Distance (in PCA space)')
        plt.title("Outlier Detection via PCA Residual (Projection onto PC1)")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()