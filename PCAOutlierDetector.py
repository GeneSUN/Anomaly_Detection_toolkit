import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

class PCAOutlierDetector:
    def __init__(self, n_components=2, scale_data=True, outlier_percentile=98):
        self.n_components = n_components
        self.scale_data = scale_data
        self.outlier_percentile = outlier_percentile

        self.scaler = StandardScaler() if scale_data else None
        self.pca = PCA(n_components=self.n_components)

        # Internal state
        self.X_raw = None
        self.X_scaled = None
        self.X_proj = None
        self.X_pca = None
        self.residuals = None
        self.outlier_mask = None

    def fit(self, X):
        """
        Fit PCA, compute projection, residuals, and identify outliers.
        """
        self.X_raw = X
        self.X_scaled = self.scaler.fit_transform(X) if self.scale_data else X.copy()

        self.X_pca = self.pca.fit_transform(self.X_scaled)
        self.X_proj = self.pca.inverse_transform(self.X_pca)

        # Compute residuals in scaled space
        self.residuals = np.linalg.norm(self.X_scaled - self.X_proj, axis=1)

        # Identify top outliers
        threshold = np.percentile(self.residuals, self.outlier_percentile)
        self.outlier_mask = self.residuals > threshold
        return self


    def visualize_original_space(self):
        """
        Visualize in original coordinate space (2D only).
        """
        if self.X_raw.shape[1] != 2:
            raise ValueError("This visualization requires 2D input data.")

        # Arrow: principal direction in original space
        mean_scaled = np.mean(self.X_scaled, axis=0)
        vector_scaled = self.pca.components_[0] * 3
        mean = self.scaler.inverse_transform([mean_scaled])[0] if self.scale_data else mean_scaled
        vec_end = self.scaler.inverse_transform([mean_scaled + vector_scaled])[0] if self.scale_data else mean_scaled + vector_scaled
        dx, dy = vec_end - mean

        # Projections in original scale
        X_proj_orig = self.scaler.inverse_transform(self.X_proj) if self.scale_data else self.X_proj

        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_raw[:, 0], self.X_raw[:, 1], c=self.residuals, cmap='coolwarm', edgecolor='k', label='Data Points')

        for i in range(len(self.X_raw)):
            plt.plot([self.X_raw[i, 0], X_proj_orig[i, 0]], [self.X_raw[i, 1], X_proj_orig[i, 1]], 'gray', alpha=0.3)

        plt.arrow(mean[0], mean[1], dx, dy, color='green', width=0.05, head_width=0.3, label='PC1 direction')
        plt.plot(X_proj_orig[:, 0], X_proj_orig[:, 1], 'k.', alpha=0.5, label='Projection onto PC1')

        plt.scatter(self.X_raw[self.outlier_mask, 0], self.X_raw[self.outlier_mask, 1],
                    edgecolor='black', facecolor='none', s=120, linewidth=2, label='Outliers')

        plt.colorbar(label='Residual Distance')
        plt.title("Outlier Detection in Original Feature Space")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def visualize_pca_space(self):
        """
        Visualize data projected onto PC1 and PC2 axes.
        """
        if self.n_components < 2:
            raise ValueError("n_components must be at least 2 to visualize PCA space.")

        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], c=self.residuals, cmap='coolwarm', edgecolor='k', label='Projected Points')

        plt.scatter(self.X_pca[self.outlier_mask, 0], self.X_pca[self.outlier_mask, 1],
                    edgecolor='black', facecolor='none', s=120, linewidth=2, label='Outliers')

        plt.colorbar(label='Residual Distance')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Outlier Detection in PCA Space (PC1 vs PC2)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()