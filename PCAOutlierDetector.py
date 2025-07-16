import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAOutlierDetector:
    def __init__(self, X_raw, n_components=2, scale_data=True, outlier_percentile=99, score_method="log_likelihood"):
        """
        Parameters:
        - X_raw: Input raw data
        - n_components: Number of principal components to keep
        - scale_data: Whether to standardize the data before PCA
        - outlier_percentile: Threshold percentile to flag outliers
        - score_method: One of ["log_likelihood", "projection"]
        """
        self.X_raw = X_raw
        self.n_components = n_components
        self.scale_data = scale_data
        self.outlier_percentile = outlier_percentile
        self.score_method = score_method

        self.scaler = StandardScaler() if scale_data else None
        self.pca = PCA(n_components=self.n_components)

        self.X_scaled = None
        self.X_pca = None
        self.residuals = None
        self.outlier_mask = None

    def fit(self, X_raw=None, score_method=None):
        """
        Fit PCA and compute outlier scores based on the selected scoring method.
        """
        if X_raw is not None:
            self.X_raw = X_raw
        if score_method is not None:
            self.score_method = score_method

        X = self.X_raw
        self.X_scaled = self.scaler.fit_transform(X) if self.scale_data else X.copy()
        self.X_pca = self.pca.fit_transform(self.X_scaled)

        if self.score_method == "log_likelihood":
            self.compute_outlier_score_by_log_likelihood()
        elif self.score_method == "projection":
            self.compute_outlier_score_by_minor_projection()
        else:
            raise ValueError(f"Unsupported score_method: {self.score_method}")

        return self

    def compute_outlier_score_by_log_likelihood(self):
        """
        Use PCA log-likelihood as an outlier score.
        Lower log-likelihood = more anomalous.
        """
        log_likelihoods = self.pca.score_samples(self.X_scaled)
        self.residuals = -log_likelihoods  # higher = more outlier
        threshold = np.percentile(self.residuals, self.outlier_percentile)
        self.outlier_mask = self.residuals > threshold
        return self

    def compute_outlier_score_by_minor_projection(self):
        """
        Use squared distance projected onto discarded components,
        normalized by their eigenvalues.
        """
        full_pca = PCA(n_components=self.X_scaled.shape[1])
        full_pca.fit(self.X_scaled)
        X_centered = self.X_scaled - full_pca.mean_

        V_minor = full_pca.components_[self.n_components:]
        λ_minor = full_pca.explained_variance_[self.n_components:]

        projections = X_centered @ V_minor.T
        squared_scores = (projections ** 2) / λ_minor
        self.residuals = np.sum(squared_scores, axis=1)

        threshold = np.percentile(self.residuals, self.outlier_percentile)
        self.outlier_mask = self.residuals > threshold
        return self

    def score_samples(self, X_new):
        """
        Return log-likelihoods of new samples under fitted PCA model.
        """
        if self.X_raw is None:
            raise RuntimeError("Model must be fitted before scoring new data.")
        X_new_scaled = self.scaler.transform(X_new) if self.scale_data else X_new
        return self.pca.score_samples(X_new_scaled)

    def plot_all_stages(self):
        """
        Plot original, scaled, and PCA-transformed space.
        Highlight outliers in each space.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Original
        ax = axes[0]
        ax.scatter(self.X_raw[:, 0], self.X_raw[:, 1], c='lightblue', edgecolor='k')
        ax.scatter(self.X_raw[self.outlier_mask, 0], self.X_raw[self.outlier_mask, 1],
                   facecolors='none', edgecolors='black', s=120, linewidth=2, label='Outlier')
        ax.set_title("Original (X_raw)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

        # Scaled
        ax = axes[1]
        ax.scatter(self.X_scaled[:, 0], self.X_scaled[:, 1], c='lightgreen', edgecolor='k')
        ax.scatter(self.X_scaled[self.outlier_mask, 0], self.X_scaled[self.outlier_mask, 1],
                   facecolors='none', edgecolors='black', s=120, linewidth=2, label='Outlier')
        ax.set_title("Standardized (X_scaled)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

        # PCA space
        ax = axes[2]
        ax.scatter(self.X_pca[:, 0], self.X_pca[:, 1], c='lightcoral', edgecolor='k')
        ax.scatter(self.X_pca[self.outlier_mask, 0], self.X_pca[self.outlier_mask, 1],
                   facecolors='none', edgecolors='black', s=120, linewidth=2, label='Outlier')
        ax.set_title("PCA Space (X_pca)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()

        plt.tight_layout()
        plt.suptitle("PCA Outlier Detection Pipeline", fontsize=16, y=1.05)
        plt.show()
