import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAOutlierDetector:
    def __init__(self, X_raw, n_components=2, scale_data=True, outlier_percentile=99, score_method = "log_likelihood"):
        self.X_raw = X_raw
        self.n_components = n_components
        self.scale_data = scale_data
        self.outlier_percentile = outlier_percentile

        self.scaler = StandardScaler() if scale_data else None
        self.pca = PCA(n_components=self.n_components)

        self.X_scaled = None
        self.X_scaled = None
        self.residuals = None
        self.outlier_mask = None

        self.score_method = score_method

    def fit(self, X_raw = None, score_method = None):
        if X_raw is None:
            X_raw = self.X_raw
        if score_method is None:
            score_method = self.score_method
            
        self.X_scaled = self.scaler.fit_transform(X_raw) if self.scale_data else self.X_raw.copy()
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        if score_method == "log_likelihood":
            self.compute_outlier_score_by_log_likelihood()
        elif score_method == "projection":
            self.compute_outlier_score_by_minor_projection()    
        
        return self

    def compute_outlier_score_by_log_likelihood(self):
        """
        Use PCA log-likelihood as an outlier score.
        Lower log-likelihood = more anomalous.
        """
        X_scaled = self.X_scaled
        log_likelihoods = self.pca.score_samples(X_scaled)  # shape: (n_samples,)
        self.residuals = -log_likelihoods  # convert to "outlier score"
        threshold = np.percentile(self.residuals, self.outlier_percentile)
        self.outlier_mask = self.residuals > threshold
        return self

    def compute_outlier_score_by_minor_projection(self):
        """
        Use projections onto discarded PCs (orthogonal to retained hyperplane).
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

    def plot_all_stages(self):
        """
        Plot 2x2 subplots: X_raw, X_scaled, X_pca, X_proj.
        Outliers are marked with black-edged circles.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
        # Plot 1: Raw
        ax = axes[0]
        ax.scatter(self.X_raw[:, 0], self.X_raw[:, 1], c='lightblue')
        ax.scatter(self.X_raw[self.outlier_mask, 0], self.X_raw[self.outlier_mask, 1],
                   facecolors='none', edgecolors='black', s=120, linewidth=2, label='Outlier')
        ax.set_title("Original (X_raw)")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.legend()
    
        # Plot 2: Scaled
        ax = axes[1]
        ax.scatter(self.X_scaled[:, 0], self.X_scaled[:, 1], c='lightgreen')
        ax.scatter(self.X_scaled[self.outlier_mask, 0], self.X_scaled[self.outlier_mask, 1],
                   facecolors='none', edgecolors='black', s=120, linewidth=2, label='Outlier')
        ax.set_title("Standardized (X_scaled)")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.legend()
        
        # Plot 3: PCA space
        ax = axes[2]
        ax.scatter(self.X_pca[:, 0], self.X_pca[:, 1], c='lightcoral')
        ax.scatter(self.X_pca[self.outlier_mask, 0], self.X_pca[self.outlier_mask, 1],
                   facecolors='none', edgecolors='black', s=120, linewidth=2, label='Outlier')
        ax.set_title("PCA Space (X_pca)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        
        # Compute common range
        x_min, x_max = self.X_pca[:, 0].min(), self.X_pca[:, 0].max()
        y_min, y_max = self.X_pca[:, 1].min(), self.X_pca[:, 1].max()
        xy_min = min(x_min, y_min) -1
        xy_max = max(x_max, y_max) + 1
        ax.set_xlim(xy_min, xy_max)
        ax.set_ylim(xy_min, xy_max)
        
        ax.legend()

        
    def score_samples(self, X_new, visualize=False):
        """
        Compute log-likelihood scores of new samples under the current PCA model.
        Return scores, threshold, and binary outlier flags.
    
        Parameters:
        - X_new: array-like of shape (n_samples, n_features)
        - visualize: whether to plot X_new against X_raw with outlier labels
    
        Returns:
        - log_likelihoods: ndarray of log-likelihoods
        - threshold: float, the outlier cutoff
        - is_outlier: boolean array of shape (n_samples,)
        """
        if self.X_raw is None:
            raise RuntimeError("Model must be fitted before scoring new data.")
    
        X_new_scaled = self.scaler.transform(X_new) if self.scale_data else X_new
        log_likelihoods = self.pca.score_samples(X_new_scaled)
        residuals = -log_likelihoods
        threshold = np.percentile(self.residuals, self.outlier_percentile)
        is_outlier = residuals > threshold
    
        if visualize:
            if self.X_raw.shape[1] != 2:
                raise ValueError("Visualization only supports 2D input data.")
    
            # Separate original data
            X_raw_inlier = self.X_raw[~self.outlier_mask]
            X_raw_outlier = self.X_raw[self.outlier_mask]
    
            # Separate new data
            X_new = np.array(X_new)
            X_new_inlier = X_new[~is_outlier]
            X_new_outlier = X_new[is_outlier]
    
            plt.figure(figsize=(8, 6))
    
            # Plot original inliers and outliers
            plt.scatter(X_raw_inlier[:, 0], X_raw_inlier[:, 1], c='lightblue', label="X_raw_inlier", alpha=0.3, s= 40)
            plt.scatter(X_raw_outlier[:, 0], X_raw_outlier[:, 1], c='yellow', label="X_raw_outlier", alpha=0.6)
    
            # Plot new inliers and annotate
            if len(X_new_inlier) > 0:
                plt.scatter(X_new_inlier[:, 0], X_new_inlier[:, 1], c='green', edgecolor='black', s=120, label="X_new_inlier", zorder=3)
                for x, y in X_new_inlier:
                    plt.text(x + 0.3, y, "inlier", fontsize=9, color='green')
    
            # Plot new outliers and annotate
            if len(X_new_outlier) > 0:
                plt.scatter(X_new_outlier[:, 0], X_new_outlier[:, 1], c='red', edgecolor='black', s=120, label="X_new_outlier", zorder=3)
                for x, y in X_new_outlier:
                    plt.text(x + 0.3, y, "outlier", fontsize=9, color='red')
    
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.title("New Sample Log-Likelihood Classification")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
        return log_likelihoods, threshold, is_outlier

