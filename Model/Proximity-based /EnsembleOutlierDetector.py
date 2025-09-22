from .BaseOutlierDetector import BaseOutlierDetector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class EnsembleOutlierDetector(BaseOutlierDetector):
    def __init__(self, df, detectors, method='average_score', weights=None, threshold=0.5):
        """
        Ensemble Outlier Detector.

        Parameters
        ----------
        detectors : list
            List of fitted detector objects (each must be a child of BaseOutlierDetector).
        method : str
            {'majority_vote', 'weighted_vote', 'average_score', 'max_score'}.
        weights : list of float
            Same length as detectors. Used for weighted methods.
        threshold : float
            Threshold for deciding outliers in voting/score methods.
        """
        # Initialize BaseOutlierDetector with a dummy df & features
        # Will be replaced once we aggregate from detectors
        self.df = df.copy()

        self.detectors = detectors
        self.method = method
        self.weights = weights or [1.0] * len(detectors)
        assert len(self.weights) == len(detectors), "Weights must match detectors"
        self.threshold = threshold

        self.detector_names = [det.__class__.__name__ for det in detectors]
        self.fitted = False

    def fit(self):
        # Assume all detectors are already fitted on the same df
        for det in self.detectors:
            if not det.fitted:
                det.fit()

        # Use df_clean from the first detector as reference
        base_df = self.detectors[0].df_clean.copy()
        scores = np.vstack([det.df_clean["outlier_score"].values for det in self.detectors]).T
        labels = np.vstack([det.df_clean["is_outlier"].values for det in self.detectors]).T

        # Normalize scores per detector
        norm_scores = np.zeros_like(scores, dtype=float)
        for j in range(scores.shape[1]):
            s = scores[:, j]
            min_s, max_s = np.min(s), np.max(s)
            if max_s > min_s:
                norm_scores[:, j] = (s - min_s) / (max_s - min_s)
            else:
                norm_scores[:, j] = 0.0

        # Combine scores/labels
        if self.method == 'average_score':
            w = np.array(self.weights) / np.sum(self.weights)
            combined = np.dot(norm_scores, w)
            is_outlier = combined >= self.threshold

        elif self.method == 'max_score':
            combined = np.max(norm_scores, axis=1)
            is_outlier = combined >= self.threshold

        elif self.method == 'majority_vote':
            votes = labels.astype(int)
            vote_sum = np.dot(votes, self.weights)
            thresh_votes = np.sum(self.weights) / 2.0
            is_outlier = vote_sum >= thresh_votes
            combined = vote_sum / np.sum(self.weights)

        elif self.method == 'weighted_vote':
            votes = labels.astype(int)
            vote_sum = np.dot(votes, self.weights)
            is_outlier = vote_sum >= self.threshold * np.sum(self.weights)
            combined = vote_sum / np.sum(self.weights)

        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

        # Store results
        base_df["outlier_score"] = combined
        base_df["is_outlier"] = is_outlier

        # Replace df_clean (so inherited plot methods work)
        self.df_clean = base_df
        self.features = self.detectors[0].features
        self.time_col = self.detectors[0].time_col
        self.fitted = True
        return self

    def plot(self):
        """Extend plot with info about ensemble composition."""
        super().plot()
        print(f"[INFO] Ensemble method: {self.method}")
        print(f"[INFO] Detectors used: {', '.join(self.detector_names)}")
