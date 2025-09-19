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
        
        self.df_clean["outlier_score"] = self.scores
        self.is_outlier  = self.df_clean["is_outlier"] = y_pred == -1
        self.fitted = True