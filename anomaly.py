import numpy as np
import pandas as pd


class PopulationAnomalyDetector:
    """
    Population-referenced anomaly detector using z-scores.
    
    Flags samples where multiple features deviate significantly from population norms.
    Transparent, deterministic, and reviewer-safe.
    """

    def __init__(
        self,
        features,
        z_threshold=2.0,
        min_features_trigger=2,
    ):
        """
        Initialize the anomaly detector.
        
        Parameters
        ----------
        features : list
            Names of features to use for anomaly detection.
        z_threshold : float
            Absolute z-score threshold for flagging a single feature (default: 2.0).
        min_features_trigger : int
            Minimum number of features exceeding threshold to flag as anomaly (default: 2).
        """
        self.features = features
        self.z_threshold = z_threshold
        self.min_features_trigger = min_features_trigger
        self.means_ = None
        self.stds_ = None

    def fit(self, X):
        """
        Compute population statistics from training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data to compute population mean and std.
            
        Returns
        -------
        self
        """
        self.means_ = X[self.features].mean()
        self.stds_ = X[self.features].std()
        return self

    def score_samples(self, X):
        """
        Compute absolute z-scores for each sample and feature.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to score.
            
        Returns
        -------
        pd.DataFrame
            Absolute z-scores for each feature.
        """
        z = (X[self.features] - self.means_) / self.stds_
        return z.abs()

    def predict(self, X):
        """
        Flag anomalies based on z-score thresholding.
        
        A sample is flagged as anomalous if >= min_features_trigger features
        have |z| > z_threshold.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to predict anomalies for.
            
        Returns
        -------
        anomaly_flags : np.ndarray
            Binary array where 1 = anomaly, 0 = normal.
        z_scores : pd.DataFrame
            Per-feature absolute z-scores for interpretability.
        """
        z_scores = self.score_samples(X)
        triggers = (z_scores > self.z_threshold).sum(axis=1)
        anomaly_flags = (triggers >= self.min_features_trigger).astype(int)
        return anomaly_flags, z_scores
