import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from models import get_models
from utils import compute_metrics
from anomaly import PopulationAnomalyDetector


def cross_validate_models(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []

    for name, model in get_models(seed).items():
        fold_metrics = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            fold_metrics.append(compute_metrics(y_test, y_prob))

        df = pd.DataFrame(fold_metrics)
        results.append(
            {
                "model": name,
                **df.mean().to_dict(),
                **{f"{k}_std": v for k, v in df.std().to_dict().items()},
            }
        )

    return pd.DataFrame(results)

def add_noise(X, noise_level=0.5):
    X_noisy = X.copy()
    for col in X_noisy.select_dtypes(include="number").columns:
        std = X_noisy[col].std()
        X_noisy[col] += np.random.normal(0, noise_level * std, len(X_noisy))
    return X_noisy

def feature_dropout(X, dropout_rate=0.6):
    X_dropped = X.copy()
    for col in X_dropped.columns:
        mask = np.random.rand(len(X_dropped)) < dropout_rate
        X_dropped.loc[mask, col] = np.nan
    return X_dropped

from simulator import simulate_dataset

def prevalence_shift_experiment(prevalence_values):
    results = []

    for p in prevalence_values:
        X, y = simulate_dataset(prevalence=p)
        df = cross_validate_models(X, y)
        df["prevalence"] = p
        results.append(df)

    return pd.concat(results)


def evaluate_anomaly_detector(X, y):
    """
    Evaluate population-referenced anomaly detection against ground truth labels.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Ground truth depression risk labels.
        
    Returns
    -------
    dict
        Performance metrics: precision, recall, f1, flag_rate.
    """
    features = ["rem_percentage", "resting_hr", "hrv_sdnn"]

    detector = PopulationAnomalyDetector(
        features=features,
        z_threshold=2.0,
        min_features_trigger=2,
    )

    detector.fit(X)
    anomaly_flags, z_scores = detector.predict(X)

    return {
        "precision": precision_score(y, anomaly_flags),
        "recall": recall_score(y, anomaly_flags),
        "f1": f1_score(y, anomaly_flags),
        "flag_rate": anomaly_flags.mean(),
    }


def anomaly_sensitivity_analysis(X, y):
    """
    Sensitivity analysis: show how anomaly detection performance varies
    across different z-score thresholds and trigger counts.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Ground truth depression risk labels.
        
    Returns
    -------
    pd.DataFrame
        Results with varying thresholds and trigger counts.
    """
    results = []

    for z in [1.5, 2.0, 2.5]:
        for k in [1, 2]:
            detector = PopulationAnomalyDetector(
                features=["rem_percentage", "resting_hr", "hrv_sdnn"],
                z_threshold=z,
                min_features_trigger=k,
            )
            detector.fit(X)
            flags, _ = detector.predict(X)

            results.append(
                {
                    "z_threshold": z,
                    "min_features": k,
                    "precision": precision_score(y, flags),
                    "recall": recall_score(y, flags),
                    "f1": f1_score(y, flags),
                    "flag_rate": flags.mean(),
                }
            )

    return pd.DataFrame(results)


def train_final_xgb(X, y, seed=42):
    """
    Train XGBoost on the full dataset for interpretability analysis.
    
    Uses the same pipeline structure as cross-validation to ensure
    SHAP explanations are faithful to the evaluation setting.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target labels.
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    Pipeline
        Fitted pipeline with imputer, scaler, and XGBoost.
    """
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=seed,
                ),
            ),
        ]
    )

    pipeline.fit(X, y)
    return pipeline