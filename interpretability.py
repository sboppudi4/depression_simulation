import shap
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def compute_shap_values(pipeline, X):
    """
    Compute SHAP values for the trained XGBoost model.
    
    Uses TreeExplainer which is exact and fast for tree-based models.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline containing imputer, scaler, and XGBoost model.
    X : pd.DataFrame
        Input features (will be transformed by the pipeline).
        
    Returns
    -------
    shap_values : np.ndarray
        SHAP values for each sample and feature.
    X_transformed : np.ndarray
        Transformed feature matrix used for SHAP computation.
    """
    # Extract transformed features (imputer + scaler)
    X_transformed = pipeline[:-1].transform(X)

    # Get the XGBoost model
    xgb_model = pipeline[-1]
    
    # Compute SHAP values using TreeExplainer (exact for trees)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed)

    return shap_values, X_transformed


def global_feature_importance(shap_values, feature_names):
    """
    Compute global feature importance using mean absolute SHAP values.
    
    This is more faithful than XGBoost's built-in importance because it
    accounts for feature interactions and is instance-weighted.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values from compute_shap_values.
    feature_names : list
        Names of features corresponding to columns.
        
    Returns
    -------
    pd.DataFrame
        Features ranked by mean |SHAP|, descending.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values(by="mean_abs_shap", ascending=False)


def local_explanation(shap_values, X_transformed, feature_names, index):
    """
    Return SHAP values and feature values for a single instance.
    
    Useful for understanding individual predictions (e.g., failure cases).
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values from compute_shap_values.
    X_transformed : np.ndarray
        Transformed features.
    feature_names : list
        Names of features.
    index : int
        Index of the instance to explain.
        
    Returns
    -------
    pd.DataFrame
        Feature-level SHAP contributions for this instance.
    """
    return pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_values[index],
            "feature_value": X_transformed[index],
        }
    ).sort_values(by="shap_value", key=abs, ascending=False)
