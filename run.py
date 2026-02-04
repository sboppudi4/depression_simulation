from simulator import simulate_dataset
from experiments import cross_validate_models
from experiments import add_noise, feature_dropout
from experiments import evaluate_anomaly_detector, anomaly_sensitivity_analysis
from experiments import train_final_xgb
from interpretability import compute_shap_values, global_feature_importance, local_explanation
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# Base evaluation
X, y = simulate_dataset()
base_results = cross_validate_models(X, y)
print("\nBase Results\n", base_results)

# Noise stress test (now using 0.5 default)
X_noise = add_noise(X)
noise_results = cross_validate_models(X_noise, y)
print("\nNoise Stress Test\n", noise_results)

# Dropout stress test (now using 0.6 default)
X_dropout = feature_dropout(X)
dropout_results = cross_validate_models(X_dropout, y)
print("\nFeature Dropout Test\n", dropout_results)

# Anomaly detection evaluation
print("\n" + "="*60)
print("ANOMALY DETECTION EVALUATION")
print("="*60)
anomaly_metrics = evaluate_anomaly_detector(X, y)
print("\nPerformance Metrics:")
for metric, value in anomaly_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Sensitivity analysis
print("\n" + "="*60)
print("ANOMALY SENSITIVITY ANALYSIS")
print("="*60)
sensitivity = anomaly_sensitivity_analysis(X, y)
print("\n", sensitivity)

print(f"\nPrevalence (y.mean()): {y.mean():.4f}")

# SHAP Interpretability
print("\n" + "="*60)
print("SHAP INTERPRETABILITY")
print("="*60)

# Train final XGBoost model on full dataset
xgb_pipeline = train_final_xgb(X, y)
print("\nTrained XGBoost pipeline for interpretability analysis.")

# Compute SHAP values
shap_values, X_transformed = compute_shap_values(xgb_pipeline, X)
feature_names = X.columns.tolist()

# Global feature importance
global_shap = global_feature_importance(shap_values, feature_names)
print("\nGlobal Feature Importance (Mean |SHAP|):")
print(global_shap.head(10))

# Failure Analysis
print("\n" + "="*60)
print("FAILURE ANALYSIS")
print("="*60)

# Get predictions
y_prob = xgb_pipeline.predict_proba(X)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(f"                Predicted Neg  Predicted Pos")
print(f"Actual Neg      {cm[0,0]:>13}  {cm[0,1]:>13}")
print(f"Actual Pos      {cm[1,0]:>13}  {cm[1,1]:>13}")

# Analyze false negatives (missed at-risk cases)
false_negatives = np.where((y == 1) & (y_pred == 0))[0]
false_positives = np.where((y == 0) & (y_pred == 1))[0]

print(f"\nFalse Negatives (missed at-risk): {len(false_negatives)}")
print(f"False Positives (false alarms): {len(false_positives)}")

# Example false negative analysis
if len(false_negatives) > 0:
    idx = false_negatives[0]
    print(f"\n--- Example False Negative (Index {idx}) ---")
    print(f"True Label: {y[idx]}, Predicted Prob: {y_prob[idx]:.4f}")
    print("\nSHAP Contributions (sorted by magnitude):")
    local_shap = local_explanation(shap_values, X_transformed, feature_names, idx)
    print(local_shap.head(10))
    print("\nInterpretation: Features near population mean → model misses subtle risk.")

# Example false positive analysis
if len(false_positives) > 0:
    idx = false_positives[0]
    print(f"\n--- Example False Positive (Index {idx}) ---")
    print(f"True Label: {y[idx]}, Predicted Prob: {y_prob[idx]:.4f}")
    print("\nSHAP Contributions (sorted by magnitude):")
    local_shap = local_explanation(shap_values, X_transformed, feature_names, idx)
    print(local_shap.head(10))
    print("\nInterpretation: Extreme physiology without underlying risk (confounder effect).")


