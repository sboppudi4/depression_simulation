import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# If you already have the dataframe in memory in run.py,
# you can also save it to CSV and load it here.
# Otherwise, re-run the simulation to get X.

from simulator import simulate_dataset  # or whatever function generates X

# Generate one cohort (no labels needed for correlation)
X, y = simulate_dataset(prevalence=0.20)

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    X.corr(),
    annot=False,
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

# Save for paper
plt.savefig("fig1_correlation.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved correlation heatmap as fig1_correlation.png")
