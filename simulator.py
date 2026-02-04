import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_latent_labels(n_samples=1000, prevalence=0.15):
    """
    Latent depression risk variable.
    D = 1 -> at risk
    D = 0 -> no risk
    """
    return np.random.binomial(1, prevalence, size=n_samples)


def generate_features(D):
    """
    Generate wearable-derived features conditionally on latent variable D.
    """
    n = len(D)

    data = {}

    # ---- Sleep features ----
    data["total_sleep_time"] = np.where(
        D == 0,
        np.random.normal(410, 60, n),
        np.random.normal(370, 70, n),
    )

    data["rem_percentage"] = np.where(
        D == 0,
        np.random.normal(21, 5, n),
        np.random.normal(17, 6, n),
    )

    data["awakenings"] = np.where(
        D == 0,
        poisson.rvs(4, size=n),
        poisson.rvs(6, size=n),
    )

    data["sleep_latency"] = np.where(
        D == 0,
        np.random.normal(20, 12, n),
        np.random.normal(30, 18, n),
    )

    # ---- Physiological features ----
    data["resting_hr"] = np.where(
        D == 0,
        np.random.normal(65, 8, n),
        np.random.normal(72, 10, n),
    )

    data["hrv_sdnn"] = np.where(
        D == 0,
        np.random.normal(50, 12, n),
        np.random.normal(40, 14, n),
    )

    # ---- Behavioral features ----
    data["steps"] = np.where(
        D == 0,
        np.random.normal(7000, 2500, n),
        np.random.normal(4500, 2500, n),
    )

    # ---- Affective (ordinal) ----
    data["stress"] = np.where(
        D == 0,
        np.random.choice([1, 2, 3, 4], size=n, p=[0.35, 0.30, 0.25, 0.10]),
        np.random.choice([2, 3, 4, 5], size=n, p=[0.20, 0.30, 0.30, 0.20]),
    )

    data["anxiety"] = np.where(
        D == 0,
        np.random.choice([1, 2, 3, 4], size=n, p=[0.40, 0.30, 0.20, 0.10]),
        np.random.choice([2, 3, 4, 5], size=n, p=[0.25, 0.30, 0.25, 0.20]),
    )

    # clip to valid range [0, 5]
    data["stress"] = np.clip(data["stress"], 0, 5)
    data["anxiety"] = np.clip(data["anxiety"], 0, 5)

    # FIX 2: Add confounder (fitness level)
    # This creates realistic scenarios where:
    # - High-risk people can still look healthy (if fit)
    # - Low-risk people can look bad (if unfit)
    fitness = np.random.normal(0, 1, n)
    
    data["resting_hr"] -= 3 * fitness
    data["hrv_sdnn"] += 5 * fitness
    data["steps"] += 1500 * fitness

    return pd.DataFrame(data)


def inject_correlations(df):
    """
    Inject clinically plausible correlations using linear transformations.
    (Simple and reviewer-safe; copula can be added later.)
    """
    df = df.copy()

    # HRV negatively correlated with HR
    df["hrv_sdnn"] -= 0.15 * (df["resting_hr"] - df["resting_hr"].mean())

    # REM negatively correlated with awakenings
    df["rem_percentage"] -= 0.5 * (df["awakenings"] - df["awakenings"].mean())

    # Stress positively correlated with HR
    df["resting_hr"] += 1.2 * (df["stress"] - df["stress"].mean())

    return df


def inject_noise_and_missingness(df, missing_rate=0.1):
    """
    Add asymmetric sensor noise and missingness.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        std = df[col].std()

        # Asymmetric, feature-dependent noise
        if col in ["hrv_sdnn", "steps"]:
            df[col] += np.random.normal(0, 0.20 * std, len(df))
        else:
            df[col] += np.random.normal(0, 0.07 * std, len(df))

    # Missingness (realistic: HRV & steps often missing)
    for col in ["hrv_sdnn", "steps"]:
        mask = np.random.rand(len(df)) < missing_rate
        df.loc[mask, col] = np.nan

    return df


def simulate_dataset(
    n_samples=1000,
    prevalence=0.15,
    missing_rate=0.1,
    label_noise=0.07,
):
    """
    End-to-end simulator with:
    - latent variable labels
    - overlapping feature distributions
    - asymmetric sensor noise
    - realistic missingness
    - label noise
    """

    D = generate_latent_labels(n_samples, prevalence)
    X = generate_features(D)
    X = inject_correlations(X)
    X = inject_noise_and_missingness(X, missing_rate=missing_rate)

    y = inject_label_noise(D, noise_rate=label_noise)

    return X, y

def inject_label_noise(y, noise_rate=0.07):
    """
    Flip a fraction of labels to simulate diagnostic / reporting noise.
    """
    y_noisy = y.copy()
    flip = np.random.rand(len(y)) < noise_rate
    y_noisy[flip] = 1 - y_noisy[flip]
    return y_noisy