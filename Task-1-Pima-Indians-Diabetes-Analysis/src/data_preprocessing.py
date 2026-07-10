"""
data_preprocessing.py
----------------------
Loads the Pima Indians Diabetes dataset, cleans it, and prepares
train/test splits ready for model training.

The raw dataset encodes missing clinical measurements as 0 for the
columns Glucose, BloodPressure, SkinThickness, Insulin and BMI (a
value of 0 is not physiologically possible for these features). This
script replaces those zeros with NaN and imputes them using the
median of each column, then scales all features with StandardScaler.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Columns where a value of 0 actually means "missing data"
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace implausible zeros with NaN and impute with column median."""
    df = df.copy()
    for col in ZERO_AS_MISSING_COLS:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    return df


def get_feature_target_split(df: pd.DataFrame):
    """Separate features (X) from the target label (y)."""
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return X, y


def preprocess(path: str = RAW_DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline:
      1. Load raw data
      2. Clean implausible zero values
      3. Split into train/test sets
      4. Scale features using StandardScaler (fit on train only)

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    df = load_data(path)
    df = clean_data(df)
    X, y = get_feature_target_split(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")
    print(f"Features: {feature_names}")
    print(f"Outcome distribution (train):\n{y_train.value_counts(normalize=True)}")
