"""
Data pipeline: load, preprocess, split IBM HR Analytics dataset.
Saves processed arrays and a manifest JSON for reproducibility.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config_loader import load_config

cfg = load_config()


def load_and_preprocess():
    """Load IBM HR dataset, drop constants, encode categoricals."""
    df = pd.read_csv(cfg["DATA_PATH"])

    # Drop constant / identifier columns (no predictive value)
    drop_cols = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode binary outcome
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    # Binary encode two binary features
    df["Gender"] = (df["Gender"] == "Male").astype(int)
    df["OverTime"] = (df["OverTime"] == "Yes").astype(int)

    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    y = df["Attrition"].values
    X = df.drop(columns=["Attrition"]).values
    feature_names = [c for c in df.columns if c != "Attrition"]

    return X, y, feature_names, df


def split_data(X, y):
    """
    Three-way split:
      - test: 20% of total (stratified, seed=42)
      - calibration hold: 20% of train (for Platt/isotonic fitting)
      - train_fit: remaining 80% of train

    Returns:
        X_train_fit, X_cal, X_test, y_train_fit, y_cal, y_test
    """
    seed = cfg["RANDOM_SEED"]
    test_size = cfg["TEST_SIZE"]
    cal_split = cfg["CALIBRATION_SPLIT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
        X_train, y_train,
        test_size=cal_split,
        stratify=y_train,
        random_state=seed + 1
    )
    return X_train_fit, X_cal, X_test, y_train_fit, y_cal, y_test


def scale_for_lr(X_train_fit, X_cal, X_test):
    """Apply StandardScaler for Logistic Regression (fit on train_fit only)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fit)
    X_cal_scaled = scaler.transform(X_cal)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_cal_scaled, X_test_scaled, scaler


def run_pipeline():
    X, y, feature_names, df = load_and_preprocess()
    X_train_fit, X_cal, X_test, y_train_fit, y_cal, y_test = split_data(X, y)
    X_train_scaled, X_cal_scaled, X_test_scaled, scaler = scale_for_lr(
        X_train_fit, X_cal, X_test
    )

    manifest = {
        "n_total": int(len(y)),
        "n_train_fit": int(len(y_train_fit)),
        "n_cal": int(len(y_cal)),
        "n_test": int(len(y_test)),
        "attrition_rate_total": float(y.mean()),
        "attrition_rate_train_fit": float(y_train_fit.mean()),
        "attrition_rate_test": float(y_test.mean()),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "random_seed": cfg["RANDOM_SEED"],
    }

    out_path = os.path.join(cfg["RESULTS_DIR"], "data_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Data manifest saved to {out_path}")
    print(f"  Total: N={manifest['n_total']}, attrition={manifest['attrition_rate_total']:.3f}")
    print(f"  Train fit: {manifest['n_train_fit']}, Cal: {manifest['n_cal']}, Test: {manifest['n_test']}")
    print(f"  Features: {manifest['n_features']}")

    return (
        X_train_fit, X_cal, X_test,
        y_train_fit, y_cal, y_test,
        X_train_scaled, X_cal_scaled, X_test_scaled,
        feature_names, manifest
    )


if __name__ == "__main__":
    run_pipeline()
