"""
Train three baseline models (LR, RF, GB), compute standard discriminative metrics.
5-fold stratified CV + final test evaluation.
Saves results/baseline_results.json
"""
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score
)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config_loader import load_config
from data_pipeline_runner import get_data

cfg = load_config()


def build_models():
    return {
        "LR": LogisticRegression(**cfg["LR_PARAMS"]),
        "RF": RandomForestClassifier(**cfg["RF_PARAMS"]),
        "GB": GradientBoostingClassifier(**cfg["GB_PARAMS"]),
    }


def evaluate_model(model, X_train, y_train, X_test, y_test,
                   X_train_unscaled=None, X_test_unscaled=None, name=""):
    """
    5-fold CV + held-out test evaluation.
    LR uses scaled features; RF/GB use unscaled.
    """
    seed = cfg["RANDOM_SEED"]
    skf = StratifiedKFold(n_splits=cfg["N_FOLDS"], shuffle=True, random_state=seed)

    cv_scoring = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "f1": "f1",
        "average_precision": "average_precision",
    }
    cv_results = cross_validate(model, X_train, y_train, cv=skf,
                                scoring=cv_scoring, return_train_score=False)

    model.fit(X_train, y_train)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= 0.5).astype(int)

    result = {
        "cv_auc_mean": float(cv_results["test_roc_auc"].mean()),
        "cv_auc_std": float(cv_results["test_roc_auc"].std()),
        "cv_f1_mean": float(cv_results["test_f1"].mean()),
        "cv_f1_std": float(cv_results["test_f1"].std()),
        "cv_accuracy_mean": float(cv_results["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv_results["test_accuracy"].std()),
        "cv_avg_precision_mean": float(cv_results["test_average_precision"].mean()),
        "cv_avg_precision_std": float(cv_results["test_average_precision"].std()),
        "test_auc": float(roc_auc_score(y_test, y_prob_test)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "test_precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "test_avg_precision": float(average_precision_score(y_test, y_prob_test)),
    }
    return result, model, y_prob_test


def run():
    data = get_data()
    X_train_fit = data["X_train_fit"]
    X_test = data["X_test"]
    y_train_fit = data["y_train_fit"]
    y_test = data["y_test"]
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]

    models = build_models()
    all_results = {}
    trained_models = {}
    test_probs = {}

    for name, model in models.items():
        print(f"Training {name}...")
        if name == "LR":
            result, fitted_model, y_prob = evaluate_model(
                model, X_train_scaled, y_train_fit, X_test_scaled, y_test, name=name
            )
        else:
            result, fitted_model, y_prob = evaluate_model(
                model, X_train_fit, y_train_fit, X_test, y_test, name=name
            )
        all_results[name] = result
        trained_models[name] = fitted_model
        test_probs[name] = y_prob
        print(f"  {name}: AUC={result['test_auc']:.4f}, F1={result['test_f1']:.4f}, "
              f"Acc={result['test_accuracy']:.4f}")

    out_path = os.path.join(cfg["RESULTS_DIR"], "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBaseline results saved to {out_path}")

    return trained_models, test_probs, all_results


if __name__ == "__main__":
    run()
