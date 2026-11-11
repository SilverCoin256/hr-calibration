"""
SHAP bootstrap stability analysis.
Computes SSI (SHAP Stability Index) within confidence zones.
WARNING: computationally intensive (~15-30 min on laptop). Run once, cache results.
Saves results/shap_stability_results.json
"""
import json
import os
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config_loader import load_config
from data_pipeline_runner import get_data
from utils import compute_confidence

cfg = load_config()


def get_shap_importance(model, X, feature_names, model_name: str) -> np.ndarray:
    """
    Compute mean absolute SHAP values per feature.
    Returns 1D array of shape (n_features,).
    Handles both old (list) and new (ndarray) SHAP API output formats.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap package required: pip install shap")

    if model_name == "LR":
        masker = shap.maskers.Independent(X)
        explainer = shap.LinearExplainer(model, masker)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # Handle different SHAP return shapes across API versions
    if isinstance(shap_values, list):
        # Old API: list of per-class arrays; each is (n_samples, n_features)
        sv = np.array(shap_values[1])
    else:
        sv = np.array(shap_values)

    if sv.ndim == 3:
        # Two possible 3D layouts:
        #   (n_classes, n_samples, n_features) — old ndarray format
        #   (n_samples, n_features, n_classes) — new shap >= 0.42 format
        if sv.shape[0] == 2 and sv.shape[-1] != 2:
            # (2, n_samples, n_features) — take class-1 slice
            sv = sv[1]
        elif sv.shape[-1] == 2:
            # (n_samples, n_features, 2) — take class-1 along last axis
            sv = sv[:, :, 1]
        else:
            # Ambiguous; heuristic: last dim more likely n_classes
            sv = sv[:, :, 1] if sv.shape[-1] < sv.shape[0] else sv[1]

    # Now sv should be (n_samples, n_features)
    if sv.ndim != 2:
        raise ValueError(f"Unexpected SHAP array shape after slicing: {sv.shape}")

    return np.abs(sv).mean(axis=0)


def get_top_k_ranking(importance: np.ndarray, k: int = 5) -> np.ndarray:
    """Return indices of top-k features by importance (rank 0 = highest)."""
    return np.argsort(-importance)[:k]


def get_per_sample_shap(model, X, model_name: str) -> np.ndarray:
    """
    Compute per-sample SHAP values.
    Returns 2D array of shape (n_samples, n_features).
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap package required")

    if model_name == "LR":
        masker = shap.maskers.Independent(X)
        explainer = shap.LinearExplainer(model, masker)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1])
    else:
        sv = np.array(shap_values)

    if sv.ndim == 3:
        if sv.shape[0] == 2 and sv.shape[-1] != 2:
            sv = sv[1]
        elif sv.shape[-1] == 2:
            sv = sv[:, :, 1]
        else:
            sv = sv[:, :, 1] if sv.shape[-1] < sv.shape[0] else sv[1]

    return sv  # (n_samples, n_features)


def compute_per_sample_ssi(shap_matrix_list: list) -> tuple:
    """
    Compute per-sample SHAP Stability Index.

    For each sample, compute pairwise Spearman rho between its SHAP feature
    importance rankings across bootstrap runs. Average over all samples and pairs.

    shap_matrix_list: list of n_bootstrap arrays, each shape (n_samples, n_features).
    Returns (mean_ssi, std_ssi, n_pairs_evaluated).
    """
    n_boot = len(shap_matrix_list)
    if n_boot < 2:
        return np.nan, np.nan, 0

    n_samples = shap_matrix_list[0].shape[0]
    sample_rhos = []

    for s in range(n_samples):
        boot_rhos = []
        for b1 in range(n_boot):
            for b2 in range(b1 + 1, n_boot):
                v1 = np.abs(shap_matrix_list[b1][s])
                v2 = np.abs(shap_matrix_list[b2][s])
                # Need at least 2 unique values for Spearman to be defined
                if np.unique(v1).size < 2 or np.unique(v2).size < 2:
                    continue
                rho, _ = spearmanr(v1, v2)
                boot_rhos.append(float(rho))
        if boot_rhos:
            sample_rhos.append(float(np.mean(boot_rhos)))

    if not sample_rhos:
        return np.nan, np.nan, 0
    return float(np.mean(sample_rhos)), float(np.std(sample_rhos)), len(sample_rhos)


def bootstrap_shap_stability(model_class, model_params: dict, model_name: str,
                              X_train: np.ndarray, y_train: np.ndarray,
                              X_test_zone: np.ndarray,
                              n_bootstrap: int, seed_base: int) -> dict:
    """
    Bootstrap SHAP stability via per-sample Spearman correlation.
    For each bootstrap: refit model, compute individual SHAP for each test case.
    SSI = mean pairwise Spearman rho of per-sample SHAP rankings across bootstraps.
    """
    shap_matrices = []
    for b in range(n_bootstrap):
        rng = np.random.RandomState(seed_base + b)
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train[idx]
        y_boot = y_train[idx]
        model = model_class(**model_params)
        model.fit(X_boot, y_boot)
        sv = get_per_sample_shap(model, X_test_zone, model_name)
        shap_matrices.append(sv)
        if (b + 1) % 10 == 0:
            print(f"    Bootstrap {b+1}/{n_bootstrap}")

    ssi_mean, ssi_std, n_samples_eval = compute_per_sample_ssi(shap_matrices)
    return {
        "ssi_mean": ssi_mean,
        "ssi_std": ssi_std,
        "n_bootstrap": n_bootstrap,
        "n_samples_evaluated": n_samples_eval,
    }


MODEL_CLASSES = {
    "LR": (LogisticRegression, "LR_PARAMS"),
    "RF": (RandomForestClassifier, "RF_PARAMS_BOOT"),
    "GB": (GradientBoostingClassifier, "GB_PARAMS_BOOT"),
}


def run(calibration_results: dict, trained_models: dict):
    data = get_data()
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    seed_base = cfg["BOOTSTRAP_SEED"]
    n_bootstrap = cfg["N_SHAP_BOOTSTRAP"]
    all_results = {}

    for name in ["LR", "RF", "GB"]:
        print(f"\nSHAP stability for {name} (n_bootstrap={n_bootstrap})...")
        model = trained_models[name]
        best_prob = np.array(calibration_results[name]["best_prob_test"])
        confidence = compute_confidence(best_prob)

        if name == "LR":
            X_train = data["X_train_scaled"]
            X_test = data["X_test_scaled"]
        else:
            X_train = data["X_train_fit"]
            X_test = data["X_test"]
        y_train = data["y_train_fit"]

        # Define zones by confidence quartile on test set
        q25 = np.percentile(confidence, 25)
        q75 = np.percentile(confidence, 75)
        zone_a_mask = confidence < q25
        zone_d_mask = confidence >= q75

        print(f"  Zone A: n={zone_a_mask.sum()}, Zone D: n={zone_d_mask.sum()}")

        model_class, param_key = MODEL_CLASSES[name]
        model_params = cfg[param_key]

        # Base SHAP importance (full test set)
        print("  Computing base SHAP importance...")
        base_importance = get_shap_importance(model, X_test, feature_names, name)
        base_ranking = get_top_k_ranking(base_importance, k=5)
        top5_features = [feature_names[int(i)] for i in base_ranking]

        # Zone D stability
        print(f"  Zone D bootstrap ({zone_d_mask.sum()} test cases)...")
        X_test_zone_d = X_test[zone_d_mask]
        if len(X_test_zone_d) < 10:
            zone_d_result = {"ssi_mean": np.nan, "ssi_std": np.nan,
                             "note": "insufficient cases"}
        else:
            zone_d_result = bootstrap_shap_stability(
                model_class, model_params, name,
                X_train, y_train, X_test_zone_d,
                n_bootstrap, seed_base
            )

        # Zone A stability
        print(f"  Zone A bootstrap ({zone_a_mask.sum()} test cases)...")
        X_test_zone_a = X_test[zone_a_mask]
        if len(X_test_zone_a) < 10:
            zone_a_result = {"ssi_mean": np.nan, "ssi_std": np.nan,
                             "note": "insufficient cases"}
        else:
            zone_a_result = bootstrap_shap_stability(
                model_class, model_params, name,
                X_train, y_train, X_test_zone_a,
                n_bootstrap, seed_base + 10000
            )

        all_results[name] = {
            "base_top5_features": top5_features,
            "base_importance_scores": base_importance[base_ranking].tolist(),
            "zone_d": zone_d_result,
            "zone_a": zone_a_result,
            "ssi_zone_d": zone_d_result.get("ssi_mean"),
            "ssi_zone_a": zone_a_result.get("ssi_mean"),
            "ssi_zone_d_std": zone_d_result.get("ssi_std"),
            "ssi_zone_a_std": zone_a_result.get("ssi_std"),
            "zone_a_n": int(zone_a_mask.sum()),
            "zone_d_n": int(zone_d_mask.sum()),
        }
        print(f"  SSI Zone D={zone_d_result.get('ssi_mean', 'n/a'):.4f}, "
              f"Zone A={zone_a_result.get('ssi_mean', 'n/a'):.4f}")

    out_path = os.path.join(cfg["RESULTS_DIR"], "shap_stability_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSHAP stability results saved to {out_path}")
    return all_results


if __name__ == "__main__":
    calib_path = os.path.join(cfg["RESULTS_DIR"], "calibration_results.json")
    with open(calib_path) as f:
        calibration_results = json.load(f)
    from run_all import get_trained_models_and_probs
    trained_models, test_probs, _ = get_trained_models_and_probs()
    run(calibration_results, trained_models)
