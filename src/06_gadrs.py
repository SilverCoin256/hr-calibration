"""
GADRS: Governance-Aware Deployment Readiness Score.
Aggregates ECE_calibrated, AURC_delta, LowConf_ECE, SSI_lowconf.
Includes sensitivity analysis and covariate shift test.
Saves results/gadrs_results.json
"""
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config_loader import load_config
from data_pipeline_runner import get_data
from utils import compute_ece, compute_aurc, coverage_accuracy_curve, compute_random_aurc
from importlib import import_module

cfg = load_config()
MODELS = ["LR", "RF", "GB"]


def extract_sub_scores(calib: dict, selective: dict, shap: dict) -> dict:
    """Extract the four raw sub-scores for each model."""
    scores = {}
    for name in MODELS:
        ece_calibrated = calib[name]["ece_calibrated"]
        aurc_delta = selective[name]["aurc_delta"]
        lowconf_ece = calib[name].get("zone_a_ece") or calib[name]["zone_ece"][0]["ece"]
        ssi_lowconf = shap[name].get("ssi_zone_a", np.nan)
        if ssi_lowconf is None:
            ssi_lowconf = np.nan
        scores[name] = {
            "ece_calibrated": float(ece_calibrated),
            "aurc_delta": float(aurc_delta),
            "lowconf_ece": float(lowconf_ece) if lowconf_ece is not None else np.nan,
            "ssi_lowconf": float(ssi_lowconf),
        }
    return scores


def minmax_normalize(values: dict) -> dict:
    """Min-max normalize dict of {model: value} across models. NaN handled."""
    v_arr = np.array([values[m] for m in MODELS], dtype=float)
    valid = v_arr[~np.isnan(v_arr)]
    if len(valid) == 0 or valid.max() == valid.min():
        return {m: 0.5 for m in MODELS}
    vmin, vmax = valid.min(), valid.max()
    return {m: float((values[m] - vmin) / (vmax - vmin)) if not np.isnan(values[m]) else 0.5
            for m in MODELS}


def compute_gadrs(sub_scores: dict, weights: dict) -> dict:
    """Compute GADRS_risk and GADRS_readiness for each model."""
    # Normalize each sub-score across models
    ece_norm = minmax_normalize({m: sub_scores[m]["ece_calibrated"] for m in MODELS})
    aurc_norm = minmax_normalize({m: sub_scores[m]["aurc_delta"] for m in MODELS})
    lowconf_norm = minmax_normalize({m: sub_scores[m]["lowconf_ece"] for m in MODELS})
    ssi_norm = minmax_normalize({m: sub_scores[m]["ssi_lowconf"] for m in MODELS})

    results = {}
    for name in MODELS:
        gadrs_risk = (
            weights["ece_calibrated"] * ece_norm[name]
            + weights["aurc_delta"] * (1.0 - aurc_norm[name])   # higher aurc_delta = lower risk
            + weights["lowconf_ece"] * lowconf_norm[name]
            + weights["ssi_lowconf"] * (1.0 - ssi_norm[name])   # higher SSI = lower risk
        )
        gadrs_readiness = 1.0 - gadrs_risk
        results[name] = {
            "gadrs_risk": float(gadrs_risk),
            "gadrs_readiness": float(gadrs_readiness),
            "ece_norm": float(ece_norm[name]),
            "aurc_norm": float(aurc_norm[name]),
            "lowconf_ece_norm": float(lowconf_norm[name]),
            "ssi_norm": float(ssi_norm[name]),
        }
    return results


def covariate_shift_test(trained_models: dict, calibration_results: dict,
                          selective_results: dict, calib_mod) -> dict:
    """
    Add Gaussian noise to top-N SHAP-important features, recompute ECE + AURC.
    Returns DELTA GADRS_risk per model.
    """
    data = get_data()
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    shift_scale = cfg["SHIFT_NOISE_SCALE"]
    n_top = cfg["N_TOP_SHAP_FOR_SHIFT"]
    rng = np.random.RandomState(cfg["RANDOM_SEED"] + 99)

    shift_results = {}
    for name in MODELS:
        # Get top SHAP features from shap_stability base ranking
        shap_path = os.path.join(cfg["RESULTS_DIR"], "shap_stability_results.json")
        if not os.path.exists(shap_path):
            shift_results[name] = {"note": "shap_stability_results.json not found; shift test skipped"}
            continue
        with open(shap_path) as f:
            shap_data = json.load(f)
        top_features = shap_data[name].get("base_top5_features", [])[:n_top]
        top_indices = [feature_names.index(f) for f in top_features if f in feature_names]

        if name == "LR":
            X_test = data["X_test_scaled"].copy()
        else:
            X_test = data["X_test"].copy().astype(float)

        # Add Gaussian noise to top features
        for feat_idx in top_indices:
            feat_std = float(np.std(X_test[:, feat_idx]))
            noise = rng.normal(0, shift_scale * feat_std, size=len(X_test))
            X_test[:, feat_idx] = X_test[:, feat_idx] + noise

        # Get shifted probabilities
        model = trained_models[name]
        shifted_prob = model.predict_proba(X_test)[:, 1]

        # Re-apply best calibration method
        best_method = calibration_results[name]["best_method"]
        if name == "LR":
            X_cal = data["X_cal_scaled"]
        else:
            X_cal = data["X_cal"]
        y_cal = data["y_cal"]
        raw_prob_cal = model.predict_proba(X_cal)[:, 1]

        if best_method == "platt":
            from sklearn.linear_model import LogisticRegression as LR2
            lr_cal = LR2(C=1e10, solver="lbfgs", max_iter=1000)
            lr_cal.fit(raw_prob_cal.reshape(-1, 1), y_cal)
            shifted_prob_cal = lr_cal.predict_proba(shifted_prob.reshape(-1, 1))[:, 1]
        else:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_prob_cal, y_cal)
            shifted_prob_cal = iso.predict(shifted_prob)

        # Recompute ECE and AURC
        ece_shifted = compute_ece(y_test, shifted_prob_cal)
        coverages, accuracies = coverage_accuracy_curve(y_test, shifted_prob_cal)
        aurc_shifted = compute_aurc(coverages, accuracies)
        aurc_random = compute_random_aurc(y_test, shifted_prob_cal)
        aurc_delta_shifted = float(aurc_random - aurc_shifted)

        shift_results[name] = {
            "ece_calibrated_shifted": float(ece_shifted),
            "aurc_delta_shifted": float(aurc_delta_shifted),
            "perturbed_features": top_features,
        }

    return shift_results


def run(calibration_results: dict, selective_results: dict,
        shap_results: dict, trained_models: dict):

    sub_scores = extract_sub_scores(calibration_results, selective_results, shap_results)
    print("Sub-scores extracted:")
    for name in MODELS:
        s = sub_scores[name]
        print(f"  {name}: ECE_cal={s['ece_calibrated']:.4f}, AURC_delta={s['aurc_delta']:.4f}, "
              f"LowConf_ECE={s['lowconf_ece']:.4f}, SSI_low={s['ssi_lowconf']:.4f}")

    # Primary weights
    gadrs_primary = compute_gadrs(sub_scores, cfg["GADRS_WEIGHTS"])
    # Sensitivity: equal weights
    gadrs_equal = compute_gadrs(sub_scores, cfg["GADRS_WEIGHTS_EQUAL"])
    # Sensitivity: ECE-heavy
    gadrs_ece_heavy = compute_gadrs(sub_scores, cfg["GADRS_WEIGHTS_ECE_HEAVY"])

    print("\nGADRS Readiness Scores:")
    for name in MODELS:
        print(f"  {name}: primary={gadrs_primary[name]['gadrs_readiness']:.4f}, "
              f"equal={gadrs_equal[name]['gadrs_readiness']:.4f}, "
              f"ece_heavy={gadrs_ece_heavy[name]['gadrs_readiness']:.4f}")

    # Covariate shift test
    print("\nRunning covariate shift test...")
    import importlib
    calib_mod = importlib.import_module("03_calibration_analysis")
    shift_results = covariate_shift_test(trained_models, calibration_results,
                                          selective_results, calib_mod)

    # Compute shift GADRS for models where shift succeeded
    shift_sub_scores = {}
    for name in MODELS:
        if "ece_calibrated_shifted" in shift_results.get(name, {}):
            shift_sub_scores[name] = {
                "ece_calibrated": shift_results[name]["ece_calibrated_shifted"],
                "aurc_delta": shift_results[name]["aurc_delta_shifted"],
                "lowconf_ece": sub_scores[name]["lowconf_ece"],  # not recomputed
                "ssi_lowconf": sub_scores[name]["ssi_lowconf"],  # not recomputed
            }
        else:
            shift_sub_scores[name] = sub_scores[name]

    gadrs_shifted = compute_gadrs(shift_sub_scores, cfg["GADRS_WEIGHTS"])

    delta_gadrs = {
        name: float(gadrs_shifted[name]["gadrs_risk"] - gadrs_primary[name]["gadrs_risk"])
        for name in MODELS
    }

    final_results = {
        "sub_scores": sub_scores,
        "gadrs": {
            "primary": gadrs_primary,
            "equal_weights": gadrs_equal,
            "ece_heavy": gadrs_ece_heavy,
        },
        "covariate_shift": shift_results,
        "gadrs_shifted": gadrs_shifted,
        "delta_gadrs_risk": delta_gadrs,
    }

    out_path = os.path.join(cfg["RESULTS_DIR"], "gadrs_results.json")
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nGADRS results saved to {out_path}")
    return final_results


if __name__ == "__main__":
    results_dir = cfg["RESULTS_DIR"]
    for fname, key in [
        ("calibration_results.json", None),
        ("selective_prediction_results.json", None),
        ("shap_stability_results.json", None),
    ]:
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required: {path}. Run earlier pipeline steps first.")
    with open(os.path.join(results_dir, "calibration_results.json")) as f:
        calibration_results = json.load(f)
    with open(os.path.join(results_dir, "selective_prediction_results.json")) as f:
        selective_results = json.load(f)
    with open(os.path.join(results_dir, "shap_stability_results.json")) as f:
        shap_results = json.load(f)
    from run_all import get_trained_models_and_probs
    trained_models, _, _ = get_trained_models_and_probs()
    run(calibration_results, selective_results, shap_results, trained_models)
