"""
Calibration analysis: ECE, MCE, Platt scaling, isotonic regression, zone-level calibration.
Saves results/calibration_results.json
"""
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config_loader import load_config
from data_pipeline_runner import get_data
from utils import compute_ece, compute_mce, compute_reliability_bins, compute_zone_ece

cfg = load_config()


def platt_calibrate(y_prob_cal: np.ndarray, y_cal: np.ndarray,
                    y_prob_test: np.ndarray) -> np.ndarray:
    """
    Fit logistic regression on calibration hold-out raw scores.
    Apply to test scores.
    """
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(y_prob_cal.reshape(-1, 1), y_cal)
    return lr.predict_proba(y_prob_test.reshape(-1, 1))[:, 1]


def isotonic_calibrate(y_prob_cal: np.ndarray, y_cal: np.ndarray,
                       y_prob_test: np.ndarray) -> np.ndarray:
    """
    Fit isotonic regression on calibration hold-out scores.
    Apply to test scores, clipping to [0, 1].
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob_cal, y_cal)
    return iso.predict(y_prob_test)


def analyze_model_calibration(name: str, model, data: dict, raw_prob_test: np.ndarray) -> dict:
    """
    Full calibration analysis for one model.
    Returns nested result dict.
    """
    cfg_local = cfg
    y_cal = data["y_cal"]
    y_test = data["y_test"]

    # Get calibration hold-out raw probabilities
    if name == "LR":
        raw_prob_cal = model.predict_proba(data["X_cal_scaled"])[:, 1]
    else:
        raw_prob_cal = model.predict_proba(data["X_cal"])[:, 1]

    n_bins = cfg_local["ECE_BINS"]

    # --- Raw calibration ---
    ece_raw = compute_ece(y_test, raw_prob_test, n_bins=n_bins)
    mce_raw = compute_mce(y_test, raw_prob_test, n_bins=n_bins)
    reliability_raw = compute_reliability_bins(y_test, raw_prob_test, n_bins=n_bins)

    # --- Platt calibration ---
    platt_prob_test = platt_calibrate(raw_prob_cal, y_cal, raw_prob_test)
    ece_platt = compute_ece(y_test, platt_prob_test, n_bins=n_bins)
    mce_platt = compute_mce(y_test, platt_prob_test, n_bins=n_bins)
    reliability_platt = compute_reliability_bins(y_test, platt_prob_test, n_bins=n_bins)

    # --- Isotonic calibration ---
    iso_prob_test = isotonic_calibrate(raw_prob_cal, y_cal, raw_prob_test)
    ece_iso = compute_ece(y_test, iso_prob_test, n_bins=n_bins)
    mce_iso = compute_mce(y_test, iso_prob_test, n_bins=n_bins)
    reliability_iso = compute_reliability_bins(y_test, iso_prob_test, n_bins=n_bins)

    # --- Select best calibration method (including raw as candidate) ---
    candidates = {
        "raw": (ece_raw, mce_raw, raw_prob_test),
        "platt": (ece_platt, mce_platt, platt_prob_test),
        "isotonic": (ece_iso, mce_iso, iso_prob_test),
    }
    best_method = min(candidates, key=lambda k: candidates[k][0])
    ece_best, mce_best, best_prob_test = candidates[best_method]

    # --- Zone-level calibration (using best method probabilities) ---
    zone_ece = compute_zone_ece(y_test, best_prob_test, n_zones=4, n_bins=n_bins)

    # --- Alternative bin counts robustness ---
    ece_5bins = compute_ece(y_test, best_prob_test, n_bins=5)
    ece_15bins = compute_ece(y_test, best_prob_test, n_bins=15)

    return {
        "raw": {
            "ece": float(ece_raw), "mce": float(mce_raw),
            "reliability_bins": reliability_raw
        },
        "platt": {
            "ece": float(ece_platt), "mce": float(mce_platt),
            "reliability_bins": reliability_platt
        },
        "isotonic": {
            "ece": float(ece_iso), "mce": float(mce_iso),
            "reliability_bins": reliability_iso
        },
        "best_method": best_method,
        "ece_calibrated": float(ece_best),
        "mce_calibrated": float(mce_best),
        "zone_ece": zone_ece,
        "zone_a_ece": float(zone_ece[0]["ece"]) if zone_ece else None,
        "zone_d_ece": float(zone_ece[3]["ece"]) if zone_ece else None,
        "ece_5bins": float(ece_5bins),
        "ece_15bins": float(ece_15bins),
        "best_prob_test": best_prob_test.tolist(),
    }


def run(trained_models: dict, test_probs: dict):
    data = get_data()
    all_results = {}

    for name in ["LR", "RF", "GB"]:
        print(f"Calibration analysis for {name}...")
        result = analyze_model_calibration(
            name, trained_models[name], data, np.array(test_probs[name])
        )
        all_results[name] = result
        print(f"  ECE raw={result['raw']['ece']:.4f} | "
              f"Platt={result['platt']['ece']:.4f} | "
              f"Isotonic={result['isotonic']['ece']:.4f} | "
              f"Best ({result['best_method']})={result['ece_calibrated']:.4f}")
        print(f"  Zone A ECE={result['zone_a_ece']:.4f}, Zone D ECE={result['zone_d_ece']:.4f}")

    out_path = os.path.join(cfg["RESULTS_DIR"], "calibration_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCalibration results saved to {out_path}")
    return all_results


if __name__ == "__main__":
    from run_all import get_trained_models_and_probs
    trained_models, test_probs, _ = get_trained_models_and_probs()
    run(trained_models, test_probs)
