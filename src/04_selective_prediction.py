"""
Selective prediction: coverage-accuracy curves, AURC, governance threshold analysis.
Uses best-calibrated probabilities from calibration_results.json.
Saves results/selective_prediction_results.json
"""
import json
import os
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config_loader import load_config
from data_pipeline_runner import get_data
from utils import (
    coverage_accuracy_curve, compute_aurc, compute_random_aurc,
    compute_confidence, compute_ece, bootstrap_ci
)

cfg = load_config()


def governance_threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray,
                                   threshold: float) -> dict:
    """
    At a given confidence threshold, compute abstention zone statistics.

    Args:
        y_true: ground truth labels
        y_prob: predicted probabilities (best-calibrated)
        threshold: confidence threshold; abstain if max(p, 1-p) < threshold

    Returns dict with abstention_rate, hc_accuracy, hc_ece, hc_n,
    abstain_accuracy, abstain_n, hc_attrition_rate, abstain_attrition_rate
    """
    confidence = compute_confidence(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    hc_mask = confidence >= threshold    # high-confidence (predict)
    ab_mask = ~hc_mask                   # abstention zone

    n_total = len(y_true)
    n_hc = int(hc_mask.sum())
    n_ab = int(ab_mask.sum())

    abstention_rate = float(n_ab / n_total)

    if n_hc == 0:
        hc_accuracy = np.nan
        hc_ece = np.nan
    else:
        hc_accuracy = float((y_true[hc_mask] == y_pred[hc_mask]).mean())
        hc_ece = float(compute_ece(y_true[hc_mask], y_prob[hc_mask]))

    if n_ab == 0:
        ab_accuracy = np.nan
    else:
        ab_accuracy = float((y_true[ab_mask] == y_pred[ab_mask]).mean())

    return {
        "threshold": float(threshold),
        "abstention_rate": abstention_rate,
        "hc_n": n_hc,
        "abstain_n": n_ab,
        "hc_accuracy": hc_accuracy,
        "hc_ece": hc_ece,
        "abstain_accuracy": ab_accuracy,
        "hc_attrition_rate": float(y_true[hc_mask].mean()) if n_hc > 0 else np.nan,
        "abstain_attrition_rate": float(y_true[ab_mask].mean()) if n_ab > 0 else np.nan,
    }


def run(calibration_results: dict):
    data = get_data()
    y_test = data["y_test"]
    all_results = {}

    for name in ["LR", "RF", "GB"]:
        print(f"Selective prediction for {name}...")
        best_prob = np.array(calibration_results[name]["best_prob_test"])

        # Coverage-accuracy curve
        coverages, accuracies = coverage_accuracy_curve(y_test, best_prob, n_steps=50)

        # AURC
        aurc_model = compute_aurc(coverages, accuracies)
        aurc_random = compute_random_aurc(y_test, best_prob)
        aurc_delta = float(aurc_random - aurc_model)

        # Bootstrap CI for AURC delta
        rng = np.random.RandomState(cfg["BOOTSTRAP_SEED"])
        aurc_deltas_boot = []
        n = len(y_test)
        for _ in range(min(200, cfg["N_BOOTSTRAP_CI"])):
            idx = rng.choice(n, size=n, replace=True)
            c_boot, a_boot = coverage_accuracy_curve(y_test[idx], best_prob[idx], n_steps=50)
            aurc_rand_boot = compute_random_aurc(y_test[idx], best_prob[idx])
            aurc_boot = compute_aurc(c_boot, a_boot)
            aurc_deltas_boot.append(aurc_rand_boot - aurc_boot)
        aurc_delta_ci_low = float(np.percentile(aurc_deltas_boot, 2.5))
        aurc_delta_ci_high = float(np.percentile(aurc_deltas_boot, 97.5))

        # Governance threshold analysis
        thresholds = {
            "gov_conservative": cfg["GOV_CONSERVATIVE"],
            "gov_moderate": cfg["GOV_MODERATE"],
            "gov_strict": cfg["GOV_STRICT"],
        }
        threshold_results = {}
        for thresh_name, thresh_val in thresholds.items():
            threshold_results[thresh_name] = governance_threshold_analysis(
                y_test, best_prob, thresh_val
            )

        model_result = {
            "coverage_accuracy_curve": {
                "coverages": coverages.tolist(),
                "accuracies": accuracies.tolist(),
            },
            "aurc_model": float(aurc_model),
            "aurc_random": float(aurc_random),
            "aurc_delta": float(aurc_delta),
            "aurc_delta_ci_low": aurc_delta_ci_low,
            "aurc_delta_ci_high": aurc_delta_ci_high,
            "threshold_analysis": threshold_results,
        }
        all_results[name] = model_result

        print(f"  AURC_delta={aurc_delta:.4f} [CI: {aurc_delta_ci_low:.4f}, {aurc_delta_ci_high:.4f}]")
        gov_m = threshold_results["gov_moderate"]
        print(f"  Gov-M: abstention={gov_m['abstention_rate']:.3f}, "
              f"HC accuracy={gov_m['hc_accuracy']:.4f}, "
              f"Abstain accuracy={gov_m['abstain_accuracy']:.4f}")

    out_path = os.path.join(cfg["RESULTS_DIR"], "selective_prediction_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSelective prediction results saved to {out_path}")
    return all_results


if __name__ == "__main__":
    calib_path = os.path.join(cfg["RESULTS_DIR"], "calibration_results.json")
    with open(calib_path) as f:
        calibration_results = json.load(f)
    run(calibration_results)
