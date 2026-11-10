"""
Shared utility functions: ECE, MCE, bootstrap CI, confidence, coverage-accuracy, AURC.
No model training here. Pure statistical computations.
"""
import numpy as np
from typing import Tuple, List


def compute_confidence(y_prob: np.ndarray) -> np.ndarray:
    """
    Confidence = max(p, 1-p) for each prediction.
    Measures certainty about the predicted class (not about the positive class specifically).
    Range: [0.50, 1.00]
    """
    return np.maximum(y_prob, 1.0 - y_prob)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error using equal-width bins.
    Formula from Guo et al. (2017): ECE = sum_b (|b|/N) * |acc(b) - conf(b)|
    where acc(b) = fraction positive in bin, conf(b) = mean predicted probability.

    Args:
        y_true: Binary ground truth labels (0/1), shape (N,)
        y_prob: Predicted probabilities of positive class, shape (N,)
        n_bins: Number of equal-width calibration bins (default 10)

    Returns:
        ECE as a float in [0, 1]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        if mask.sum() == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Maximum Calibration Error: worst-case bin miscalibration.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        if mask.sum() == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        mce = max(mce, abs(bin_acc - bin_conf))
    return float(mce)


def compute_reliability_bins(y_true: np.ndarray, y_prob: np.ndarray,
                              n_bins: int = 10) -> dict:
    """
    Compute reliability diagram data: per-bin accuracy, confidence, count.

    Returns:
        dict with keys 'bin_centers', 'bin_accuracies', 'bin_confidences', 'bin_counts'
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        if mask.sum() == 0:
            bin_accuracies.append(np.nan)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)
        else:
            bin_accuracies.append(float(y_true[mask].mean()))
            bin_confidences.append(float(y_prob[mask].mean()))
            bin_counts.append(int(mask.sum()))
    return {
        "bin_centers": bin_centers.tolist(),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }


def compute_zone_ece(y_true: np.ndarray, y_prob: np.ndarray,
                     n_zones: int = 4, n_bins: int = 10) -> List[dict]:
    """
    Zone-level calibration: split predictions by confidence quartile, compute ECE per zone.

    Confidence = max(p, 1-p). Zones defined by confidence quartiles.
    Zone 1 = lowest confidence (most uncertain), Zone 4 = highest confidence.

    Returns:
        List of dicts, one per zone, with keys: zone, n_cases, prop_total, ece, mce,
        conf_range_low, conf_range_high, accuracy
    """
    confidence = compute_confidence(y_prob)
    quantile_boundaries = np.percentile(confidence, [25, 50, 75])
    zone_results = []
    zone_thresholds = [
        (0.0, quantile_boundaries[0]),
        (quantile_boundaries[0], quantile_boundaries[1]),
        (quantile_boundaries[1], quantile_boundaries[2]),
        (quantile_boundaries[2], 1.01),
    ]
    n_total = len(y_true)
    for zone_idx, (low, high) in enumerate(zone_thresholds, start=1):
        if zone_idx == n_zones:
            mask = confidence >= low
        else:
            mask = (confidence >= low) & (confidence < high)
        n_zone = int(mask.sum())
        if n_zone == 0:
            zone_results.append({
                "zone": zone_idx, "n_cases": 0, "prop_total": 0.0,
                "ece": np.nan, "mce": np.nan,
                "conf_range_low": float(low), "conf_range_high": float(high),
                "accuracy": np.nan
            })
            continue
        zone_y_true = y_true[mask]
        zone_y_prob = y_prob[mask]
        zone_pred = (zone_y_prob >= 0.5).astype(int)
        zone_results.append({
            "zone": zone_idx,
            "n_cases": n_zone,
            "prop_total": float(n_zone / n_total),
            "ece": compute_ece(zone_y_true, zone_y_prob, n_bins=n_bins),
            "mce": compute_mce(zone_y_true, zone_y_prob, n_bins=n_bins),
            "conf_range_low": float(low),
            "conf_range_high": float(high),
            "accuracy": float((zone_y_true == zone_pred).mean()),
        })
    return zone_results


def coverage_accuracy_curve(y_true: np.ndarray, y_prob: np.ndarray,
                             n_steps: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selective prediction coverage-accuracy curve.
    Sort predictions by confidence (highest first), compute accuracy at each coverage level.

    Returns:
        (coverages, accuracies) both as 1D numpy arrays, length n_steps+1
    """
    confidence = compute_confidence(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    n = len(y_true)
    sorted_idx = np.argsort(-confidence)

    step_size = max(1, n // n_steps)
    k_values = list(range(step_size, n, step_size)) + [n]
    k_values = sorted(set(k_values))

    coverages = []
    accuracies = []
    for k in k_values:
        selected = sorted_idx[:k]
        acc = float((y_true[selected] == y_pred[selected]).mean())
        coverages.append(k / n)
        accuracies.append(acc)
    return np.array(coverages), np.array(accuracies)


def compute_aurc(coverages: np.ndarray, accuracies: np.ndarray) -> float:
    """
    Area Under Risk-Coverage curve (risk = 1 - accuracy).
    Lower AURC = better selective prediction (lower risk at all coverage levels).
    """
    risks = 1.0 - accuracies
    trap_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return float(trap_fn(risks, coverages))


def compute_random_aurc(y_true: np.ndarray, y_prob: np.ndarray = None) -> float:
    """
    AURC for a random (non-confident) baseline: a selector that ignores confidence
    yields constant accuracy = test_accuracy at all coverage levels.
    AURC_random = (1 - test_accuracy) * 1.0 = overall test risk.
    If y_prob is provided, test_accuracy is at threshold 0.5; otherwise majority class.
    """
    if y_prob is not None:
        y_pred = (y_prob >= 0.5).astype(int)
        overall_acc = float((y_true == y_pred).mean())
    else:
        majority_class = int(np.mean(y_true) < 0.5)
        overall_acc = float((y_true == majority_class).mean())
    return 1.0 - overall_acc


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000,
                 ci: float = 0.95, seed: int = 1042) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval for mean.

    Returns:
        (mean, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = [
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_bootstrap)
    ]
    boot_means = np.array(boot_means)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return float(np.mean(values)), lower, upper
