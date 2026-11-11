"""
08_statistical_hardening.py
Computes equal-frequency ECE (EF-ECE) and bootstrap CIs for zone-level ECE.
Outputs results/hardening_results.json with defensible statistics.
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "ibm_hr_attrition.csv")
N_BOOTSTRAP  = 1000
RANDOM_SEED  = 42
MODELS       = ["LR", "RF", "GB"]

# ── helpers ──────────────────────────────────────────────────────────────────

def ece_equal_width(probs, labels, n_bins=10):
    """Standard 10-bin equal-width ECE."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(acc - conf)
    return float(ece)


def ece_equal_frequency(probs, labels, n_bins=10):
    """Adaptive equal-frequency (quantile-based) ECE — robust to imbalance."""
    quantiles = np.percentile(probs, np.linspace(0, 100, n_bins + 1))
    # de-duplicate bin edges
    quantiles = np.unique(quantiles)
    ece = 0.0
    n = len(probs)
    for i in range(len(quantiles) - 1):
        lo, hi = quantiles[i], quantiles[i + 1]
        if i == len(quantiles) - 2:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def get_zone_mask(confidence, zone_idx, n_zones=4):
    """Replicates utils.py compute_zone_ece zone partitioning exactly."""
    q_bounds = np.percentile(confidence, np.linspace(0, 100, n_zones + 1)[1:-1])
    thresholds = [(0.0, q_bounds[0]),
                  (q_bounds[0], q_bounds[1]),
                  (q_bounds[1], q_bounds[2]),
                  (q_bounds[2], 1.01)]
    lo, hi = thresholds[zone_idx]
    if zone_idx == n_zones - 1:
        return confidence >= lo
    return (confidence >= lo) & (confidence < hi)


def zone_ece_equal_frequency(probs, labels, confidence, zone_idx, n_zones=4, n_bins=5):
    """EF-ECE restricted to a confidence zone, using utils.py zone logic."""
    mask = get_zone_mask(confidence, zone_idx, n_zones)
    zp, zl = probs[mask], labels[mask]
    if len(zp) < 3:
        return float("nan"), int(mask.sum())
    return ece_equal_frequency(zp, zl, n_bins=min(n_bins, len(zp))), int(mask.sum())


def bootstrap_zone_ece(probs, labels, confidence, zone_idx, n_zones=4,
                       n_bins=10, n_bootstrap=1000, rng=None):
    """Bootstrap 95% CI for zone-level equal-width ECE, using utils.py zone logic."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    mask = get_zone_mask(confidence, zone_idx, n_zones)
    zp, zl = probs[mask], labels[mask]
    n_z = len(zp)
    if n_z < 5:
        return float("nan"), float("nan"), float("nan")
    point_est = ece_equal_width(zp, zl, n_bins=min(n_bins, n_z))
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_z, size=n_z)
        boot.append(ece_equal_width(zp[idx], zl[idx], n_bins=min(n_bins, n_z)))
    boot = np.array(boot)
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))
    return float(point_est), ci_lo, ci_hi


# ── load test labels ──────────────────────────────────────────────────────────

def load_test_labels():
    df = pd.read_csv(DATA_PATH)
    drop_cols = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    le = LabelEncoder()
    for col in ["Gender", "OverTime"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "Attrition" in cat_cols:
        cat_cols.remove("Attrition")
    df = pd.get_dummies(df, columns=cat_cols)

    y = (df["Attrition"] == "Yes").astype(int).values
    df = df.drop(columns=["Attrition"])

    # Same split as pipeline: 64 / 16 / 20, stratified, seed 42
    X_train_fit, X_temp, y_train_fit, y_temp = train_test_split(
        df, y, test_size=530, stratify=y, random_state=RANDOM_SEED)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=294, stratify=y_temp, random_state=RANDOM_SEED)

    return y_test


# ── main ─────────────────────────────────────────────────────────────────────

def run():
    print("Loading calibration results...")
    calib = json.load(open(os.path.join(RESULTS_DIR, "calibration_results.json")))

    print("Reconstructing test labels from IBM HR data...")
    y_test = load_test_labels()
    assert len(y_test) == 294, f"Expected 294 test labels, got {len(y_test)}"
    print(f"  Test labels: {len(y_test)} samples, {y_test.sum()} positives "
          f"({y_test.mean()*100:.1f}% attrition)")

    rng = np.random.default_rng(RANDOM_SEED)
    output = {}

    for model in MODELS:
        probs_list = calib[model]["best_prob_test"]
        probs = np.array(probs_list)
        labels = y_test

        # Confidence = max(p, 1-p)
        confidence = np.maximum(probs, 1 - probs)

        # ── global ECE (EW vs EF) ────────────────────────────────────────
        ew_global = ece_equal_width(probs, labels, n_bins=10)
        ef_global = ece_equal_frequency(probs, labels, n_bins=10)

        # ── zone-level ECE (EW with bootstrap CIs, EF point) ────────────
        zone_results = []
        for z in range(4):   # 0=Zone1 (low conf) .. 3=Zone4 (high conf)
            pt, ci_lo, ci_hi = bootstrap_zone_ece(
                probs, labels, confidence, zone_idx=z,
                n_zones=4, n_bins=10, n_bootstrap=N_BOOTSTRAP, rng=rng)
            ef_pt, nz = zone_ece_equal_frequency(
                probs, labels, confidence, zone_idx=z, n_zones=4)
            zone_results.append({
                "zone":          z + 1,
                "ece_ew":        round(pt, 5),
                "ece_ew_ci95_lo": round(ci_lo, 5),
                "ece_ew_ci95_hi": round(ci_hi, 5),
                "ece_ef":        round(ef_pt, 5) if not (isinstance(ef_pt, float) and np.isnan(ef_pt)) else None,
                "n_cases":       int(nz),
            })

        output[model] = {
            "global_ece_ew": round(ew_global, 5),
            "global_ece_ef": round(ef_global, 5),
            "zone_hardening": zone_results,
        }

        print(f"\n{model}:")
        print(f"  Global ECE (EW): {ew_global:.4f}  |  Global ECE (EF): {ef_global:.4f}")
        for z in zone_results:
            print(f"  Zone {z['zone']}: EW={z['ece_ew']:.4f} "
                  f"[{z['ece_ew_ci95_lo']:.4f}, {z['ece_ew_ci95_hi']:.4f}] "
                  f"EF={z['ece_ef']}  N={z['n_cases']}")

    out_path = os.path.join(RESULTS_DIR, "hardening_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run()
