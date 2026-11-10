"""
Configuration: paths, seeds, constants.
All experimental parameters centralized here.
"""
import os

# --- Paths ---
DATA_PATH = "data/ibm_hr_attrition.csv"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- Seeds ---
RANDOM_SEED = 42
BOOTSTRAP_SEED = 1042
N_BOOTSTRAP_CI = 1000       # bootstrap resamples for confidence intervals
N_SHAP_BOOTSTRAP = 50       # bootstrap resamples for SHAP stability; C(50,2)=1225 pairwise Spearman rhos
N_FOLDS = 5

# --- Data splits ---
TEST_SIZE = 0.20
CALIBRATION_SPLIT = 0.20    # fraction of training for Platt/isotonic fitting

# --- Calibration parameters ---
ECE_BINS = 10
ECE_GOVERNANCE_THRESHOLD = 0.05   # proposed governance reference; not an authoritative standard

# --- Governance confidence thresholds ---
# Confidence = max(p, 1-p); abstain if below threshold
GOV_CONSERVATIVE = 0.65
GOV_MODERATE = 0.70
GOV_STRICT = 0.75

# --- GADRS weights (theory-motivated; sensitivity analysis uses equal and ECE-heavy) ---
GADRS_WEIGHTS = {
    "ece_calibrated": 0.30,
    "aurc_delta": 0.25,
    "lowconf_ece": 0.25,
    "ssi_lowconf": 0.20,
}
GADRS_WEIGHTS_EQUAL = {k: 0.25 for k in GADRS_WEIGHTS}
GADRS_WEIGHTS_ECE_HEAVY = {
    "ece_calibrated": 0.50,
    "aurc_delta": 0.17,
    "lowconf_ece": 0.17,
    "ssi_lowconf": 0.16,
}

# --- Covariate shift simulation ---
SHIFT_NOISE_SCALE = 0.5   # sigma = SHIFT_NOISE_SCALE * feature_std
N_TOP_SHAP_FOR_SHIFT = 3  # perturb top-N SHAP-important features

# --- Model hyperparameters ---
LR_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_SEED,
}
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 1,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}
GB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "random_state": RANDOM_SEED,
}

# Lighter params for SHAP bootstrap refits (100 vs 200 estimators for speed)
# SHAP stability does not require maximum-quality models, only stable sampling
RF_PARAMS_BOOT = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_leaf": 1,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}
GB_PARAMS_BOOT = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 4,
    "random_state": RANDOM_SEED,
}
