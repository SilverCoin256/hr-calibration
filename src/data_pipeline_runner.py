"""
Singleton data loader: runs pipeline once and caches result for subsequent modules.
Import get_data() from this module to get preprocessed splits.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from config_loader import load_config
from importlib import import_module

_cache = None


def get_data() -> dict:
    """
    Run data pipeline if not cached. Return dict with all splits.
    """
    global _cache
    if _cache is not None:
        return _cache

    pipeline_mod = import_module("01_data_pipeline")
    result = pipeline_mod.run_pipeline()
    (
        X_train_fit, X_cal, X_test,
        y_train_fit, y_cal, y_test,
        X_train_scaled, X_cal_scaled, X_test_scaled,
        feature_names, manifest
    ) = result

    _cache = {
        "X_train_fit": X_train_fit,
        "X_cal": X_cal,
        "X_test": X_test,
        "y_train_fit": y_train_fit,
        "y_cal": y_cal,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_cal_scaled": X_cal_scaled,
        "X_test_scaled": X_test_scaled,
        "feature_names": feature_names,
        "manifest": manifest,
    }
    return _cache
