"""
Master runner: executes full pipeline in order.
Usage: python src/run_all.py [--skip-shap] [--figures-only]

--skip-shap: Skip the computationally intensive SHAP bootstrap (step 5).
             Run without this flag for first complete execution.
--figures-only: Regenerate figures from cached results without rerunning analysis.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config_loader import load_config

cfg = load_config()

_trained_models_cache = None
_test_probs_cache = None
_baseline_results_cache = None


def get_trained_models_and_probs():
    """Cache-backed model trainer. Returns (trained_models, test_probs, baseline_results)."""
    global _trained_models_cache, _test_probs_cache, _baseline_results_cache
    if _trained_models_cache is not None:
        return _trained_models_cache, _test_probs_cache, _baseline_results_cache
    import importlib
    mod = importlib.import_module("02_baseline_models")
    trained_models, test_probs, results = mod.run()
    _trained_models_cache = trained_models
    _test_probs_cache = test_probs
    _baseline_results_cache = results
    return trained_models, test_probs, results


def main():
    parser = argparse.ArgumentParser(description="JEI Calibration Governance Paper Pipeline")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP bootstrap stability (step 5)")
    parser.add_argument("--figures-only", action="store_true",
                        help="Regenerate figures from cached results only")
    args = parser.parse_args()

    if args.figures_only:
        print("=" * 60)
        print("Figures-only mode: regenerating from cached results")
        print("=" * 60)
        import importlib
        fig_mod = importlib.import_module("07_figures")
        fig_mod.run()
        return

    print("=" * 60)
    print("JEI Calibration Governance Paper — Full Pipeline")
    print("=" * 60)

    # Step 1: Data pipeline
    print("\n[Step 1/6] Data pipeline...")
    import importlib
    data_mod = importlib.import_module("01_data_pipeline")
    data_mod.run_pipeline()

    # Step 2: Baseline models
    print("\n[Step 2/6] Baseline models (5-fold CV + test evaluation)...")
    trained_models, test_probs, baseline_results = get_trained_models_and_probs()

    # Step 3: Calibration analysis
    print("\n[Step 3/6] Calibration analysis (ECE, Platt, isotonic, zone-level)...")
    calib_mod = importlib.import_module("03_calibration_analysis")
    calibration_results = calib_mod.run(trained_models, test_probs)

    # Step 4: Selective prediction
    print("\n[Step 4/6] Selective prediction (coverage-accuracy, AURC, thresholds)...")
    sel_mod = importlib.import_module("04_selective_prediction")
    selective_results = sel_mod.run(calibration_results)

    # Step 5: SHAP stability (optional skip)
    if args.skip_shap:
        print("\n[Step 5/6] SHAP stability: SKIPPED (--skip-shap flag set)")
        shap_path = os.path.join(cfg["RESULTS_DIR"], "shap_stability_results.json")
        if os.path.exists(shap_path):
            with open(shap_path) as f:
                shap_results = json.load(f)
            print(f"  Loaded cached SHAP results from {shap_path}")
        else:
            print("  WARNING: No cached SHAP results found. GADRS will use NaN for SSI.")
            shap_results = {m: {"ssi_zone_a": None, "ssi_zone_d": None,
                               "base_top5_features": []} for m in ["LR", "RF", "GB"]}
            with open(shap_path, "w") as f:
                json.dump(shap_results, f, indent=2)
    else:
        print("\n[Step 5/6] SHAP bootstrap stability (may take 15-30 min)...")
        shap_mod = importlib.import_module("05_shap_stability")
        shap_results = shap_mod.run(calibration_results, trained_models)

    # Step 6: GADRS
    print("\n[Step 6/6] GADRS computation + covariate shift test...")
    gadrs_mod = importlib.import_module("06_gadrs")
    gadrs_results = gadrs_mod.run(calibration_results, selective_results,
                                   shap_results, trained_models)

    # Figures
    print("\n[Figures] Generating all manuscript figures...")
    fig_mod = importlib.import_module("07_figures")
    fig_mod.run()

    print("\n" + "=" * 60)
    print("Pipeline complete. Key outputs:")
    print(f"  Results JSON: {cfg['RESULTS_DIR']}/")
    print(f"  Figures: {cfg['FIGURES_DIR']}/")
    print("\nKey findings summary:")
    for name in ["LR", "RF", "GB"]:
        b = baseline_results[name]
        c = calibration_results[name]
        g = gadrs_results["gadrs"]["primary"][name]
        print(f"  {name}: AUC={b['test_auc']:.3f} | "
              f"ECE_raw={c['raw']['ece']:.3f} -> ECE_cal={c['ece_calibrated']:.3f} | "
              f"GADRS_readiness={g['gadrs_readiness']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
