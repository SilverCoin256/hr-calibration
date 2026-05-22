# hr-calibration

Code for the paper **"Calibration and Explanation Stability in Organizational Machine Learning: Governance-Aware Evaluation"** (under review, JEI).

The paper evaluates three classifiers on the IBM HR attrition benchmark across four governance-relevant properties — global ECE, zone-level calibration, selective prediction utility, and per-sample SHAP stability — and introduces the Governance-Aware Deployment Readiness Score (GADRS) as a composite summary metric.

## What this does

Standard HR analytics evaluations report AUC. This project tests whether the probability outputs behind those AUC numbers are actually trustworthy enough to use for review triage. The short answer: they often aren't, and the problem concentrates in low-confidence predictions — which are precisely the cases you'd route to human review.

## Key results

| Model | AUC | ECE (pre-cal) | ECE (post-cal) | GADRS |
|---|---|---|---|---|
| Logistic Regression | 0.777 | 0.041 | 0.031 | 0.25 |
| Random Forest | 0.802 | 0.065 | 0.032 | 0.75 |
| Gradient Boosting | 0.798 | 0.058 | 0.039 | 0.60 |

Zone 1 (lowest confidence quartile) ECE is 3.6–5.4× higher than Zone 4 across all three models, even after post-hoc calibration. Per-sample SHAP stability (SSI) did not show zone-level differentiation: 0.69–0.75 across confidence zones for all models.

## Structure

```
src/
  00_config.py                  # paths and hyperparameters
  01_data_pipeline.py           # load + preprocess IBM HR data
  02_baseline_models.py         # LR, RF, GB training
  03_calibration_analysis.py    # ECE, zone-level, post-hoc calibration
  04_selective_prediction.py    # coverage-accuracy under abstention thresholds
  05_shap_stability.py          # bootstrap SSI computation
  06_gadrs.py                   # composite score aggregation
  07_figures.py                 # all paper figures
  run_all.py                    # full pipeline end to end
data/
  README.md                     # how to get the dataset
results/                        # JSON outputs from each stage
figures/                        # generated plots
```

## Setup

```bash
pip install -r requirements.txt
```

Place `ibm_hr_attrition.csv` in `data/raw/` (see `data/README.md` for the Kaggle link), then:

```bash
python src/run_all.py
```

Outputs go to `results/` and `figures/`.

## Dataset

IBM HR Analytics Employee Attrition dataset — 1,470 observations, 35 features, 16.1% positive class. Synthetic, no real employee records. Publicly available on [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).

## Requirements

See `requirements.txt`. Python 3.10+, scikit-learn ≥ 1.3.0, shap ≥ 0.42.0.
