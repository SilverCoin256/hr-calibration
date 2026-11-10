# hr-calibration

Study notes for **"Calibration and Explanation Stability in Organizational Machine Learning: Governance-Aware Evaluation"** (under review, JEI).

On the synthetic IBM HR benchmark, calibration error in the lowest-confidence quartile was 3.6–5.4 times that of the highest-confidence quartile after calibration. The study also tests how abstaining on low-confidence predictions changes coverage and accuracy, and how stable SHAP attributions are for individual cases.

## Repository contents

This checkout contains the study summary, dependencies, and [dataset instructions](data/README.md). Reproducing the reported values requires the `src/` scripts and their generated outputs, which are absent from this checkout.

## Key results

| Model | AUC | ECE (pre-cal) | ECE (post-cal) | GADRS |
|---|---|---|---|---|
| Logistic Regression | 0.777 | 0.041 | 0.031 | 0.25 |
| Random Forest | 0.802 | 0.065 | 0.032 | 0.75 |
| Gradient Boosting | 0.798 | 0.058 | 0.039 | 0.60 |

Per-sample SHAP stability (SSI) did not show zone-level differentiation: 0.69–0.75 across confidence zones for all models.

## Dataset

The [dataset notes](data/README.md) give the download link and file path for the synthetic IBM HR Analytics Employee Attrition & Performance dataset. It has 1,470 rows and 35 columns.
