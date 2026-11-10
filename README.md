# hr-calibration

Reproduction code for *Calibration and Explanation Stability in Organizational Machine Learning: A Governance-Aware Evaluation Framework*, the JEI manuscript.

On the synthetic IBM HR dataset, calibration error in the lowest-confidence quartile was 3.6–5.4 times that of the highest-confidence quartile after calibration. The study also measures what happens when models abstain on low-confidence predictions and how stable feature attributions are across bootstrap samples.

## Reproduce the study

Download the dataset using the [data instructions](data/README.md). From the repository root, run:

```bash
python -m pip install -r requirements.txt
python src/run_all.py
python src/08_statistical_hardening.py
```

The first script writes the model results and regenerates the figures. The second repeats the calibration checks with bootstrap intervals and alternative binning. The recovered [JSON results](results/) matched both reruns byte for byte.

## Test results

| Model | AUC | ECE before calibration | ECE after calibration | GADRS |
|---|---:|---:|---:|---:|
| Logistic regression | 0.802 | 0.047 | 0.047 | 0.252 |
| Random forest | 0.777 | 0.041 | 0.032 | 0.750 |
| Gradient boosting | 0.779 | 0.065 | 0.047 | 0.304 |

AUC measures how well a model ranks cases. Expected calibration error (ECE) compares predicted probabilities with observed outcomes. The Governance-Aware Deployment Readiness Score (GADRS) combines calibration, abstention, and explanation stability. See the [model results](results/baseline_results.json), [calibration results](results/calibration_results.json), and [GADRS results](results/gadrs_results.json).

SHAP explanation stability ranged from 0.69 to 0.75 across models and confidence zones.
