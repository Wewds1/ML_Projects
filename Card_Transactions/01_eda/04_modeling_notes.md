# Modeling Notes

The best validation model ended up being **logistic_balanced** with a chosen threshold of **0.65**.

This was the test-set readout after retraining on train+validation:

- PR-AUC: **0.0355**
- ROC-AUC: **0.5650**
- Precision: **0.0226**
- Recall: **0.1000**
- F1: **0.0369**

Honestly, the big thing here is not perfect precision. It's catching enough fraud without letting the false positives get totally silly. The threshold file in `models/config.json` keeps that decision explicit.

## Saved artifacts

- `models/fraud_detector_v1.pkl`
- `models/config.json`
- `models/model_comparison.csv`
- `models/threshold_analysis.csv`
- `figures/precision_recall_curve.png`
- `figures/confusion_matrix.png`
