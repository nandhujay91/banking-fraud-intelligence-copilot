# XGBoost Decision Threshold Analysis

## 1. Objective

Evaluate different decision thresholds for the selected XGBoost fraud detection model using the validation dataset.

The objective is to understand the trade-off between fraud detection and false positive alerts.

The saved XGBoost model was reused. No retraining was performed.

## 2. Evaluation methodology

- Model: Full-feature XGBoost
- Features: `type`, `amount`, `oldbalanceOrg`, `log_amount`
- Dataset: Validation set (steps 417–520)
- Validation records: 91,560
- Actual fraud cases: 1,074
- Actual non-fraud cases: 90,486
- Evaluation metrics: Precision, Recall, F1-score, False Positives and False Negatives

The final test dataset was not used for threshold selection.

## 3. Threshold comparison

| Threshold | Precision | Recall | F1-score | False Positives | False Negatives |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 43.93% | 100.00% | 61.04% | 1,371 | 0 |
| 0.70 | 52.80% | 100.00% | 69.11% | 960 | 0 |
| 0.80 | 61.02% | 100.00% | 75.79% | 686 | 0 |
| 0.85 | 65.64% | 99.63% | 79.14% | 560 | 4 |
| 0.90 | 71.44% | 98.04% | 82.65% | 421 | 21 |
| 0.92 | 76.10% | 95.16% | 84.57% | 321 | 52 |
| **0.939572** | **87.04%** | **87.52%** | **87.28%** | **140** | **134** |
| 0.95 | 90.00% | 82.12% | 85.88% | 98 | 192 |
| 0.97 | 92.79% | 77.93% | 84.72% | 65 | 237 |
| 0.99 | 96.63% | 61.45% | 75.13% | 23 | 414 |

## 4. Findings

- Lower thresholds increase fraud recall but also generate more false positive alerts.
- Higher thresholds improve precision but miss more fraudulent transactions.
- Threshold 0.939572 provides the highest F1-score among the evaluated thresholds.
- At this threshold, validation precision is 87.04% and recall is 87.52%.
- The model produces 140 false positives and 134 false negatives on the validation dataset.

## 5. Threshold selection

The current threshold of **0.939572** is retained.

Selection criterion: Maximum validation F1-score.

This threshold provides a balanced precision-recall trade-off among the evaluated candidates.

## 6. Limitations

- The threshold is optimized using synthetic PaySim data.
- F1-score does not account for the different financial costs of false positives and false negatives.
- Real banking deployment would require business-defined alert costs, operational capacity and additional validation.
- Validation performance does not guarantee equivalent performance on future real-world transactions.

## 7. Output

Detailed threshold metrics are available in:

`docs/reports/analysis/threshold_analysis.csv`

The original model and saved threshold remain unchanged.
