# Temporal Fraud Detection Performance Analysis

## 1. Objective

Evaluate the selected XGBoost fraud detection model across different time periods using the validation dataset.

The objective is to identify variations in fraud detection performance over time.

## 2. Methodology

- Model: Full-feature XGBoost
- Decision threshold: 0.939572
- Dataset: Validation set
- Validation steps: 417–520
- Period grouping: 10 simulation steps
- Periods analyzed: 11

Metrics include precision, recall, F1-score, false positives and false negatives.

## 3. Key findings

- Highest F1-score: 92.31% (steps 431–440).
- Lowest F1-score: 76.13% (steps 471–480).
- Highest recall: 91.67% (steps 501–510).
- Lowest recall: 77.63% (steps 471–480).
- Lowest precision: 74.68% (steps 471–480).
- The model's performance varies across validation periods.

## 4. Observations

Steps 471–480 show weaker fraud detection performance, with 17 false negatives and 20 false positives.

Steps 501–510 show stronger performance, with 110 detected fraud cases out of 120 actual fraud cases.

The results indicate that overall model performance can hide variations between time periods.

## 5. Limitations

- The analysis uses synthetic PaySim transactions.
- Periods have different transaction volumes and fraud counts.
- Metrics from smaller periods may be less stable.
- Simulation steps do not represent real calendar dates.
- This analysis does not establish real-world model stability or drift.

## 6. Output

Detailed period-level metrics:

`docs/reports/analysis/temporal_performance.csv`

The model and decision threshold were not modified during this analysis.