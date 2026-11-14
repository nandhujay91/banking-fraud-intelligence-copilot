# Fraud Detection Model Comparison and Validation Report

## 1. Objective

Evaluate fraud detection models using a chronological train-validation-test strategy and select a suitable candidate for the Banking Fraud Intelligence Copilot.

The evaluation focuses on precision, recall, F1-score, ROC-AUC and PR-AUC because fraud represents a highly imbalanced class.

## 2. Dataset and Evaluation Strategy

| Dataset | Rows | Fraud cases | Fraud rate |
|---|---:|---:|---:|
| Training | 5,990,447 | 4,707 | 0.0786% |
| Validation | 91,560 | 1,074 | 1.1730% |
| Final test | 280,613 | 2,432 | 0.8667% |

- Training: steps 1–416
- Validation: steps 417–520
- Final test: steps 521–743

The final test dataset was not used for threshold selection.

## 3. Models Evaluated

Five model configurations were evaluated:

1. Logistic Regression baseline.
2. Full-feature XGBoost.
3. XGBoost without transaction type.
4. XGBoost without sender balance (`oldbalanceOrg`).
5. XGBoost using transaction type and sender balance only.

The full-feature XGBoost model uses:

- `type`
- `amount`
- `oldbalanceOrg`
- `log_amount`

Each XGBoost ablation was trained separately using the same chronological datasets and model configuration. Its decision threshold was independently selected using validation F1.

## 4. Final Test Performance

| Metric | Logistic Regression | Full XGBoost | Without type | Without balance | Type + balance |
|---|---:|---:|---:|---:|---:|
| Accuracy | 99.14% | 99.68% | 99.41% | 99.22% | 99.35% |
| Precision | 50.70% | 79.64% | 63.52% | 66.08% | 72.39% |
| Recall | 34.46% | 84.58% | 74.55% | 20.19% | 41.28% |
| F1-score | 0.4103 | **0.8203** | 0.6860 | 0.3093 | 0.5258 |
| ROC-AUC | Not recorded | 0.9991 | 0.9955 | 0.9171 | 0.9757 |
| PR-AUC | 0.4629 | **0.9151** | 0.7949 | 0.2932 | 0.5538 |
| False positives | 815 | 526 | 1,041 | 252 | 383 |
| False negatives | 1,594 | 375 | 619 | 1,941 | 1,428 |

## 5. Feature Ablation Analysis

### 5.1 Removing transaction type

Removing transaction type reduced:

- Precision from 79.64% to 63.52%.
- Recall from 84.58% to 74.55%.
- F1-score from 0.8203 to 0.6860.
- PR-AUC from 0.9151 to 0.7949.

Transaction type provides useful predictive information in this dataset.

### 5.2 Removing sender balance

Removing `oldbalanceOrg` reduced:

- Recall from 84.58% to 20.19%.
- F1-score from 0.8203 to 0.3093.
- PR-AUC from 0.9151 to 0.2932.

This indicates that sender balance is an important predictive feature for the PaySim dataset.

### 5.3 Using transaction type and sender balance only

This experiment excluded `amount` and `log_amount`.

Results:

- Precision: 72.39%.
- Recall: 41.28%.
- F1-score: 0.5258.
- PR-AUC: 0.5538.

The reduced feature set performs worse than the full-feature model, indicating that transaction amount contributes additional predictive information.

### 5.4 Ablation conclusion

The full-feature XGBoost model achieved the strongest test performance among the evaluated configurations.

However, the observed feature relationships may reflect synthetic PaySim transaction-generation patterns and should not be assumed to generalize to real banking data.

## 6. Duplicate Pattern Investigation

Cross-split duplicate checks were performed using the model input features:

- `type`
- `amount`
- `oldbalanceOrg`
- `log_amount`

| Comparison | Shared unique combinations |
|---|---:|
| Training vs validation | 2,749 |
| Training vs test | 8,370 |
| Validation vs test | 127 |

No conflicting labels were found among the shared feature combinations.

### Seen vs unseen test evaluation

| Metric | Seen combinations | Unseen combinations |
|---|---:|---:|
| Test rows | 8,496 | 272,117 |
| Fraud cases | 66 | 2,366 |
| Precision | 100.00% | 79.18% |
| Recall | 84.85% | 84.57% |
| F1-score | 0.9180 | 0.8179 |
| PR-AUC | 1.0000 | 0.9125 |
| ROC-AUC | 1.0000 | 0.9991 |

The seen subset contains only 66 fraud cases, so its metrics should be interpreted cautiously. Fingerprint-based matching also does not establish the absence of all forms of data leakage.

## 7. Explainability Analysis

SHAP was implemented to investigate global feature contributions and individual fraud predictions.

The global analysis uses a sample of 2,000 test transactions. Individual waterfall plots were generated for three fraud-labelled test examples.

Global SHAP analysis identified transaction amount, sender balance and transaction-type indicators among the important model features.

SHAP values explain model behaviour but do not establish causal relationships.

## 8. Decision Threshold Analysis

Threshold analysis was conducted using the validation dataset only.

The selected threshold was 0.939572, based on maximum validation F1 among the evaluated thresholds.

At this threshold:

| Metric | Validation result |
|---|---:|
| Precision | 87.04% |
| Recall | 87.52% |
| F1-score | 0.8728 |
| False positives | 140 |
| False negatives | 134 |

The threshold provides a balanced precision-recall trade-off among the tested candidates.

Detailed results: `docs/reports/analysis/threshold_analysis.csv`.

## 9. Temporal Performance Analysis

The validation dataset was grouped into 10-step simulation periods.

Across 11 periods:

- Highest F1-score: 92.31% (steps 431–440).
- Lowest F1-score: 76.13% (steps 471–480).
- Highest recall: 91.67% (steps 501–510).
- Lowest recall: 77.63% (steps 471–480).

The results show variation in model performance across validation periods.

These findings do not establish real-world model stability or concept drift. The periods have different transaction volumes and fraud counts.

Detailed results: `docs/reports/analysis/temporal_performance.csv`.

## 10. Model Selection

**Selected candidate: Full-feature XGBoost.**

Reasons:

- Highest test F1-score among the evaluated configurations.
- Highest test PR-AUC among the evaluated configurations.
- Higher precision and recall than the Logistic Regression baseline.
- Fewer false positives and false negatives than the other XGBoost configurations.
- Strong performance on unseen feature combinations.

Selected model:

`models/xgboost/xgboost_fraud_model.joblib`

Decision threshold:

`0.939572`

The model is retained as the primary development candidate.

## 11. Limitations and Further Validation

- PaySim is a synthetic transaction dataset, not real banking transaction data.
- Fraud