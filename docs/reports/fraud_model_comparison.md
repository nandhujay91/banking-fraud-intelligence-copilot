# Fraud Detection Model Comparison Report

## 1. Objective

Evaluate fraud detection models using a chronological train-validation-test strategy and select a suitable model for the Banking Fraud Intelligence Copilot.

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

Three model configurations were evaluated:

1. Logistic Regression baseline with transaction type and numerical features.
2. XGBoost with transaction type and numerical features.
3. XGBoost without transaction type, using numerical features only.

The numerical features were `amount`, `oldbalanceOrg` and `log_amount`.

## 4. Final Test Performance

| Metric | Logistic Regression | XGBoost with type | XGBoost without type |
|---|---:|---:|---:|
| Accuracy | 99.14% | 99.68% | 99.41% |
| Precision | 50.70% | 79.64% | 63.52% |
| Recall | 34.46% | 84.58% | 74.55% |
| F1-score | 0.4103 | 0.8203 | 0.6860 |
| ROC-AUC | Not recorded | 0.9991 | 0.9955 |
| PR-AUC | 0.4629 | 0.9151 | 0.7949 |
| False positives | 815 | 526 | 1,041 |
| False negatives | 1,594 | 375 | 619 |

## 5. XGBoost Feature Ablation

An ablation experiment was performed by removing transaction type while keeping the same chronological datasets and XGBoost hyperparameters.

The threshold was independently selected using validation F1 for each model.

| Metric | With type | Without type |
|---|---:|---:|
| Precision | 79.64% | 63.52% |
| Recall | 84.58% | 74.55% |
| F1-score | 0.8203 | 0.6860 |
| ROC-AUC | 0.9991 | 0.9955 |
| PR-AUC | 0.9151 | 0.7949 |

Removing transaction type reduced test F1 by 0.1343 and PR-AUC by 0.1202.

This indicates that transaction type provides useful predictive information. The numerical-only model also retains meaningful predictive performance.

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

The model maintains strong performance on unseen feature combinations. However, the seen subset contains only 66 fraud cases, so its perfect precision and ranking metrics should be interpreted cautiously.

This investigation does not prove that duplicate patterns have no effect on model performance.

## 7. Model Selection

**Selected candidate: XGBoost with transaction type.**

Reasons:

- Highest test F1-score among the evaluated models.
- Higher precision and recall than Logistic Regression.
- Fewer false positives and false negatives than the other configurations.
- Strong PR-AUC on the chronological test dataset.
- Retains strong performance on unseen feature combinations.

The model uses a decision threshold selected on validation data: `0.939572`.

## 8. Explainability

SHAP was implemented to investigate global feature contributions and individual fraud predictions.

The global analysis uses a sample of 2,000 test transactions. Individual waterfall plots were generated for three fraud-labelled test examples.

Feature importance and SHAP explanations provide complementary views of model behaviour. They do not establish causal relationships.

## 9. Limitations and Further Validation

- PaySim is a synthetic transaction dataset, not real banking transaction data.
- Fraud prevalence differs considerably between training, validation and test periods.
- Transaction type is strongly associated with the target in this dataset.
- The high ROC-AUC values require further investigation of synthetic dataset patterns and potential shortcuts.
- Duplicate feature combinations exist across chronological splits.
- Threshold performance may change with different fraud prevalence and operational costs.
- SHAP analysis uses a sample rather than the entire test dataset.
- Production performance, latency, calibration and real-world false-positive costs have not yet been established.

## 10. Conclusion

XGBoost with transaction type is the strongest candidate among the three evaluated configurations on the chronological test dataset.

The model achieved 79.64% precision, 84.58% recall, 0.8203 F1-score and 0.9151 PR-AUC.

The feature ablation and duplicate investigations provide additional evidence about model behaviour. Further validation is required before claiming production readiness.

**Project status:** Fraud detection modelling and initial explainability implemented. Further validation, knowledge retrieval, multi-agent orchestration, backend integration, testing and deployment remain outstanding.
