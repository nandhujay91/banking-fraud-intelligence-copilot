# XGBoost Fraud Detection Model Report

## 1. Objective

Develop a machine learning model to identify potentially fraudulent transactions using the PaySim synthetic mobile money dataset.

The objective is to detect fraudulent transactions while balancing precision and recall, given the highly imbalanced target variable.

## 2. Dataset

| Metric | Value |
|---|---:|
| Total transactions | 6,362,620 |
| Fraudulent transactions | 8,213 |
| Non-fraudulent transactions | 6,354,407 |
| Overall fraud rate | 0.1291% |

The dataset is synthetic. Results should not be interpreted as evidence of performance on real banking transactions.

## 3. Data Split

A chronological split was used to avoid training on future transactions.

| Dataset | Steps | Rows | Fraud cases | Fraud rate |
|---|---|---:|---:|---:|
| Training | 1–416 | 5,990,447 | 4,707 | 0.0786% |
| Validation | 417–520 | 91,560 | 1,074 | 1.1730% |
| Test | 521–743 | 280,613 | 2,432 | 0.8667% |

The validation and test periods have different fraud rates from the training period. This indicates temporal distribution differences that should be considered when interpreting model performance.

## 4. Features

The model uses four input features:

- `type` — transaction type.
- `amount` — transaction amount.
- `oldbalanceOrg` — sender's balance before the transaction.
- `log_amount` — logarithmic transformation of transaction amount.

The following fields were excluded:

- `isFraud` — prediction target.
- `isFlaggedFraud` — existing simulator-generated flag.
- `newbalanceOrig` and `newbalanceDest` — post-transaction information.
- `nameOrig` and `nameDest` — entity identifiers.
- `oldbalanceDest` — excluded from the initial model.
- `step` — retained for chronological splitting, not used as a model feature.

## 5. Model Configuration

**Algorithm:** XGBoost Classifier

| Parameter | Value |
|---|---:|
| Number of estimators | 100 |
| Maximum depth | 6 |
| Learning rate | 0.1 |
| Subsample | 0.8 |
| Column sample by tree | 0.8 |
| Scale positive weight | 1271.67 |
| Objective | binary:logistic |
| Evaluation metric | aucpr |
| Tree method | hist |
| Random state | 42 |

Class imbalance was addressed using `scale_pos_weight`.

Categorical transaction type was one-hot encoded. Numerical features were passed through the preprocessing pipeline.

## 6. Threshold Selection

The classification threshold was selected using the validation dataset by maximising the F1-score.

Selected threshold:

`0.939572`

The final test dataset was evaluated using this validation-selected threshold.

## 7. Model Performance

### Validation results

| Metric | Result |
|---|---:|
| Precision | 0.8704 |
| Recall | 0.8752 |
| F1-score | 0.8728 |
| PR-AUC | 0.9514 |

Confusion matrix:

| | Predicted non-fraud | Predicted fraud |
|---|---:|---:|
| Actual non-fraud | 90,346 | 140 |
| Actual fraud | 134 | 940 |

### Final test results

| Metric | Result |
|---|---:|
| Precision | 0.7964 |
| Recall | 0.8458 |
| F1-score | 0.8203 |
| PR-AUC | 0.9151 |

Confusion matrix:

| | Predicted non-fraud | Predicted fraud |
|---|---:|---:|
| Actual non-fraud | 277,655 | 526 |
| Actual fraud | 375 | 2,057 |

On the final test set, the model detected 2,057 of 2,432 fraudulent transactions and generated 526 false-positive alerts.

## 8. Comparison with Logistic Regression

Both models were evaluated on the same final test dataset.

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Precision | 0.5070 | 0.7964 |
| Recall | 0.3446 | 0.8458 |
| F1-score | 0.4103 | 0.8203 |
| PR-AUC | 0.4629 | 0.9151 |
| False positives | 815 | 526 |
| False negatives | 1,594 | 375 |

XGBoost achieved substantially stronger results across the reported metrics and reduced both false-positive alerts and missed fraud cases compared with the baseline.

## 9. Model Artifacts

The trained model and selected decision threshold are saved at:

- `models/xgboost/xgboost_fraud_model.joblib`
- `models/xgboost/decision_threshold.json`

These generated model artifacts are excluded from Git.

## 10. Limitations

- PaySim is a synthetic dataset and does not represent the full complexity of real banking fraud.
- The fraud distribution changes across chronological periods.
- Validation and test performance differ, indicating that model stability requires further investigation.
- The selected threshold is based on F1-score, not a bank-specific operational cost function.
- The model has not been validated on real-world banking data.

## 11. Conclusion

XGBoost outperformed the Logistic Regression baseline on the chronological test dataset, achieving an F1-score of 0.8203 and PR-AUC of 0.9151.

The model is a promising candidate for the next stages of the project. Further work is required on explainability, temporal stability, and operational evaluation before considering production use.