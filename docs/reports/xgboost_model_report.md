# XGBoost Fraud Detection Model Report

## 1. Objective

The objective is to develop a machine learning model that identifies potentially fraudulent financial transactions using transaction-level features.

The model is trained and evaluated using the PaySim synthetic mobile money dataset.

The primary focus is on identifying fraudulent transactions while balancing precision and recall, given the highly imbalanced target variable.

## 2. Dataset Overview

| Metric | Value |
|---|---:|
| Total transactions | 6,362,620 |
| Fraudulent transactions | 8,213 |
| Non-fraudulent transactions | 6,354,407 |
| Fraud rate | 0.1291% |

Fraudulent transactions occur only in the `CASH_OUT` and `TRANSFER` transaction types in this dataset.

## 3. Feature Selection

The following features were used:

| Feature | Description |
|---|---|
| `type` | Transaction type |
| `amount` | Transaction amount |
| `oldbalanceOrg` | Origin account balance before the transaction |
| `log_amount` | Log-transformed transaction amount |

The following fields were excluded:

- `nameOrig` and `nameDest`: account identifiers.
- `newbalanceOrig` and `newbalanceDest`: post-transaction information that would not be available for pre-transaction prediction.
- `isFlaggedFraud`: simulator-generated flag with very low recall.
- `step`: retained for chronological splitting, but excluded from model input.

## 4. Data Splitting Strategy

A chronological split was used to reduce the risk of training on future transactions and evaluating on earlier transactions.

| Dataset | Steps | Rows | Fraud cases | Fraud rate |
|---|---:|---:|---:|---:|
| Training | 1–416 | 5,990,447 | 4,707 | 0.0786% |
| Validation | 417–520 | 91,560 | 1,074 | 1.1730% |
| Final test | 521–743 | 280,613 | 2,432 | 0.8667% |

The validation set was used for threshold selection. The final test set was kept separate and used for final evaluation.

The fraud prevalence differs across periods, indicating temporal distribution shift in the dataset.

## 5. Model Configuration

**Algorithm:** XGBoost Classifier

| Parameter | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `scale_pos_weight` | 1271.67 |
| `objective` | binary:logistic |
| `eval_metric` | aucpr |
| `tree_method` | hist |
| `random_state` | 42 |

Class imbalance was addressed using `scale_pos_weight`, calculated from the training class distribution.

The transaction type was one-hot encoded. Numeric features were passed through the preprocessing pipeline.

## 6. Decision Threshold

The default classification threshold of 0.5 was not used.

Instead, the decision threshold was selected by maximising the F1-score on the validation dataset.

**Selected threshold: 0.939572**

This same threshold was applied to the training, validation and final test evaluations.

## 7. Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

PR-AUC, precision and recall are particularly important because fraudulent transactions represent a very small proportion of the dataset.

### 7.1 Training Performance

| Metric | Result |
|---|---:|
| Accuracy | 99.83% |
| Precision | 29.16% |
| Recall | 85.55% |
| F1-score | 0.4349 |
| ROC-AUC | 0.9991 |
| PR-AUC | 0.6823 |

Confusion matrix:

| | Predicted non-fraud | Predicted fraud |
|---|---:|---:|
| Actual non-fraud | 5,975,956 | 9,784 |
| Actual fraud | 680 | 4,027 |

### 7.2 Validation Performance

| Metric | Result |
|---|---:|
| Accuracy | 99.70% |
| Precision | 87.04% |
| Recall | 87.52% |
| F1-score | 0.8728 |
| ROC-AUC | 0.9994 |
| PR-AUC | 0.9514 |

Confusion matrix:

| | Predicted non-fraud | Predicted fraud |
|---|---:|---:|
| Actual non-fraud | 90,346 | 140 |
| Actual fraud | 134 | 940 |

### 7.3 Final Test Performance

<badge color="success">Final test evaluation</badge>

| Metric | Result |
|---|---:|
| Accuracy | 99.68% |
| Precision | 79.64% |
| Recall | 84.58% |
| F1-score | 0.8203 |
| ROC-AUC | 0.9991 |
| PR-AUC | 0.9151 |

Confusion matrix:

| | Predicted non-fraud | Predicted fraud |
|---|---:|---:|
| Actual non-fraud | 277,655 | 526 |
| Actual fraud | 375 | 2,057 |

On the final test dataset, the model identified 2,057 of 2,432 fraudulent transactions and produced 526 false-positive predictions.

## 8. Logistic Regression vs XGBoost

Both models were evaluated on the same final chronological test dataset.

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Accuracy | 99.14% | 99.68% |
| Precision | 50.70% | 79.64% |
| Recall | 34.46% | 84.58% |
| F1-score | 0.4103 | 0.8203 |
| PR-AUC | 0.4629 | 0.9151 |
| ROC-AUC | Not reported | 0.9991 |
| False positives | 815 | 526 |
| False negatives | 1,594 | 375 |

XGBoost performed better than the Logistic Regression baseline across the reported test metrics.

It identified more fraudulent transactions while producing fewer false-positive predictions.

ROC-AUC was not calculated in the original Logistic Regression evaluation, so no comparison is reported for that metric.

## 9. Feature Importance Analysis

Feature importance was extracted from the trained XGBoost model using Gain, Weight and Cover.

| Feature | Gain | Weight | Cover |
|---|---:|---:|---:|
| `categorical__type_CASH_IN` | 74,785.2266 | 109 | 244,628.8438 |
| `categorical__type_PAYMENT` | 33,809.1172 | 136 | 133,063.2031 |
| `categorical__type_TRANSFER` | 19,228.9160 | 266 | 90,171.2422 |
| `numerical__oldbalanceOrg` | 17,462.2773 | 1,311 | 85,219.3594 |
| `categorical__type_CASH_OUT` | 14,234.7246 | 240 | 71,803.5781 |
| `numerical__log_amount` | 13,289.1553 | 542 | 84,150.1875 |
| `numerical__amount` | 11,764.0371 | 1,266 | 69,379.9922 |
| `categorical__type_DEBIT` | 1,714.2883 | 54 | 96,080.1094 |

### Interpretation

- Transaction type has substantial influence on the model's predictions.
- `oldbalanceOrg`, `log_amount` and `amount` also contribute to model decisions.
- `CASH_IN` and `PAYMENT` have high Gain despite having no fraud-labelled transactions in this dataset. They may help the model distinguish non-fraudulent transaction patterns.
- Feature importance does not indicate whether a feature increases or decreases the fraud prediction.
- These results describe the model's learned patterns, not causal relationships.

SHAP analysis is a planned next step to understand the direction and contribution of individual features to predictions.

## 10. Generalisation and Overfitting Assessment

The ROC-AUC values were similar across the three datasets:

| Dataset | ROC-AUC |
|---|---:|
| Training | 0.9991 |
| Validation | 0.9994 |
| Final test | 0.9991 |

The similar ROC-AUC values do not show an obvious separation between training and test ranking performance.

However, training precision and PR-AUC are substantially lower than validation and test results. These metrics are influenced by fraud prevalence, which differs considerably between the splits. Training metrics are also calculated on data used to fit the model.

Therefore, the differences should not be interpreted as proof of underfitting or overfitting.

Further validation on independent real-world banking data and additional time periods would be required before making production generalisation claims.

## 11. Limitations

- PaySim is a synthetic dataset and may not represent real banking fraud behaviour.
- Fraud prevalence varies across chronological splits.
- Transaction type is strongly associated with fraud labels in this dataset, which may make the model dependent on simulator-specific patterns.
- The model uses a limited set of transaction-level features.
- No real-time inference or production deployment has been validated yet.
- ROC-AUC alone does not establish operational fraud detection quality.
- The selected threshold has not been validated against real-world investigation costs or customer impact.

## 12. Conclusion

XGBoost achieved an F1-score of **0.8203**, precision of **79.64%**, recall of **84.58%** and PR-AUC of **0.9151** on the final chronological test dataset.

Compared with the Logistic Regression baseline, XGBoost substantially improved fraud detection performance and reduced both false positives and false negatives.

The results demonstrate promising performance on the PaySim dataset. They should not be treated as evidence of production readiness or validated performance on real banking transactions.

## 13. Next Steps

1. Perform SHAP-based explainability analysis.
2. Build a fraud investigation knowledge base for RAG.
3. Develop specialised agents for transaction analysis and fraud investigation.
4. Integrate the model into a FastAPI backend.
5. Add model evaluation and monitoring.
6. Validate the system with additional independent data before making production claims.