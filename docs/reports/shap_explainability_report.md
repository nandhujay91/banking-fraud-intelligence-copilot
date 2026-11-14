# SHAP Explainability Report — Banking Fraud Intelligence Copilot

## 1. Objective

The objective of this analysis is to understand how the trained XGBoost fraud detection model makes predictions.

SHAP (SHapley Additive exPlanations) is used to:

- Identify the features that influence model predictions globally.
- Understand whether feature values increase or decrease the model output.
- Explain individual fraudulent transaction predictions.
- Investigate false-negative fraud predictions.

The analysis uses the existing trained XGBoost pipeline without retraining the model.

## 2. Model and Dataset

| Item | Details |
|---|---|
| Dataset | PaySim synthetic mobile money dataset |
| Model | XGBoost Classifier |
| Model pipeline | Scikit-learn Pipeline |
| Preprocessing | OneHotEncoder and numerical passthrough |
| SHAP explainer | TreeExplainer |
| SHAP version | 0.51.0 |
| Global analysis sample | 2,000 test transactions |
| Individual explanations | 3 actual fraud transactions |
| Decision threshold | 0.939572 |

The model uses four input features:

- `type`
- `amount`
- `oldbalanceOrg`
- `log_amount`

The SHAP analysis uses the same preprocessing pipeline as model training.

## 3. Global SHAP Feature Importance

Mean absolute SHAP values were used to measure the overall contribution of each transformed feature to model predictions.

| Rank | Feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | `numerical__amount` | 2.05 |
| 2 | `numerical__oldbalanceOrg` | 1.75 |
| 3 | `categorical__type_PAYMENT` | 1.39 |
| 4 | `categorical__type_CASH_IN` | 1.00 |
| 5 | `numerical__log_amount` | 0.99 |
| 6 | `categorical__type_CASH_OUT` | 0.66 |
| 7 | `categorical__type_TRANSFER` | 0.36 |
| 8 | `categorical__type_DEBIT` | 0.02 |

### Interpretation

- Transaction amount has the highest overall SHAP importance.
- Origin account balance is the second most influential feature.
- Transaction type also contributes significantly to model decisions.
- `PAYMENT` and `CASH_IN` have substantial global importance, even though these transaction types contain no fraud-labelled records in this dataset.
- `amount` and `log_amount` represent related information and both influence predictions.

Mean absolute SHAP importance measures contribution magnitude, not whether a feature increases or decreases fraud probability.

## 4. SHAP Summary Plot

**Output:** `docs/reports/figures/shap_summary.png`

The beeswarm plot shows the distribution of SHAP values for each feature across the sampled transactions.

It provides two important perspectives:

1. Feature importance through the spread and magnitude of SHAP values.
2. Direction of impact through positive and negative SHAP values.

Positive SHAP values move the model output towards fraud, while negative SHAP values move it towards non-fraud.

The plot indicates that transaction amount and origin balance have substantial influence on model predictions.

## 5. SHAP Feature Importance Plot

**Output:** `docs/reports/figures/shap_feature_importance.png`

The SHAP bar plot presents the mean absolute SHAP contribution for each feature.

It confirms the importance ranking:

1. Transaction amount
2. Origin account balance
3. Payment transaction type
4. Cash-in transaction type
5. Log-transformed transaction amount

This analysis complements the earlier XGBoost Gain-based feature importance analysis.

The two methods measure different aspects of model behaviour and their rankings do not need to match.

## 6. Individual Fraud Transaction Explanations

Three actual fraudulent transactions were selected from the final test dataset using a reproducible random sample (`random_state=42`).

Waterfall plots explain how individual feature contributions combine to produce the model output.

### Example 1

**Output:** `shap_fraud_example_1.png`

- Actual class: Fraud
- Model fraud probability: 99.9170%
- Predicted class: Fraud
- Result: Correct detection

The origin account balance and transaction amount-related features contributed to the model's fraud score.

### Example 2

**Output:** `shap_fraud_example_2.png`

- Actual class: Fraud
- Model fraud probability: 92.4376%
- Predicted class: Non-fraud
- Result: False negative

The model assigned a relatively high fraud probability, but it did not cross the selected classification threshold.

### Example 3

**Output:** `shap_fraud_example_3.png`

- Actual class: Fraud
- Model fraud probability: 92.6606%
- Predicted class: Non-fraud
- Result: False negative

The model again assigned a relatively high fraud probability, but the prediction remained below the classification threshold.

## 7. Individual Prediction Verification

The three predictions were independently checked using the saved XGBoost model and decision threshold.

**Selected decision threshold: 0.9395722151**

| Example | Actual label | Fraud probability | Predicted label | Result |
|---|---:|---:|---:|---|
| 1 | 1 | 0.999170 | 1 | Correct detection |
| 2 | 1 | 0.924376 | 0 | False negative |
| 3 | 1 | 0.926606 | 0 | False negative |

The results confirm that one of the three sampled fraud cases was detected, while two were missed.

These three examples are illustrative cases, not a representative estimate of the model's overall recall.

## 8. Decision Threshold Analysis

The classification threshold was selected by maximising F1-score on the validation dataset.

| Parameter | Value |
|---|---:|
| Threshold | 0.939572 |
| Selection metric | Validation F1 |
| Validation precision | 87.04% |
| Validation recall | 87.52% |
| Validation F1-score | 0.8728 |

A predicted fraud probability below the threshold results in a non-fraud classification, even when the probability is relatively high.

The two false negatives in the individual explanations demonstrate this behaviour.

Threshold selection involves a trade-off between detecting fraudulent transactions and generating false-positive alerts.

The operational threshold should ultimately reflect investigation capacity, fraud losses, customer impact and the cost of missed fraud.

## 9. Key Findings

1. Transaction amount and origin account balance are the strongest global SHAP contributors.
2. Transaction type has substantial influence on model predictions.
3. Individual SHAP explanations help identify which features pushed a prediction towards fraud or non-fraud.
4. Two of the three selected actual fraud examples were classified as non-fraud.
5. The classification threshold plays an important role in fraud detection outcomes.
6. SHAP provides an explanation of model behaviour, not proof of causal fraud indicators.

## 10. Limitations

- The PaySim dataset is synthetic and may not represent real banking fraud behaviour.
- Global SHAP analysis uses a sample of 2,000 transactions rather than the entire test dataset.
- Only three actual fraud transactions were selected for individual explanations.
- SHAP values explain the trained model's output, not the real-world cause of a transaction.
- Feature contributions can depend on relationships between correlated features.
- The selected decision threshold has not been optimised using real-world investigation costs.
- The analysis does not establish production readiness.

## 11. Output Files

| File | Description |
|---|---|
| `shap_summary.png` | Global SHAP beeswarm plot |
| `shap_feature_importance.png` | Global mean absolute SHAP importance |
| `shap_fraud_example_1.png` | Individual fraud explanation |
| `shap_fraud_example_2.png` | Individual fraud explanation |
| `shap_fraud_example_3.png` | Individual fraud explanation |

All visualisations are stored in:

`docs/reports/figures/`

## 12. Conclusion

SHAP explainability provides additional insight into the behaviour of the XGBoost fraud detection model.

The global analysis identifies transaction amount and origin account balance as the strongest contributors. Individual waterfall plots demonstrate how feature contributions influence specific predictions.

The investigation also confirms that actual fraudulent transactions can receive relatively high fraud probabilities but remain classified as non-fraud when they fall below the selected threshold.

This analysis adds an explainability layer to the Banking Fraud Intelligence Copilot and provides a foundation for subsequent fraud investigation and decision-support capabilities.