# XGBoost Feature Importance Analysis

## 1. Objective

Analyse the features used by the trained XGBoost fraud detection model to understand which variables contribute most to its predictions.

Three importance measures were extracted:

- **Gain:** Improvement in model performance from splits using a feature.
- **Weight:** Number of times a feature is used in tree splits.
- **Cover:** Number of observations affected by splits using a feature.

## 2. Model Information

| Property | Value |
|---|---|
| Model | XGBoost Classifier |
| Dataset | PaySim synthetic mobile money dataset |
| Model file | `models/xgboost/xgboost_fraud_model.joblib` |
| Features | 8 encoded features |
| Importance measures | Gain, Weight, Cover |

## 3. Feature Importance Results

The following values were extracted from the trained model.

| Rank | Feature | Gain | Weight | Cover |
|---:|---|---:|---:|---:|
| 1 | CASH_IN | 74,785.2266 | 109 | 244,628.8438 |
| 2 | PAYMENT | 33,809.1172 | 136 | 133,063.2031 |
| 3 | TRANSFER | 19,228.9160 | 266 | 90,171.2422 |
| 4 | oldbalanceOrg | 17,462.2773 | 1,311 | 85,219.3594 |
| 5 | CASH_OUT | 14,234.7246 | 240 | 71,803.5781 |
| 6 | log_amount | 13,289.1553 | 542 | 84,150.1875 |
| 7 | amount | 11,764.0371 | 1,266 | 69,379.9922 |
| 8 | DEBIT | 1,714.2883 | 54 | 96,080.1094 |

## 4. Key Findings

### 4.1 Transaction type

CASH_IN has the highest Gain, followed by PAYMENT and TRANSFER.

The dataset investigation established that fraud-labelled transactions occur only in CASH_OUT and TRANSFER.

CASH_IN, PAYMENT and DEBIT have no fraud-labelled transactions in this dataset.

This suggests that transaction type provides important separation between fraud and non-fraud transactions.

### 4.2 Sender balance

`oldbalanceOrg` has the highest split frequency, with 1,311 uses.

This indicates that sender balance is repeatedly used by the decision trees when creating prediction rules.

### 4.3 Transaction amount

`amount` appears in 1,266 splits, while `log_amount` appears in 542 splits.

Both features contribute to the model's decisions, but their importance values alone do not establish whether higher values increase or decrease fraud probability.

## 5. Potential Dataset-Specific Patterns

The importance of transaction type requires careful interpretation.

Because fraud labels occur only in CASH_OUT and TRANSFER, the model may have learned a dataset-specific pattern.

This is not necessarily target leakage: transaction type is available before the transaction outcome. However, the pattern may not generalise to real banking transactions.

Further validation is needed before treating this as a reliable real-world fraud detection strategy.

## 6. Limitations

- Built-in XGBoost importance does not explain the direction of a feature's effect.
- Importance scores do not establish causality.
- One-hot encoded transaction types appear as separate features.
- The results are based on a synthetic dataset.
- Feature importance alone cannot explain individual predictions.

## 7. Next Steps

The next stage is SHAP explainability.

Planned analysis:

1. Global feature importance using SHAP values.
2. Feature impact direction using SHAP summary plots.
3. Individual transaction explanations using SHAP waterfall plots.
4. Investigation of transaction type and amount effects.
5. Explanation of correctly detected and missed fraudulent transactions.

## 8. Conclusion

The XGBoost model relies on transaction type, sender balance and transaction amount features.

CASH_IN has the highest Gain, while `oldbalanceOrg` is the most frequently used feature in tree splits.

These results provide an initial understanding of model behaviour. SHAP analysis will provide more detailed explanations of feature contributions and individual fraud predictions.