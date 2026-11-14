# Fraud Detection — Baseline Model Report

## 1. Objective

Establish a reproducible machine learning baseline for identifying fraudulent transactions using the PaySim synthetic mobile money dataset.

The initial model uses Logistic Regression with class weighting to address the severe imbalance between fraudulent and legitimate transactions.

## 2. Dataset

| Item | Value |
|---|---:|
| Total transactions | 6,362,620 |
| Fraudulent transactions | 8,213 |
| Non-fraudulent transactions | 6,354,407 |
| Overall fraud rate | 0.1291% |
| Dataset | PaySim |
| Target | `isFraud` |

The dataset contains highly imbalanced classes. Fraudulent transactions represent approximately 0.13% of all records.

## 3. Chronological Data Split

The data was split chronologically to reduce temporal leakage.

| Dataset | Step range | Transactions | Fraud cases | Fraud rate |
|---|---|---:|---:|---:|
| Training | 1–416 | 5,990,447 | 4,707 | 0.0786% |
| Validation | 417–520 | 91,560 | 1,074 | 1.1730% |
| Final test | 521–743 | 280,613 | 2,432 | 0.8667% |

The validation set was created from the original training period. The final test set remained untouched during model training and threshold selection.

The different fraud rates across periods indicate temporal distribution shift.

## 4. Feature Selection

### Input features

| Feature | Description |
|---|---|
| `type` | Transaction type |
| `amount` | Transaction amount |
| `oldbalanceOrg` | Sender's balance before the transaction |
| `log_amount` | Log-transformed transaction amount |

### Excluded features

| Feature | Reason |
|---|---|
| `isFraud` | Prediction target |
| `isFlaggedFraud` | Existing simulator-generated fraud flag |
| `newbalanceOrig` | Post-transaction information |
| `newbalanceDest` | Post-transaction information |
| `nameOrig` | Account identifier |
| `nameDest` | Account identifier |
| `step` | Retained for chronological splitting, excluded from model inputs |

The initial model uses only four input features.

## 5. Model Configuration

**Algorithm:** Logistic Regression

**Preprocessing:**
- OneHotEncoder for transaction type.
- StandardScaler for numerical features.
- `handle_unknown="ignore"` for unseen transaction categories.

**Classifier configuration:**
- `class_weight="balanced"`
- `max_iter=1000`
- `random_state=42`
- `solver="lbfgs"`

No SMOTE oversampling was applied in this baseline.

## 6. Decision Threshold Selection

The default classification threshold of 0.5 was not retained.

Instead, the model generated fraud probabilities on the validation dataset. The threshold maximizing validation F1-score was selected.

**Selected threshold:** `0.997692`

The selected threshold was then applied unchanged to the final test dataset.

## 7. Validation Results

| Metric | Result |
|---|---:|
| Precision | 0.6386 |
| Recall | 0.4032 |
| F1-score | 0.4943 |
| PR-AUC | 0.5569 |

### Validation confusion matrix

| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 90,241 | 245 |
| Actual fraud | 641 | 433 |

## 8. Final Test Results

The final test dataset was evaluated using the threshold selected on validation data.

| Metric | Result |
|---|---:|
| Precision | 0.5070 |
| Recall | 0.3446 |
| F1-score | 0.4103 |
| PR-AUC | 0.4629 |

### Final test confusion matrix

| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 277,366 | 815 |
| Actual fraud | 1,594 | 838 |

### Interpretation

- The model correctly identified 838 fraudulent transactions.
- It missed 1,594 fraudulent transactions.
- It generated 815 false-positive alerts.
- Approximately 50.7% of predicted fraud alerts were actually fraudulent.

The model provides a more selective alert profile than the initial threshold-0.5 baseline