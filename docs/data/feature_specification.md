# Feature Specification — Banking Fraud Intelligence Copilot

## 1. Objective

Define the initial features for the PaySim fraud detection model.

**Prediction scenario:** Pre-transaction fraud risk assessment.

Only information available before transaction execution can be used.

## 2. Approved features

| Feature | Source | Definition | Status |
|---|---|---|---|
| `type` | type | Transaction category | Approved |
| `amount` | amount | Proposed transaction amount | Approved |
| `oldbalanceOrg` | oldbalanceOrg | Previous sender balance | Approved |
| `log_amount` | amount | log(1 + amount) | Approved |

## 3. Candidate features

| Feature | Formula | Current decision |
|---|---|---|
| `amount_to_sender_balance_ratio` | amount / oldbalanceOrg | Pending evaluation |
| `oldbalanceDest` | Original dataset column | Conditional availability |
| `amount_to_recipient_balance_ratio` | amount / oldbalanceDest | Conditional |
| `sender_balance_after_estimate` | oldbalanceOrg - amount | Rejected |

## 4. Edge-case handling

### Zero sender balance

33.04% of transactions have `oldbalanceOrg = 0`.

A sender balance ratio must not divide by zero.

If the ratio is approved later, we will evaluate:
- A separate zero-balance indicator.
- Missing values for undefined ratios.
- Model performance with and without the ratio.

### Zero recipient balance

42.50% of transactions have `oldbalanceDest = 0`.

Recipient-based features remain conditional until availability is confirmed.

### Transaction amount

Original amount is highly right-skewed.

The `log_amount` transformation is approved:

`log_amount = log1p(amount)`

Observed skewness:
- Original amount: 30.9939
- Log amount: -0.5550

## 5. Excluded features

| Feature | Reason |
|---|---|
| `newbalanceOrig` | Post-transaction information |
| `newbalanceDest` | Post-transaction information |
| `isFlaggedFraud` | Simulator-generated flag |
| `nameOrig` | Identifier excluded initially |
| `nameDest` | Identifier excluded initially |

`isFraud` is the classification target, not an input feature.

## 6. Initial feature set

The initial feature engineering pipeline will use:

- `type`
- `amount`
- `oldbalanceOrg`
- `log_amount`

The original amount is retained alongside its transformed version.

## 7. Validation requirements

Before model training, verify:

1. No unexpected missing feature values.
2. No infinite numerical values.
3. `log_amount` is calculated correctly.
4. Transaction categories are valid.
5. Target labels remain separate from input features.
6. No post-transaction fields enter the model input.
7. Feature values remain within expected numerical ranges.

## 8. Status

**Document status:** Initial specification.

**Current phase:** Phase 2 — Data Engineering and Feature Engineering.

**Next task:** Implement the initial feature engineering pipeline and validate its output.
