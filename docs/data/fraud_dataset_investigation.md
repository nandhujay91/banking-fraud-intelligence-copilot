PaySim Fraud Dataset Investigation
Project: Banking Fraud Intelligence Copilot
Dataset: PaySim synthetic mobile-money transactions
Status: Initial exploratory investigation completed
1. Purpose
This report records the measured findings from the initial PaySim
dataset investigation. It covers dataset structure, data quality, fraud
distribution, transaction patterns, balance behaviour, feature
relationships, potential target leakage, and proposed feature
availability rules.

These findings apply to this synthetic dataset. They must not be
presented as evidence of real-world banking fraud behaviour.
2. Dataset overview
The dataset contains 6,362,620 transactions and 11 columns.

  Column             Description
  step             Simulation time step
type             Transaction type
amount           Transaction amount
nameOrig         Origin account identifier
oldbalanceOrg    Origin balance before transaction
newbalanceOrig   Origin balance after transaction
nameDest         Destination account identifier
oldbalanceDest   Destination balance before transaction
newbalanceDest   Destination balance after transaction
isFraud          Fraud label
isFlaggedFraud   Simulator-generated rule-based flag

Transaction types: CASH_IN, CASH_OUT, DEBIT, PAYMENT, and
TRANSFER.

Data limitation: PaySim is synthetic mobile-money transaction data,
not real bank transaction data. step is simulation time, not a
real-world timestamp.
3. Initial data-quality findings
  Check                                          Finding
  Total rows                                   6,362,620
Columns                                             11
Missing values                                       0
Negative transaction amounts                         0
Invalid transaction types                            0
Invalid fraud labels                                 0
Invalid simulator-flag labels                        0
Steps below 1                                        0
Duplicate rows reported by initial check             0

Duplicate-check limitation: The initial duplicate check compared
records within chunks and adjacent chunk boundaries. It does not prove
global uniqueness across the full dataset.
4. Fraud distribution
  Class         Transactions       Share
  Fraud                8,213     0.1291%
Non-fraud        6,354,407    99.8709%
Total            6,362,620   100.0000%

The target is highly imbalanced. Accuracy alone will be insufficient for
evaluating a fraud model. Evaluation should include precision, recall,
F1-score, PR-AUC, and a confusion matrix. Threshold selection should
consider missed fraud alerts and false alerts.
4.1 Fraud by transaction type
  Transaction    Total transactions Fraud transactions         Fraud rate
type
  CASH_IN               1,399,284                  0            0.0000%

  CASH_OUT              2,237,500              4,116            0.1840%

  DEBIT                    41,432                  0            0.0000%

  PAYMENT               2,151,495                  0            0.0000%
TRANSFER                532,909              4,097            0.7688%
All fraud-labelled transactions in this dataset are TRANSFER or
CASH_OUT. TRANSFER has the highest observed fraud rate. Zero
observed fraud in other types does not prove that those transaction
types are inherently safe.
4.2 Transaction amounts
  Statistic              Non-fraud           Fraud
  Count                  6,354,407           8,213
Minimum                     0.01            0.00
First quartile         13,368.40      127,091.33
Median                 74,684.72      441,423.44
Mean                  178,197.04    1,467,967.30
Third quartile        208,364.76    1,517,771.48
Maximum            92,445,516.64   10,000,000.00

Fraud-labelled transactions have higher median and mean amounts in this
dataset. These descriptive differences do not, by themselves, establish
a reliable prediction rule.
5. Fraud rates across simulation periods
Fraud rate is calculated as fraud transactions divided by all
transactions within each 50-step period.
  Simulation     Total transactions Fraud transactions         Fraud rate
period
  1--50                   1,030,263                590            0.0573%

  51--100                    29,253                600            2.0511%

  101--150                  451,852                520            0.1151%

  151--200                  870,814                544            0.0625%

  201--250                  866,110                587            0.0678%

  251--300                  825,358                560            0.0678%

  301--350                  870,265                500            0.0575%

  351--400                  843,115                576            0.0683%

  401--450                  227,687                580            0.2547%

  451--500                   47,090                504            1.0703%

  501--550                  104,212                504            0.4836%

  551--600                   93,028                548            0.5891%

  601--650                   22,742                578            2.5416%

  651--700                   69,410                532            0.7665%
701--750                   11,421                490            4.2903%
The highest observed period-level rate is 4.2903% in steps 701--750
(11,421 transactions). The lowest is 0.0573% in steps 1--50. Transaction
volumes vary considerably, so these results do not demonstrate a steady
increase in fraud over time.
6. Balance behaviour
Balance consistency was examined by transaction type because balance
changes differ by type in this dataset. Balance mismatches are not
automatically evidence of fraud.
  Transaction         Label   Transactions      Sender   Recipient     Sender   Recipient
type                                       unchanged   unchanged   mismatch    mismatch
  CASH_OUT      Non-fraud      2,233,384      45.93%       0.07%     89.12%      19.84%

  CASH_OUT          Fraud          4,116       0.90%       0.44%      0.58%      12.71%

  TRANSFER      Non-fraud        528,812      53.47%       0.02%     96.20%      21.83%
TRANSFER          Fraud          4,097       0.49%      99.29%      0.51%      99.95%
For fraud-labelled TRANSFER transactions, recipient balance behaviour
is especially distinctive. For fraud-labelled CASH_OUT transactions,
sender balance mismatch is much less common than among non-fraud
CASH_OUT transactions. These patterns may reflect the simulator's
generation rules and must not be assumed to generalise to real banking
systems.
7. Feature relationships and redundancy
Pearson correlations were calculated separately for CASH_OUT and
TRANSFER.
7.1 CASH_OUT
  Feature                       Correlation with isFraud
  Sender balance change                             0.4676
Amount                                            0.3131
Recipient balance change                          0.1723
Sender balance mismatch                          -0.1210
Sender balance difference                         0.0475

High correlations between candidate features included: -
sender_unchanged and sender_zero_before: approximately 1.0000. -
sender_zero_after and sender_balance_mismatch: 0.9701.
7.2 TRANSFER
  Feature                     Correlation with isFraud
  Recipient unchanged                             0.9836
Recipient zero after                            0.8826
Sender balance change                           0.4803
Sender balance mismatch                        -0.4017
Recipient zero before                           0.2356

High correlations included: - amount and sender_balance_difference:
-0.9904. - sender_unchanged and sender_zero_before: approximately
1.0000. - recipient_unchanged and isFraud: 0.9836.

A high feature-to-target correlation is not proof of generalisable
predictive power. In particular, recipient_unchanged uses
post-transaction information and requires caution.
8. Target leakage investigation
8.1 Post-transaction balance patterns
For TRANSFER transactions:

  Pattern                           Fraud   Non-fraud
  Recipient unchanged              99.29%       0.02%
Recipient balance zero after     99.29%       0.21%
Sender unchanged                  0.49%      53.47%

The strong association between updated recipient balance behaviour and
the fraud label is a warning against using post-transaction fields in a
pre-transaction prediction model.
8.2 Simulator-generated flag
  Metric                                Result
  Total transactions                 6,362,620
Fraud-labelled transactions            8,213
Simulator-flagged transactions            16
True positives                            16
False positives                            0
False negatives                        8,197
True negatives                     6,354,407
Flag precision                       100.00%
Flag recall                            0.19%

isFlaggedFraud is a simulator-generated rule-based flag. Exclude it
from predictive model inputs. Its relationship with the target can be
reported as a dataset diagnostic, not independent model performance.
9. Proposed feature availability policy
The primary modelling scenario is pre-transaction fraud risk
assessment. Only information available before execution should be
used.
  Column                  Proposed decision       Reason
  type                  Include                 Transaction type is
known when proposed.

  amount                Include                 Proposed transaction
amount is known.

  oldbalanceOrg         Include                 Sender's previous
balance, subject to
scoring-time
availability.

  oldbalanceDest        Conditional             Include only if
recipient's previous
balance can reliably be
retrieved before
execution.

  newbalanceOrig        Exclude                 Updated sender balance
is post-transaction
information.

  newbalanceDest        Exclude                 Updated recipient
balance is
post-transaction
information.

  isFlaggedFraud        Exclude                 Simulator-generated
rule-based flag.

  isFraud               Target only             Supervised-learning
label; never a model
input.

  nameOrig              Exclude from initial    Account identifier;
model                   consider historical
behavioural features
separately if
justified.

  nameDest              Exclude from initial    Account identifier;
model                   consider historical
behavioural features
separately if
justified.
step                  Not yet approved        Simulation time field;
requires validation and
leakage assessment.
The initial candidate feature set is type, amount, and
oldbalanceOrg. oldbalanceDest remains conditional on its
availability at scoring time. This is a proposed policy, not a claim
about the data available in a live bank.
10. Modelling and validation implications
1. Keep isFraud strictly as the target.
2. Exclude isFlaggedFraud from predictive inputs.
3. Exclude updated balances from the pre-transaction model.
4. Confirm whether destination pre-transaction balance is available
   before including it.
5. Investigate mathematical redundancy among balance-derived features.
6. Avoid direct account identifiers in the initial model.
7. Use a validation strategy that respects simulation time; do not rely
   only on a random split.
8. Evaluate class-imbalanced performance using precision, recall,
   F1-score, PR-AUC, and confusion matrices.
9. Select a decision threshold based on documented trade-offs between
   missed fraud and false alerts.
10. Compare a simple baseline with more advanced models before claiming
    improvement.
11. Limitations
- PaySim is synthetic and does not establish performance on real
  banking transactions.
- Simulator rules may create patterns a model could learn that do not
  generalise.
- step is simulated time, not a real timestamp.
- Balance mismatches are common in non-fraud records and require
  transaction-type-specific interpretation.
- Some transaction types have no fraud-labelled examples in this
  dataset.
- Correlation is descriptive and does not demonstrate causation.
- The initial duplicate check did not prove global uniqueness across
  the full dataset.
- No machine-learning model has yet been trained or evaluated in the
  analyses documented here.
- Proposed prediction-time availability must be validated against the
  intended system design.
12. Conclusion
The investigation establishes that PaySim contains 6.36 million
transactions with an overall fraud rate of approximately 0.1291%.
Fraud-labelled transactions are concentrated in TRANSFER and
CASH_OUT, with a higher observed fraud rate for TRANSFER.

The strongest balance-related associations involve post-transaction
recipient balance behaviour in fraud-labelled transfers. These fields
are not appropriate inputs for the proposed pre-transaction model. The
simulator-generated isFlaggedFraud field is also excluded from model
inputs.

The next phase should establish the data contract and prediction-time
feature set, then implement reproducible feature engineering and a
leakage-aware validation strategy.
Project integrity statement: This is an educational and
portfolio-oriented decision-support prototype using synthetic PaySim
data. It is not a deployed banking fraud detection system, and its
findings must not be represented as validated real-world banking
performance.
