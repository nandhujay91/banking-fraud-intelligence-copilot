# PostgreSQL Implementation and Banking Analytics



**Project:** Banking Fraud Intelligence Copilot

**Database:** `banking_fraud_db`

**Database engine:** PostgreSQL 15.18

**Dataset:** PaySim synthetic mobile-money transactions

**Status:** Completed and validated



\---



## 1. Objective



This phase implements PostgreSQL as the relational database layer for the Banking Fraud Intelligence Copilot.



The objectives were to:



\- Create a structured relational database.

\- Design a transaction table with appropriate data types and constraints.

\- Load the complete PaySim dataset.

\- Validate database ingestion.

\- Create indexes for analytical queries.

\- Perform banking fraud analytics using SQL.

\- Preserve reusable analytical queries.



All findings in this report relate to the synthetic PaySim dataset, not real-world banking transactions.



## 2. Database Setup



PostgreSQL 15.18 was installed and configured on Windows.



| Configuration | Value |

|---|---|

| Database engine | PostgreSQL |

| Version | 15.18 |

| Host | localhost |

| Port | 5433 |

| Database | banking_fraud_db |

| Schema | public |

| Main table | transactions |



Database creation:



```sql

CREATE DATABASE banking_fraud_db;

```



Connection:



```bash

psql -U postgres -h localhost -p 5433 -d banking_fraud_db

```



The database connection was verified successfully.



## 3. Database Schema



Created the `transactions` table with 12 columns.



| Column | Data type | Description |

|---|---|---|

| transaction_id | BIGINT | Generated primary key |

| step | BIGINT | Simulation time step |

| type | VARCHAR(20) | Transaction category |

| amount | NUMERIC(18,2) | Transaction amount |

| name_orig | VARCHAR(30) | Sender identifier |

| old_balance_org | NUMERIC(18,2) | Sender's previous balance |

| new_balance_orig | NUMERIC(18,2) | Sender's resulting balance |

| name_dest | VARCHAR(30) | Recipient identifier |

| old_balance_dest | NUMERIC(18,2) | Recipient's previous balance |

| new_balance_dest | NUMERIC(18,2) | Recipient's resulting balance |

| is_fraud | SMALLINT | Fraud label |

| is_flagged_fraud | SMALLINT | Simulator-generated flag |



### Constraints implemented



\- Primary key on `transaction_id`.

\- Allowed transaction types: CASH_IN, CASH_OUT, DEBIT, PAYMENT and TRANSFER.

\- Fraud labels restricted to 0 and 1.

\- Transaction amount must be non-negative.



## 4. Data Ingestion



**Implementation file:**



`src/banking_fraud/database/load_transactions.py`



The ingestion process:



1\. Reads the original CSV in chunks of 100,000 rows.

2\. Uses Psycopg 3 for PostgreSQL connectivity.

3\. Uses the PostgreSQL COPY protocol for bulk loading.

4\. Checks that the destination table is empty before ingestion.

5\. Loads the records within a database transaction.

6\. Compares the source and database row counts.



### Ingestion results



| Metric | Result |

|---|---:|

| Source records | 6,362,620 |

| Database records | 6,362,620 |

| Fraud records | 8,213 |

| Non-fraud records | 6,354,407 |



**Result:** All 6,362,620 transactions were successfully loaded and the database row count matched the source.



## 5. Database Indexing



Two analytical indexes were created.



### Index 1: Fraud and simulation step



```sql

CREATE INDEX idx_transactions_fraud_step

ON transactions (is_fraud, step);

```



Purpose: Support queries filtering by fraud label and simulation step.



### Index 2: Transaction type and fraud label



```sql

CREATE INDEX idx_transactions_type_fraud

ON transactions (type, is_fraud);

```



Purpose: Support analytical queries involving transaction categories and fraud labels.



Both indexes were verified through PostgreSQL's `pg_indexes` catalog.



Note: Index creation was verified, but no formal query-performance benchmark was performed.



## 6. Banking Fraud Analytics



Eight SQL analytical queries were implemented and executed successfully.



The reusable queries are maintained in:



`sql/analytics/fraud_analysis.sql`



### 6.1 Overall fraud KPIs



| Metric | Result |

|---|---:|

| Total transactions | 6,362,620 |

| Fraud transactions | 8,213 |

| Fraud rate | 0.1291% |

| Non-fraud transactions | 6,354,407 |

| Total fraud-labelled transaction amount | 12,056,415,427.84 |

| Average fraud transaction amount | 1,467,967.30 |



The dataset is highly imbalanced, with fraud representing approximately 0.1291% of all transactions.



The fraud-labelled amount is the sum of transaction values marked as fraud. It is not verified financial loss.



### 6.2 Fraud by transaction type



| Transaction type | Total transactions | Fraud transactions | Fraud rate |

|---|---:|---:|---:|

| TRANSFER | 532,909 | 4,097 | 0.7688% |

| CASH_OUT | 2,237,500 | 4,116 | 0.1840% |

| PAYMENT | 2,151,495 | 0 | 0% |

| CASH_IN | 1,399,284 | 0 | 0% |

| DEBIT | 41,432 | 0 | 0% |



**Finding:** TRANSFER has the highest observed fraud rate, while CASH_OUT has slightly more fraud-labelled transactions.



The absence of fraud labels in other transaction types does not establish that those types are inherently safe.



### 6.3 Fraud amount by transaction type



| Transaction type | Fraud count | Total fraud-labelled amount | Average amount | Maximum amount |

|---|---:|---:|---:|---:|

| TRANSFER | 4,097 | 6,067,213,184.01 | 1,480,891.67 | 10,000,000 |

| CASH_OUT | 4,116 | 5,989,202,243.83 | 1,455,102.59 | 10,000,000 |



**Finding:** TRANSFER has the highest total fraud-labelled transaction value. CASH_OUT has 19 more fraud-labelled transactions.



### 6.4 Fraud versus non-fraud transaction amounts



| Metric | Non-fraud | Fraud |

|---|---:|---:|

| Transaction count | 6,354,407 | 8,213 |

| Average amount | 178,197.04 | 1,467,967.30 |

| Median amount | 74,684.72 | 441,423.44 |

| Minimum amount | 0.01 | 0.00 |

| Maximum amount | 92,445,516.64 | 10,000,000 |



**Finding:** Fraud-labelled transactions have higher average and median amounts in this dataset. However, transaction amount alone cannot reliably determine fraud.



### 6.5 Amount analysis by transaction type and fraud label



| Type | Fraud label | Transaction count | Average amount | Median amount |

|---|---:|---:|---:|---:|

| CASH_OUT | 0 | 2,233,384 | 173,917.16 | 146,946.56 |

| CASH_OUT | 1 | 4,116 | 1,455,102.59 | 435,
