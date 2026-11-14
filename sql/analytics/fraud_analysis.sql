-- Banking Fraud Intelligence Copilot
-- PostgreSQL Business Analytics

-- 1. Overall fraud KPIs
SELECT
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS total_fraud_transactions,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS total_fraud_amount,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount END), 2) AS average_fraud_amount
FROM transactions;


-- 2. Fraud by transaction type
SELECT
    type,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY type
ORDER BY fraud_rate_pct DESC;


-- 3. Fraud amount statistics by transaction type
SELECT
    type,
    COUNT(*) AS fraud_transactions,
    ROUND(SUM(amount), 2) AS total_fraud_amount,
    ROUND(AVG(amount), 2) AS average_fraud_amount,
    ROUND(MAX(amount), 2) AS highest_fraud_amount
FROM transactions
WHERE is_fraud = 1
GROUP BY type
ORDER BY total_fraud_amount DESC;


-- 4. Fraud versus non-fraud amount comparison
SELECT
    is_fraud,
    COUNT(*) AS transaction_count,
    ROUND(AVG(amount), 2) AS average_amount,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)::numeric, 2) AS median_amount,
    ROUND(MIN(amount), 2) AS minimum_amount,
    ROUND(MAX(amount), 2) AS maximum_amount
FROM transactions
GROUP BY is_fraud
ORDER BY is_fraud;


-- 5. Amount comparison by transaction type and fraud label
SELECT
    type,
    is_fraud,
    COUNT(*) AS transaction_count,
    ROUND(AVG(amount), 2) AS average_amount,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)::numeric, 2) AS median_amount
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT')
GROUP BY type, is_fraud
ORDER BY type, is_fraud;


-- 6. Fraud distribution across simulation steps
SELECT
    step,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY step
HAVING SUM(is_fraud) > 0
ORDER BY step;


-- 7. Simulation steps with the highest fraud counts
SELECT
    step,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY step
HAVING SUM(is_fraud) > 0
ORDER BY fraud_transactions DESC
LIMIT 15;


-- 8. Fraud rates for simulation steps with at least 1,000 transactions
SELECT
    step,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY step
HAVING COUNT(*) >= 1000
   AND SUM(is_fraud) > 0
ORDER BY fraud_rate_pct DESC
LIMIT 15;
