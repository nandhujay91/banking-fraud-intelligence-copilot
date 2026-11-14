# Banking Fraud Intelligence Copilot

An end-to-end banking fraud analytics and intelligence platform designed to support fraud analysts through data engineering, machine learning, explainable AI, retrieval-augmented generation (RAG), and multi-agent orchestration.

**Project Status:** In Development — Phases 1–3 Completed

---

## 1. Project Overview

Banking Fraud Intelligence Copilot is an analytical decision-support prototype that aims to help fraud analysts:

- Analyse banking transaction patterns.
- Identify potentially suspicious transactions.
- Understand machine learning fraud predictions.
- Explore fraud trends through SQL analytics.
- Retrieve relevant fraud investigation knowledge.
- Interact with fraud analytics using natural language.

The project uses the PaySim synthetic mobile-money transaction dataset.

**Important:** This is a portfolio prototype using synthetic data. It does not process real banking transactions, autonomously block payments, or establish that a customer has committed fraud.

## 2. Business Objectives

- Build a reliable transaction data pipeline.
- Perform data quality checks and feature engineering.
- Develop fraud detection models for imbalanced transaction data.
- Evaluate fraud detection using business-relevant metrics.
- Explain model predictions using explainable AI.
- Build a retrieval-based fraud knowledge platform.
- Develop an AI-powered analyst decision-support workflow.

## 3. Dataset

**Dataset:** PaySim — Synthetic Mobile Money Transactions

**Source:** [PaySim Dataset](https://huggingface.co/datasets/LordNR/AMLGraphX-Paysim)

| Attribute | Value |
|---|---:|
| Total transactions | 6,362,620 |
| Total features | 11 |
| Fraud transactions | 8,213 |
| Non-fraud transactions | 6,354,407 |
| Fraud rate | 0.1291% |
| Transaction types | 5 |

The dataset is synthetic and does not represent real banking customers or actual financial losses.

The raw dataset and generated Parquet files are excluded from Git because of their size.

## 4. Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python 3.11 |
| Environment management | uv |
| Data processing | Pandas, NumPy, PyArrow |
| Database | PostgreSQL |
| Database connectivity | Psycopg |
| Query and analytics | SQL |
| Machine learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP |
| Backend | FastAPI |
| Generative AI | Azure OpenAI (planned) |
| Vector search | Azure AI Search (planned) |
| Frontend | React (planned) |
| Monitoring | Grafana, Prometheus |
| Testing | Pytest |
| Code quality | Ruff |
| Deployment | Docker, Azure (planned) |

Technologies marked as planned will be introduced in their respective project phases.

## 5. Implementation Progress

### Phase 1 — Dataset Investigation

**Status: Completed**

- Dataset inspection and validation.
- Fraud and non-fraud distribution analysis.
- Transaction-type analysis.
- Transaction amount and balance analysis.
- Feature leakage investigation.
- Duplicate and conflicting-record analysis.
- Fraud temporal distribution analysis.
- Chronological train/test split investigation.

### Phase 2 — Data Engineering and Feature Engineering

**Status: Completed**

- Transaction data validation.
- Feature engineering pipeline.
- Log transformation of transaction amounts.
- Parquet-based processed data storage.
- Feature data contract and specification.
- Feature quality validation.
- Unit and integration testing.
- Chronological train/test split creation.

**Chronological split:**

| Dataset | Records | Fraud records |
|---|---:|---:|
| Training | 6,082,007 | 5,781 |
| Testing | 280,613 | 2,432 |

The split uses transaction simulation steps, with training data from steps 1–520 and testing data from steps 521–743.

### Phase 3 — PostgreSQL and Banking Analytics

**Status: Completed**

- PostgreSQL database setup.
- Transaction table creation and constraints.
- Loading 6.36 million transactions.
- Database indexing.
- Transaction and fraud KPI analysis.
- Fraud analysis by transaction type.
- Fraud transaction amount analysis.
- Fraud versus non-fraud comparisons.
- Temporal fraud analysis.
- SQL analytics documentation.

### Upcoming Phases

| Phase | Description | Status |
|---|---|---|
| 4 | Fraud Detection Machine Learning | Planned |
| 5 | Explainable AI | Planned |
| 6 | RAG Knowledge Platform | Planned |
| 7 | Multi-Agent Fraud Intelligence Copilot | Planned |
| 8 | Backend API and Frontend | Planned |
| 9 | Testing and Security | Planned |
| 10 | Deployment and Monitoring | Planned |

## 6. Project Architecture

The planned architecture consists of:

1. Transaction ingestion and data validation.
2. Data engineering and feature engineering.
3. PostgreSQL analytics and fraud detection.
4. Explainable machine learning.
5. RAG-based knowledge retrieval.
6. Multi-agent orchestration.
7. Analyst-facing APIs and frontend.
8. Deployment and monitoring.

## 7. Project Structure

```text
banking-fraud-intelligence-copilot/
│
├── api/
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── docs/
│   ├── data/
│   └── reports/
│
├── frontend/
├── notebooks/
├── project_documentation/
│
├── sql/
│   └── analytics/
│
├── src/
│   └── banking_fraud/
│       ├── data_quality/
│       ├── database/
│       ├── features/
│       ├── ingestion/
│       └── preprocessing/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── deployment/
├── scripts/
│
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## 8. Getting Started

### Prerequisites

- Python 3.11
- uv
- PostgreSQL 15 or compatible version
- Git

### Clone the repository

```bash
git clone https://github.com/nandhujay91/banking-fraud-intelligence-copilot.git

cd banking-fraud-intelligence-copilot
```

### Install dependencies

```bash
uv sync
```

### Run code quality checks

```bash
uv run ruff check .
```

### Run tests

```bash
uv run pytest -v
```

**Dataset setup:** Obtain the PaySim dataset from its source and place the CSV at `data/raw/paysim.csv`. Dataset files are not included in this repository.

Database configuration is required for database ingestion and analytics workflows.

## 9. Data Integrity and Project Limitations

- The project uses synthetic PaySim data.
- The dataset does not represent real-world banking activity.
- Fraud labels are highly imbalanced.
- Chronological evaluation is used to better reflect temporal deployment conditions.
- Post-transaction fields are excluded from the initial pre-transaction model feature set.
- The simulator-generated `isFlaggedFraud` field is excluded from the initial model features to avoid leakage.
- Model predictions are decision-support signals, not definitive fraud determinations.

## 10. Project Goals

The final objective is to build a maintainable, modular fraud intelligence prototype that combines:

- Data engineering.
- Banking analytics.
- Machine learning.
- Explainable AI.
- Retrieval-augmented generation.
- Multi-agent AI systems.
- Backend and frontend development.
- Monitoring and deployment.

---

**Author:** Nandini Arjunan

**GitHub:** [nandhujay91](https://github.com/nandhujay91)

**Repository:** [Banking Fraud Intelligence Copilot](https://github.com/nandhujay91/banking-fraud-intelligence-copilot)