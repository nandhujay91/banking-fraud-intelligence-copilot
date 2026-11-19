# Banking Fraud Intelligence Copilot

An end-to-end banking fraud analytics and intelligence platform designed to support fraud analysts through data engineering, machine learning, explainable AI, retrieval-augmented generation (RAG), and multi-agent orchestration.

**Project Status:** In Development — Phases 1–5 Completed; Phase 6 In Progress.

---

## 1. Project Overview

Banking Fraud Intelligence Copilot is an analytical decision-support prototype that aims to help fraud analysts:

- Analyse banking transaction patterns.
- Identify potentially suspicious transactions.
- Understand machine learning fraud predictions.
- Explore fraud trends through SQL analytics.
- Explain model predictions using explainable AI.
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
| Local LLM | Ollama, Qwen2.5 3B |
| Embeddings | nomic-embed-text |
| Document processing | pypdf, python-docx |
| HTTP client | httpx |
| Vector database | PostgreSQL + pgvector (planned) |
| Backend | FastAPI (planned) |
| Frontend | React (planned) |
| Monitoring | Grafana, Prometheus (planned) |
| Testing | Pytest |
| Code quality | Ruff |
| Containerisation | Docker (planned) |
| Cloud deployment | Azure (planned) |

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
- Chunk-based CSV processing.
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

### Phase 4 — Fraud Detection Machine Learning

**Status: Completed — Initial Model Development and Evaluation**

- Logistic Regression baseline model.
- XGBoost fraud detection model.
- Class imbalance handling.
- Chronological training, validation and testing.
- Precision, recall, F1-score, PR-AUC and ROC-AUC evaluation.
- Threshold analysis.
- Feature ablation experiments.
- Temporal performance analysis.
- False-negative investigation.

**XGBoost test-set results:**

| Metric | Result |
|---|---:|
| Precision | 0.7964 |
| Recall | 0.8458 |
| F1-score | 0.8203 |
| PR-AUC | 0.9151 |
| ROC-AUC | 0.9991 |
| Accuracy | 0.9968 |

The selected threshold was 0.939572, chosen based on validation F1-score among the tested thresholds.

These results are specific to the synthetic PaySim dataset and should not be interpreted as expected performance on real banking transactions.

### Phase 5 — Explainable AI and Model Validation

**Status: Completed — Initial Analysis**

- SHAP TreeExplainer implementation.
- Global feature importance analysis.
- Individual transaction prediction explanations.
- Fraud false-negative investigation.
- Feature ablation and comparison.
- Duplicate-pattern analysis across chronological splits.
- Model performance analysis by transaction type.
- Threshold sensitivity analysis.
- Temporal performance investigation.

SHAP explanations help investigate how individual features contribute to model predictions. They do not establish that a transaction is genuinely fraudulent or explain real-world causation.

### Phase 6 — RAG Knowledge Platform

**Status: In Progress**

**Implemented:**

- Local Ollama infrastructure.
- Qwen2.5 3B local language model integration.
- Python HTTP client for LLM interaction.
- PDF, DOCX and TXT document extraction.
- Document metadata generation.
- SHA-256 document checksum.
- JSON-based document registry.
- Duplicate document detection.
- Document ingestion workflow.
- Character-based text chunking with overlap.
- Local embedding model integration using `nomic-embed-text`.
- Single and batch embedding generation.
- Embedding validation and error handling.
- Unit tests for LLM, ingestion, document processing, chunking and embeddings.

**Pending:**

- PostgreSQL pgvector installation and configuration.
- Vector database schema and embedding storage.
- Similarity search and top-K retrieval.
- Metadata-filtered retrieval.
- Retrieval quality evaluation.
- Grounded LLM responses with source citations.
- RAG evaluation and reliability testing.

The current embedding client generates 768-dimensional vectors. Vector storage and retrieval have not yet been implemented.

### Upcoming Phases

| Phase | Description | Status |
|---|---|---|
| 7 | Multi-Agent Fraud Intelligence Copilot | Planned |
| 8 | Backend API and Frontend | Planned |
| 9 | Testing, Security and Reliability | Planned |
| 10 | Deployment and Monitoring | Planned |

## 6. Project Architecture

The planned end-to-end architecture consists of:

1. Transaction ingestion and data validation.
2. Data engineering and feature engineering.
3. PostgreSQL analytics and fraud detection.
4. Explainable machine learning.
5. Document ingestion and RAG knowledge processing.
6. Vector storage and similarity retrieval.
7. Multi-agent fraud investigation and orchestration.
8. Analyst-facing APIs and frontend.
9. Deployment, monitoring and operational support.

## 7. Project Structure

The repository currently contains the implemented data, database, modelling, LLM and RAG components. Additional folders will be introduced as the remaining phases are developed.

```text
banking-fraud-intelligence-copilot/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── knowledge_base/
│       ├── raw/
│       ├── processed/
│       └── metadata/
│
├── docs/
│   ├── data/
│   ├── reports/
│   └── rag/
│
├── sql/
│   └── analytics/
│
├── src/
│   └── banking_fraud/
│       ├── data_quality/
│       ├── database/
│       ├── features/
│       ├── preprocessing/
│       ├── models/
│       ├── llm/
│       └── rag/
│
├── tests/
│   ├── unit/
│   │   └── rag/
│   └── integration/
│
├── scripts/
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

Ollama is required for the local LLM and embedding functionality.

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

### Local LLM setup

Install Ollama and download the models:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Ensure the Ollama service is running before using the local LLM and embedding clients.

### Dataset setup

Obtain the PaySim dataset from its source and place the CSV at:

```text
data/raw/paysim.csv
```

Dataset files are not included in this repository.

Database configuration is required for database ingestion and analytics workflows. Configure credentials through environment variables; never commit passwords or `.env` files.

## 9. Data Integrity and Project Limitations

- The project uses synthetic PaySim data.
- The dataset does not represent real-world banking activity.
- Fraud labels are highly imbalanced.
- Chronological evaluation is used to better reflect temporal deployment conditions.
- Post-transaction fields are excluded from the initial pre-transaction model feature set.
- The simulator-generated `isFlaggedFraud` field is excluded from the initial model features.
- Some model patterns may be specific to the PaySim simulator.
- Model predictions are decision-support signals, not definitive fraud determinations.
- The current RAG implementation does not yet include vector retrieval or grounded answer generation.
- The complete multi-agent workflow and production deployment are future objectives.

## 10. Project Goals

The final objective is to build a maintainable, modular fraud intelligence prototype that combines:

- Data engineering.
- Banking analytics.
- Machine learning.
- Explainable AI.
- Retrieval-augmented generation.
- Local LLM integration.
- Multi-agent AI systems.
- Backend and frontend development.
- Testing and security.
- Monitoring and deployment.

---

**Author:** Nandini Arjunan

**GitHub:** [nandhujay91](https://github.com/nandhujay91)

**Repository:** [Banking Fraud Intelligence Copilot](https://github.com/nandhujay91/banking-fraud-intelligence-copilot)
