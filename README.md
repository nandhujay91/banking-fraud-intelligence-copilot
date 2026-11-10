@'
# Banking Fraud Intelligence Copilot

An end-to-end banking fraud analytics and intelligence platform combining data engineering, machine learning, explainable AI, retrieval-augmented generation (RAG), and large language models.

## 1. Project Overview

The platform is designed to support fraud analysts in exploring suspicious transaction patterns, understanding fraud model predictions, analysing transaction trends, and retrieving relevant fraud investigation guidance.

This is an analytical decision-support prototype. It does not autonomously block transactions or establish that a customer has committed fraud.

## 2. Business Objectives

- Analyse banking transaction patterns and suspicious activities.
- Develop machine learning models for fraud detection.
- Explain model predictions using explainable AI.
- Provide natural-language access to approved fraud analytics.
- Retrieve relevant investigation guidance using RAG.
- Support evidence-based fraud investigation workflows.

## 3. Dataset

Primary dataset: PaySim

Source: https://huggingface.co/datasets/LordNR/AMLGraphX-Paysim

Dataset type: Synthetic mobile-money transactions.

The dataset is simulated and does not represent real banking customers or real-world fraud investigations.

The dataset licence and attribution requirements must be reviewed and documented before redistribution.

## 4. Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python 3.11 |
| Environment | uv |
| Data processing | Pandas, NumPy, PyArrow |
| Database | PostgreSQL |
| Data access | SQLAlchemy, Psycopg |
| Machine learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP |
| Backend | FastAPI |
| Generative AI | Azure OpenAI |
| Vector search | Azure AI Search |
| Frontend | React |
| Monitoring | Grafana, Application Insights |
| Testing | Pytest |
| Code quality | Ruff |
| Deployment | Docker, Azure |

Technologies will be introduced progressively as the project phases require them.

## 5. Architecture

The planned platform consists of five main capabilities:

1. Transaction ingestion and data quality.
2. Banking analytics and fraud detection.
3. Explainable machine learning.
4. RAG-based knowledge retrieval and LLM orchestration.
5. Analyst-facing APIs, monitoring, and deployment.

## 6. Project Structure

```text
config/
data/
docs/
src/banking_fraud/
api/
frontend/
tests/
scripts/
notebooks/
deployment/
project_documentation/