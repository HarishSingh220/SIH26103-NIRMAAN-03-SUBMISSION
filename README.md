# Nirmaan O3 — AI-Powered Infrastructure Risk Monitoring System

An AI-powered platform that turns raw, PDF-heavy infrastructure project reporting into early, explainable, and actionable risk signals — built for SIH26103.

## Table of Contents

1. [Project Information](#1-project-information)
2. [Problem Statement](#2-problem-statement)
3. [Idea / Solution / Prototype](#3-idea--solution--prototype)
4. [How It Addresses the Problem](#4-how-it-addresses-the-problem)
5. [Uniqueness](#5-uniqueness)
6. [Methodology & Implementation](#6-methodology--implementation)
7. [Technology Stack](#7-technology-stack)
8. [Architecture](#8-architecture)
9. [Challenges](#9-challenges)
10. [Feasibility, Viability & Practical Implementation](#10-feasibility-viability--practical-implementation)
11. [Final Presentation](#11-final-presentation)
12. [Demo Video](#12-demo-video)
13. [Screenshots / Prototype](#13-screenshots--prototype)
14. [Future Scope](#14-future-scope)
15. [Model Evaluation Report](#15-a-complete-report-of-the-model-evaluation-and-the-comparison-between-statistical-modelsconventional-vs-ml-models-our-approach)
16. [PAIMANA Combined Prediction API](#paimana-combined-prediction-api)
17. [Running the Website](#running-the-website)

---

## 1. Project Information

| Field | Details |
|---|---|
| **Project Title** | Nirmaan O3 — AI-Powered Infrastructure Risk Monitoring System |
| **PS ID** | SIH26103 |
| **PS Title** | Use case on web-based integrated project-monitoring platform |
| **Organisation / Ministry** | Ministry of Statistics and Programme Implementation |
| **Theme** | Infrastructure Project Monitoring |
| **Category** | Software |
| **Team Name** | NRMAAR |

**Team Members**

| Name | Role |
|---|---|
| Harish Singh (Team Leader) | Tech Lead |
| Harshita | User Interface, Presentation |
| Sanskriti | User Interface |
| Naman Sudrshan | ML Lead |
| Aarush Raj Singh | ML |
| Suryansh P. Singh | Backend Lead |

---

## 2. Problem Statement

Infrastructure projects reported under central monitoring frameworks (PAIMANA quarterly reports) routinely run over budget or behind schedule. In most cases, by the time an overrun becomes visible in the raw reporting numbers, it is already too late to intervene usefully — corrective action at that stage is reactive rather than preventive.

Reviewers today work with a large, PDF-heavy historical dataset and have to manually cross-check each project, quarter by quarter, to judge whether something looks wrong. This is slow, inconsistent across reviewers, and doesn't scale to thousands of tracked projects across dozens of sectors and states.

What reviewers actually need is more than a raw reporting dataset — they need:

- An **early signal** that a project's reported figures look unusual compared to its own history or its peers.
- A **prediction** of whether the project is heading for a cost or schedule overrun.
- An **estimate** of how large that overrun could be, in concrete terms (₹ Cr, months of delay).
- A **plain-language explanation** of *why* the model is flagging the project, so the finding is trustworthy and actionable rather than a black-box number.

Nirmaan O3 is built to close this gap — converting a static, descriptive reporting pipeline into a predictive, explainable, and prioritised one.

---

## 3. Idea / Solution / Prototype

Nirmaan O3 is an AI-powered infrastructure risk monitoring system that sits on top of the existing PAIMANA-style project reporting data and adds a predictive layer to it. Specifically, it:

- **Predicts risk, cost overruns and time overruns** — for every project, in every reporting quarter, using models trained on historical outcomes rather than fixed rules.
- **Estimates how much overrun may occur** — not just a risk flag, but a quantified cost-overrun percentage, a time-overrun percentage, and a projected delay in months.
- **Detects unusual project behaviour** — an anomaly-detection layer flags projects whose reporting pattern (spending trend, milestone counts, reporting frequency) deviates from what similar projects normally look like.
- **Provides reasons behind model predictions** — every flagged project comes with a ranked list of contributing factors (e.g. progress ratio, reporting frequency, project age) so the finding can be defended and understood.
- **Generates an overall project risk score** — the classification, regression, and anomaly outputs are combined into one summarised risk verdict (Low / Medium / High) per project.
- **LLM summarizes the complete risk analysis** — a GenAI layer converts the technical model output into a short, human-readable paragraph a non-technical reviewer can act on immediately.
- **Interactive dashboard for project exploration and visualization** — cost/expenditure trends, sector and ministry breakdowns, state-wise views, and top high-value project rankings.
- **Supports single, batch/history and CSV inputs** — a reviewer can check one project interactively, upload a CSV of many projects, or feed in a project's full quarterly history for a time-series view.

---

## 4. How It Addresses the Problem

- **Converts descriptive monitoring into predictive monitoring** — instead of only reporting what already happened, the system forecasts what is likely to happen next quarter.
- **Identifies potential problems before they become critical** — early-warning flags mean intervention can happen while it's still cheap and effective.
- **Helps detect cost and schedule risks early** — classification models surface risk probability well before an overrun shows up plainly in the raw numbers.
- **Prioritizes high-risk projects for intervention** — reviewers with limited bandwidth can focus on the small subset of projects the model flags as high risk, instead of reviewing everything uniformly.
- **Provides explainable insights, not just predictions** — SHAP-based reasoning and LLM summaries mean every prediction is accompanied by a "why," not just a probability.
- **Reduces manual analysis of large project datasets** — the system automates the cross-checking work a reviewer would otherwise do by hand across thousands of PDF-derived records.
- **Supports faster and evidence-based decision-making** — dashboards, quarterly trends, and risk scores give decision-makers a single, current view instead of scattered reports.

**Flow:** Detect → Predict → Explain → Prioritize → Act

---

## 5. Uniqueness

- **Combines multiple AI/ML techniques in one system**, each answering a different question:
  - **Classification** — Is the project at risk?
  - **Regression** — How much overrun can occur?
  - **Anomaly Detection** — Is the project behaving unusually compared to its history/peers?
- **Provides explainable reasons for predictions**, rather than treating the models as black boxes.
- **Generates a combined project risk score** that fuses classification, regression, and anomaly signals into one number a reviewer can act on.
- **Uses an LLM to convert technical outputs into simple summaries**, closing the gap between data-science output and a decision-maker's reading level.
- **Supports both current project data and project history**, so predictions can be made on a single snapshot or trended across a project's full reporting timeline.
- **Built around existing CUF fields**, meaning it plugs into the reporting structure the ministry already uses instead of requiring a new data-collection format.

---

## 6. Methodology & Implementation

**Data Acquisition**
- Collect historical PAIMANA data from quarterly reports.
- Extract data from PDFs, since most historical records exist only in that format.
- Clean, validate and standardize the dataset into a consistent, model-ready schema.

**Data Processing**
- Use existing CUF (Common Update Format) fields as the backbone of the schema, so no new reporting format is imposed on data providers.
- Handle missing and inconsistent data through imputation and validation rules.
- Perform feature engineering — deriving fields such as progress ratio, spending trend, project age, and reporting frequency from the raw quarterly figures.

**Model Development**
- **Classification** → predicts cost-overrun risk and time-overrun risk independently, as a probability per target.
- **Regression** → estimates the overrun magnitude (percentage and absolute figures) for projects flagged as at risk.
- **Anomaly Detection** → scores how unusual a project's current reporting pattern is relative to its own history and to comparable projects.

**Risk Analysis**
- Combine model outputs from all three services into a single view per project.
- Generate an overall risk score (Low / Medium / High) from the combined signals.
- Identify the major factors contributing to that risk, ranked by their contribution.

**LLM Layer**
- Summarizes model predictions — classification probability, overrun estimate, and anomaly detail — into one coherent narrative.
- Converts technical results into human-readable insights suitable for a non-technical project officer.

**Dashboard**
- Project details and exploration — search any project by code, name, agency, sector, or state.
- Risk analysis — per-project and portfolio-level risk views.
- Graphs and visual analytics — cost/expenditure trends, sector and ministry breakdowns, quarterly reporting patterns.
- Multiple input methods — single project form, batch input, and CSV upload.

---

## 7. Technology Stack

**Current Prototype**

| Technology | Purpose |
|---|---|
| Python | Core development language across the ML and backend layers |
| Pandas / NumPy | Data processing, cleaning, and feature engineering |
| Scikit-learn | Classification, regression, and anomaly-detection models |
| Plotly / Matplotlib | Visualization of model outputs and trends |
| Streamlit | Interactive dashboard prototyping |
| LLM (LangChain + LangGraph, Gemini primary / Qwen fallback) | Intelligent summarization of risk analysis |
| CSV / Structured Data | Current data storage format |

**Future Technologies**

| Technology | Purpose |
|---|---|
| PAIMANA API / Live Data Integration | Replace static CSV ingestion with live reporting data |
| PostgreSQL | Structured, queryable storage at scale |
| Apache Kafka / Spark | Handle large-scale, streaming project data |
| MLflow | Model versioning, tracking, and management |
| Docker / Cloud | Containerized, scalable deployment |
| Automated model retraining | Keep models current as new quarterly data arrives |
| LLM-powered chatbot | Conversational access to project risk data |
| Automated alerts and notifications | Push high-risk flags directly to responsible officers |

---

## 8. Architecture

See [docs/architecture.md](docs/architecture.md) for the full pipeline diagram.

```text
Raw PAIMANA quarterly reporting data (PDF → structured CSV)
  |
  v
Data cleaning, validation & feature engineering (CUF fields)
  |
  +----> Anomaly detection (unusual project behaviour)
  |
  v
Risk classification  (cost_overrun_risk, time_overrun_risk)
  |
  v
 at risk? ---- no ----> flagged low-risk, no overrun estimate computed
  |
 yes
  |
  v
Overrun magnitude regression (how much overrun may occur)
  |
  v
Risk Analysis engine  (combined project risk score + contributing factors)
  |
  v
LLM Layer  (plain-language summary of the full risk analysis)
  |
  v
Interactive Dashboard  (single / batch / CSV input, graphs & visual analytics)
```

Each stage is decoupled: the anomaly detector and the classifier both run on the cleaned, feature-engineered data independently, while the regression stage only runs for projects the classifier has already flagged as at risk — keeping the expensive computation focused on the projects that actually need it. The Risk Analysis engine is the point where all three model outputs are fused into one score, which is what both the LLM layer and the dashboard consume downstream.

---

## 9. Challenges

**Data Acquisition**
- PAIMANA does not provide easily accessible public structured data.
- Historical data is primarily available in quarterly PDFs, not clean tabular files.
- PDF extraction resulted in formatting and data errors that had to be caught and corrected before modeling.

**Data Quality**
- Missing and inconsistent values across quarters and projects.
- Different formats across reports, sometimes even within the same ministry over time.
- Duplicate or incomplete records that needed de-duplication logic.
- Required extensive cleaning and validation before the data was reliable enough to train on.

**Development**
- Building multiple models for different prediction tasks (classification, regression, anomaly detection) that all had to stay consistent with each other.
- Combining predictions from independent models into a single, meaningful risk score without double-counting risk signals.
- Making predictions explainable, not just accurate — this meant integrating SHAP-based reasoning on top of the trained models.
- Integrating ML results with an LLM reliably, including handling provider failures gracefully (fallback model routing).

---

## 10. Feasibility, Viability & Practical Implementation

**Feasibility**
- Uses open-source technologies throughout, keeping the system free of vendor lock-in.
- Works with existing CUF fields, so it doesn't require a new data-collection standard.
- Can operate initially on historical data alone, with no dependency on a live API to get started.
- Modular architecture (classification / overrun / anomaly / gateway as separate services) allows any one model to be upgraded or retrained without touching the others.
- Can integrate with live data when access is available, without a redesign.

**Viability**
- Solves a real infrastructure monitoring problem faced by reviewers today.
- Reduces manual project analysis, freeing reviewer time for the projects that actually need attention.
- Helps prioritize high-risk projects instead of treating every project with equal scrutiny.
- Can continuously improve with new project data, since the models are designed to be retrained as more history accumulates.
- Can scale across ministries and sectors, since the schema is built around a shared reporting format rather than one department's specific fields.

**Practical Implementation**

| Stage | Description |
|---|---|
| Historical Data → Model Training | Use the cleaned historical dataset to train and validate all three model types |
| PAIMANA Integration → Automated Data Ingestion | Replace manual CSV handling with a live data feed |
| Risk Engine → Continuous Risk Scoring | Score every project automatically as new quarterly data arrives |
| Alerts → Notify Responsible Officers | Push high-risk flags directly to the people who can act on them |
| Dashboard → Monitor Projects and Trends | Give reviewers an always-current view of portfolio risk |
| Future → Large-scale Government Deployment | Scale the system across ministries once the pilot is validated |

---

## 11. Final Presentation

See [submission/PRESENTATION.md](submission/SIH26103-NIRMAAN_O3-PRESENTATION.pdf) for the required format.
You can also see the ppt at (https://drive.google.com/file/d/1nvShelR5IRA0UcZmj8oIjdul3T6QGoLb/view?usp=sharing)

## 12. Demo Video

[submission/DEMO.md](submission/demo.md).

## 13. Screenshots / Prototype

See [assets/screenshots/README.md](screenshots/SCREENSHOTS_README.md) for naming conventions.

## 14. Future Scope

- **Sector-specialised models (mixture-of-experts):** the current regression architecture uses a unified ensemble that generalises across sectors; a sector-interaction variant is an early step toward per-sector specialisation, to be expanded as more sector-level history accumulates. As more data accumulates per sector, dedicated sub-models can capture sector-specific overrun patterns (e.g. railways vs. atomic energy) that a single unified model would otherwise average away.
- **Model drift monitoring:** track prediction accuracy against newly reported outcomes over time, since current validation only certifies performance up to the current data cutoff. This would allow automatic flagging when a model's real-world accuracy starts to degrade, triggering a retrain before predictions become unreliable.
- **Live PAIMANA API integration:** move from static CSV ingestion to a continuously updating data feed, so risk scores stay current without manual re-uploads.
- **Automated alerting:** notify responsible project officers directly when a project crosses into high-risk territory, rather than requiring someone to check the dashboard.
- **Conversational access via an LLM chatbot:** let reviewers ask natural-language questions about portfolio risk instead of navigating dashboard panels manually.
- See Section 7 ("Future Technologies") and Section 10 ("Practical Implementation") for the planned integration, scaling, and deployment roadmap.

## 15. A complete report of the model evaluation and the comparison between statistical models(conventional) vs ML models(our approach)
See [docs/report.md](docs/report.md) for the full report in detail.

---

# PAIMANA Combined Prediction API

One FastAPI service exposing three independently trained models behind one process:

| Service | Prefix | What it predicts | Trained on |
|---|---|---|---|
| Overrun-risk classification | `/classification/*` | Probability of a cost/time overrun ("Overrun Risk" / "No Overrun") | `paiman_cost_overrun_ensemble.joblib` / `paiman_time_overrun_ensemble.joblib` |
| Cost/time overrun regression | `/overrun/*` | `final_cost_overrun_pct` **and** `final_time_overrun_pct` (two separate models) | `data/paiman_projects_landmark_dataset.csv` |
| Hybrid anomaly detection | `/anomaly/*` | Per-quarter anomaly score / risk level for a project | `data/paiman_projects_for_anomaly_detection.csv` |

Each service keeps its own dataset, feature engineering, training script, and
model artifact(s) — they are combined only at the FastAPI-app level (shared
error handling, shared request-id logging, one process to deploy). **None of
the three services' code imports from another** (see `app/gateway/` below for
how they're chained together instead).

## The gating rule

`app/overrun` (regression) and `app/anomaly` (anomaly detection) are
expensive relative to the classifier, and only meaningful for a project the
classifier already flagged as at risk. `POST /gateway/predict` enforces that:

1. It calls `/classification` first and gets back a probability per target
   (cost, time).
2. **Only if that probability is strictly greater than 0.5** does it call
   `/overrun` for that target; same for `/anomaly` (gated on either target's
   probability exceeding 0.5).
3. Otherwise the response says which service(s) were skipped and why.

The 0.5 cutoff is independent of the classifier's own tuned decision
threshold (visible in its `reason` text) — the gate always compares the raw
probability to 0.5, not to whatever threshold the model itself uses for its
"Overrun Risk" / "No Overrun" label. It's configurable via the
`OVERRUN_RISK_PROBABILITY_THRESHOLD` env var if you ever need a different
cutoff. See `app/gateway/router.py` for the full implementation.

Each of the three services can also still be called directly and
independently (`/classification/predict/batch`, `/overrun/predict`,
`/anomaly/predict/batch`, ...) without going through the gateway at all —
the gateway is an added orchestration layer, not a replacement for the
standalone endpoints.

---

# Running the Website

This guide walks through everything needed to get the app running locally — backend API and frontend — from a clean checkout.

## Prerequisites

- Python 3.10+ and `pip`
- Node.js 18+ and `npm`
- Git

## Project Layout

```
.
├── app/              # FastAPI backend
├── frontend/         # Website (frontend)
├── models/           # Pre-trained model artifacts (shipped, ready to use)
├── data/             # Datasets
├── requirements.txt  # Backend Python dependencies
└── .env              # Backend environment variables (you create this)
```

## Step 1: Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

## Step 2: Set up environment variables

Two separate `.env` files are required — one for the backend, one for the frontend.

**Root folder `.env`** (backend)

Create a file named `.env` in the project root and add the required keys, for example:

```env
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
LOG_LEVEL=INFO
```

**Frontend folder `.env`**

Create a second `.env` file inside the `frontend/` folder for any frontend-specific variables, for example:

```env
VITE_API_BASE_URL=http://localhost:8000
```

> Fill in both `.env` files before starting the servers in the next steps — the app will not run correctly without them.

## Step 3: Install backend requirements

From the project root:

```bash
pip install -r requirements.txt
```

## Step 4: Start the backend — Terminal 1

Open a terminal in the project root and run:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

This starts the FastAPI server locally at `http://localhost:8000`. Leave this terminal running.

## Step 5: Start the frontend — Terminal 2

Open a **second** terminal, navigate into the `frontend` folder, and run the following three commands in order:

```bash
npm install
npm run build
npm run dev
```

- `npm install` — installs all frontend dependencies
- `npm run build` — creates a production build
- `npm run dev` — starts the local development server

## Step 6: Open the website

Once both terminals are running:

- **Frontend:** the URL printed by `npm run dev` (typically `http://localhost:5173`)
- **Backend API docs:** `http://localhost:8000/docs`

## Stopping the app

Press `Ctrl + C` in each terminal to stop the backend and frontend servers.

## Troubleshooting

- **Frontend can't reach the backend:** confirm the backend terminal is still running and that `VITE_API_BASE_URL` in `frontend/.env` matches the backend's address.
- **Missing API keys:** GenAI/summary endpoints return `503` if `GOOGLE_API_KEY` (and optionally `HUGGINGFACEHUB_API_TOKEN`) aren't set in the root `.env`; other endpoints still work without them.
- **Port already in use:** change `--port 8000` to a free port in Step 4, and update `VITE_API_BASE_URL` accordingly.
