# Nirmaan O3 — AI-Powered Infrastructure Risk Monitoring System


## 1. Project Information

- **Project Title:** Nirmaan O3 — AI-Powered Infrastructure Risk Monitoring System
- **PS ID:** SIH26103
- **PS Title:**   Use case on web-based integrated project-monitoring platform
- **Organisation / Ministry:** Ministry of Statistics and Programme Implementation
- **Theme:** Infrastructure Project Monitoring
- **Category:** Software 
- **Team Name:** NRMAAR 
- **Team Members:** HARISH SINGH(TEAM LEADER) - TECH LEAD <br>
                    &nbsp;
                    HARSHITA - USER INTERFACE , PRESENTATION <br>
                    &nbsp;
                    SANSKRITI - USER INTERFACE <br>
                    &nbsp;
                    NAMAN SUDRSHAN - ML LEAD <br>
                    &nbsp;
                    AARUSH RAJ SINGH - ML <br>
                    &nbsp;
                    SURYANSH P. SINGH - BACKEND LEAD <br>
                    &nbsp;

## 2. Problem Statement

Infrastructure projects reported under central monitoring frameworks (PAIMANA quarterly reports) routinely run over budget or behind schedule, but by the time an overrun is obvious it is often too late to intervene usefully. Reviewers need more than a raw reporting dataset: an early signal that a project's reported figures look unusual, a prediction of whether it is heading for a cost or schedule overrun, an estimate of how large that overrun could be, and a plain-language explanation of why — rather than having to manually analyse a large, PDF-heavy historical dataset project by project.

## 3. Idea / Solution / Prototype

- AI-powered infrastructure risk monitoring system
- Predicts risk, cost overruns and time overruns
- Estimates how much overrun may occur
- Detects unusual project behaviour
- Provides reasons behind model predictions
- Generates an overall project risk score
- LLM summarizes the complete risk analysis
- Interactive dashboard for project exploration and visualization
- Supports single, batch/history and CSV inputs

## 4. How It Addresses the Problem

- Converts descriptive monitoring into predictive monitoring
- Identifies potential problems before they become critical
- Helps detect cost and schedule risks early
- Prioritizes high-risk projects for intervention
- Provides explainable insights, not just predictions
- Reduces manual analysis of large project datasets
- Supports faster and evidence-based decision-making

**Flow:** Detect → Predict → Explain → Prioritize → Act

## 5. Uniqueness

- Combines multiple AI/ML techniques in one system
  - Classification: Is the project at risk?
  - Regression: How much overrun can occur?
  - Anomaly Detection: Is the project behaving unusually?
- Provides explainable reasons for predictions
- Generates a combined project risk score
- Uses LLM to convert technical outputs into simple summaries
- Supports both current project data and project history
- Built around existing CUF fields

## 6. Methodology & Implementation

**Data Acquisition**
- Collect historical PAIMANA data from quarterly reports
- Extract data from PDFs
- Clean, validate and standardize the dataset

**Data Processing**
- Use existing CUF fields
- Handle missing and inconsistent data
- Perform feature engineering

**Model Development**
- Classification → Risk / Cost / Time prediction
- Regression → Overrun estimation
- Anomaly Detection → Unusual behaviour

**Risk Analysis**
- Combine model outputs
- Generate overall risk score
- Identify major factors contributing to risk

**LLM Layer**
- Summarizes model predictions
- Converts technical results into human-readable insights

**Dashboard**
- Project details and exploration
- Risk analysis
- Graphs and visual analytics
- Multiple input methods

## 7. Technology Stack

**Current Prototype**
- Python – Core development
- Pandas / NumPy – Data processing
- Scikit-learn – ML models
- Plotly / Matplotlib – Visualization
- Streamlit – Interactive dashboard
- LLM – Intelligent summarization
- CSV / Structured Data – Current data storage

**Future Technologies**
- PAIMANA API / Live Data Integration
- PostgreSQL for structured storage
- Apache Kafka / Spark for large-scale data
- MLflow for model management
- Docker / Cloud for deployment
- Automated model retraining
- LLM-powered chatbot
- Automated alerts and notifications

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

## 9. Challenges

**Data Acquisition**
- PAIMANA does not provide easily accessible public structured data
- Historical data is primarily available in quarterly PDFs
- PDF extraction resulted in formatting and data errors

**Data Quality**
- Missing and inconsistent values
- Different formats across reports
- Duplicate or incomplete records
- Required extensive cleaning and validation

**Development**
- Building multiple models for different prediction tasks
- Combining predictions into a meaningful risk score
- Making predictions explainable
- Integrating ML results with an LLM

## 10. Feasibility, Viability & Practical Implementation

**Feasibility**
- Uses open-source technologies
- Works with existing CUF fields
- Can operate initially on historical data
- Modular architecture allows easy model upgrades
- Can integrate with live data when access is available

**Viability**
- Solves a real infrastructure monitoring problem
- Reduces manual project analysis
- Helps prioritize high-risk projects
- Can continuously improve with new project data
- Can scale across ministries and sectors

**Practical Implementation**
- Historical Data → Model Training
- PAIMANA Integration → Automated Data Ingestion
- Risk Engine → Continuous Risk Scoring
- Alerts → Notify responsible officers
- Dashboard → Monitor projects and trends
- Future → Large-scale government deployment

## 11. Final Presentation

See [submission/PRESENTATION.md](submission/SIH26103-NIRMAAN_O3-PRESENTATION.pdf) for the required format.
You can also see the ppt at (https://drive.google.com/file/d/1nvShelR5IRA0UcZmj8oIjdul3T6QGoLb/view?usp=sharing)

## 12. Demo Video

[submission/DEMO.md](submission/demo.md).

## 13. Screenshots / Prototype

See [assets/screenshots/README.md](screenshots/SCREENSHOTS_README.md) for naming conventions.

## 14. Future Scope

- **Sector-specialised models (mixture-of-experts):** the current regression architecture uses a unified ensemble that generalises across sectors; a sector-interaction variant is an early step toward per-sector specialisation, to be expanded as more sector-level history accumulates.
- **Model drift monitoring:** track prediction accuracy against newly reported outcomes over time, since current validation only certifies performance up to the current data cutoff.
- See Section 7 ("Future Technologies") and Section 10 ("Practical Implementation") for the planned integration, scaling, and deployment roadmap.

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
