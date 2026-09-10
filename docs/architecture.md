# NIRMAAN — Infrastructure Project Intelligence & Risk Analysis

## System Architecture

NIRMAAN is a web-based infrastructure project monitoring and risk-analysis platform built around PAIMANA project data. It combines a React frontend, FastAPI backend services, machine-learning models for overrun and anomaly analysis, a GenAI explanation layer, and SQLite-based authentication.

## High-level flow

```text
                         PAIMANA Project Data
                                  |
                                  v
                    +---------------------------+
                    | Data Preparation /        |
                    | Feature Engineering       |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | Offline ML Training       |
                    |                           |
                    | • Risk Classification     |
                    | • Cost Overrun Regression |
                    | • Time Overrun Regression |
                    | • Anomaly Detection        |
                    +-------------+-------------+
                                  |
                         Trained Model Artifacts
                                  |
                                  v
+-------------+        +---------------------------+
|    User     | -----> | React + Vite Frontend     |
|             |        |                           |
| Login /     |        | • Dashboard               |
| Register    |        | • Project Overview        |
| Project     |        | • Risk Analysis           |
| Analysis    |        | • AI Assistant            |
+-------------+        +-------------+-------------+
                                      |
                                      | HTTP / JSON / CSV
                                      v
                         +---------------------------+
                         | FastAPI Backend            |
                         |                           |
                         | Authentication             |
                         | Risk Pipeline / Gateway    |
                         | Classification             |
                         | Cost/Time Overrun          |
                         | Anomaly Detection          |
                         | GenAI Summary              |
                         +---+----------+----------+--+
                             |          |          |
                             v          v          v
                      Classification  Regression  Anomaly
                       Risk Models   Models       Model
                             \          |          /
                              \         |         /
                               +--------+--------+
                                        |
                                        v
                              Risk Prediction Result
                                        |
                              +---------+----------+
                              |                    |
                              v                    v
                       Risk Analysis UI       GenAI Summary
                                             Gemini / Qwen
                              |
                              v
                           User
```

## Components

### Frontend

The frontend is implemented with **React** and **Vite**.

It handles:

- User registration and login.
- Project dashboard and project overview.
- Project-wise risk analysis.
- Single, batch, and CSV-based prediction input.
- Visualization of risk results, overrun predictions, anomaly scores, and historical trends.
- AI-generated project summaries through the backend GenAI service.

Main frontend pages include:

```text
Home
Project Overview
Risk Analysis
AI Assistant
About
Login
Register
```

The frontend also uses the supplied PAIMANA CSV dataset for the project-overview dashboard and visual analytics.

### Backend API

The backend is implemented using **FastAPI** and serves as the central application layer.

It is responsible for:

- Request validation using Pydantic schemas.
- Authentication and JWT-based session handling.
- Feature derivation from raw project-reporting fields.
- Coordination of classification, regression, and anomaly models.
- Batch and CSV processing.
- Returning structured prediction results to the frontend.
- Generating explanatory project summaries through the GenAI layer.
- Health checks and model-status endpoints.

The main backend routes are:

```text
/api/auth/*
/classification/*
/overrun/*
/anomaly/*
/gateway/*
/predict-risk/*
/summary/*
```

### Risk Classification

The **classification service** predicts whether a project is at risk of:

- Cost overrun
- Time overrun

The service uses trained ensemble artifacts and provides probability, predicted label, and SHAP-based reasoning.

Classification acts as the first stage of the main risk pipeline.

### Cost and Time Overrun Regression

The **overrun service** estimates the magnitude of:

- `final_cost_overrun_pct`
- `final_time_overrun_pct`

Two trained model pipelines are loaded independently:

```text
models/final_cost_overrun_pct_combined_model.pkl
models/final_time_overrun_pct_combined_model.pkl
```

The service can return both model predictions and the selected preferred prediction.

### Anomaly Detection

The **anomaly service** identifies unusual or potentially problematic project-reporting patterns.

It produces:

- Anomaly score
- Risk level
- Human-readable anomaly reason
- Latest project prediction
- Quarterly predictions for historical trend analysis

The trained artifact is:

```text
models/anomaly_model.joblib
```

Projects with multiple reporting rows are treated as longitudinal history. The anomaly service uses longer histories for stronger trend-based scoring while supporting short-history cases as well.

### Integrated Risk Pipeline / Gateway

NIRMAAN combines the three ML stages into a unified pipeline:

```text
Raw Project Data
      |
      v
Feature Engineering
      |
      v
Risk Classification
      |
      v
Probability-based Risk Gate
      |
      +----------------------+
      |                      |
      v                      v
Cost/Time Regression     Anomaly Detection
      |                      |
      +----------+-----------+
                 |
                 v
          Unified Risk Result
```

The classifier's probability is used to decide whether the downstream cost/time regression and anomaly-analysis stages should run for a target.

By default, the risk-gate threshold is **0.5**.

The main integrated prediction flow is available through the classification pipeline, while `/gateway/*` remains available as a backward-compatible alias.

### Shared Feature Engineering

NIRMAAN uses a shared raw-input schema so that classification, regression, and anomaly services work from the same project data.

The backend derives features internally, including:

```text
quarter
reporting_period_date
project_age_at_report_months
planned_duration_months
progress_ratio
expenditure_to_cost_pct
contemporaneous_cost_overrun_pct
contemporaneous_time_overrun_pct
anticipated_cost_change_pct
anticipated_date_change_months
landmark_index
n_landmarks_total
horizon_months
```

This means the frontend supplies raw prediction-time project information instead of manually calculating engineered ML features.

### GenAI Summary Layer

The **GenAI summary service** converts prediction results into a human-readable project explanation.

The service uses:

```text
Primary:   Google Gemini
Fallback:  Hugging Face Qwen
```

The GenAI layer is implemented using **LangChain / LangGraph** and can generate summaries for individual or multiple projects.

Data flow:

```text
Prediction Result
       |
       v
GenAI Summary Service
       |
       +--> Gemini
       |
       +--> Qwen fallback
       |
       v
Human-readable Project Summary
```

### Authentication and Database

NIRMAAN includes application authentication using:

```text
JWT access tokens
SQLite database
PBKDF2-HMAC-SHA256 password hashing
```

The SQLite database stores user account information, while the frontend keeps the issued authentication token locally for protected application routes.

Authentication endpoints include:

```text
POST /api/auth/register
POST /api/auth/login
GET  /api/auth/me
POST /api/auth/logout
```

### Data Layer and Model Artifacts

The backend repository contains PAIMANA-derived datasets used for model training and project analysis:

```text
data/
├── paiman_projects_landmark_dataset.csv
└── paiman_projects_for_anomaly_detection.csv
```

Trained artifacts are stored under:

```text
models/
├── anomaly_model.joblib
├── final_cost_overrun_pct_combined_model.pkl
├── final_time_overrun_pct_combined_model.pkl
├── cost_overrun_feature_schema.json
├── time_overrun_feature_schema.json
└── classification/
    ├── paiman_cost_overrun_ensemble.joblib
    └── paiman_time_overrun_ensemble.joblib
```

Models are trained offline and loaded by the FastAPI application during startup; prediction requests do not retrain the models.

## End-to-end Data Flow

### Single Project

```text
User enters project data
        |
        v
React Risk Analysis page
        |
        v
FastAPI classification pipeline
        |
        v
Shared feature engineering
        |
        v
Cost-risk + time-risk classification
        |
        v
Risk gate
        |
        +----> Cost overrun regression
        |
        +----> Time overrun regression
        |
        +----> Anomaly detection
        |
        v
Unified prediction response
        |
        +----> Risk Analysis charts/tables
        |
        +----> Reasons / explanations
        |
        +----> GenAI project summary
```

### Batch / Historical Project Analysis

```text
Multiple project rows
        |
        v
Group by project_code
        |
        v
Use quarterly reporting history
        |
        v
Feature engineering
        |
        v
ML prediction for each project
        |
        v
Latest prediction
        +
Quarterly predictions
        |
        v
Project-wise risk distribution
        +
Historical trend graphs
```

### CSV Analysis

```text
CSV Upload
    |
    v
FastAPI CSV parser
    |
    v
Schema validation
    |
    v
Feature engineering
    |
    v
Classification / Regression / Anomaly
    |
    v
Project-wise results
    |
    v
Frontend visualization
```

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite, React Router |
| Backend | FastAPI, Uvicorn, Pydantic |
| Data Processing | Python, Pandas, NumPy |
| Classification | Scikit-learn, XGBoost-based ensemble artifacts |
| Regression | XGBoost, LightGBM, CatBoost, Scikit-learn |
| Explainability | SHAP |
| Anomaly Detection | Custom hybrid anomaly-scoring pipeline |
| GenAI | Google Gemini, Hugging Face Qwen, LangChain, LangGraph |
| Authentication | JWT, SQLite, PBKDF2-HMAC-SHA256 |
| Deployment | Docker |

## Project Structure

```text
NIRMAAN/
├── app/
│   ├── main.py
│   ├── auth/
│   ├── classification/
│   ├── overrun/
│   ├── anomaly/
│   ├── gateway/
│   ├── predict_risk/
│   ├── summary/
│   └── common/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── data/
│   │   ├── styles/
│   │   └── utils/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── data/
├── models/
├── tests/
├── nirmaan_auth.db
├── requirements.txt
└── Dockerfile
```

## Key Design Principle

NIRMAAN follows a **single raw-input, multi-model prediction architecture**.

The client supplies project reporting data once. The backend performs shared feature engineering and reuses the resulting features across the risk-classification, overrun-regression, and anomaly-detection services. This keeps the prediction pipeline consistent while allowing each ML component to remain independently testable and callable.

The final result brings together:

```text
Risk Classification
        +
Cost Overrun Prediction
        +
Time Overrun Prediction
        +
Anomaly Detection
        +
Explainable Reasons
        +
GenAI Summary
        =
Infrastructure Project Intelligence
```
