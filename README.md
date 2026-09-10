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

See [docs/architecture.md](https://drive.google.com/file/d/1fh3_72chZOkZwIdN2LPNIny9NcwnU8zO/view?usp=sharing) for the full pipeline diagram.

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

Keep the final SIH presentation in the repository whenever the file size allows it.

See [submission/PRESENTATION.md](submission/PRESENTATION.md) for the required format.

If the PPT is too large for GitHub, use Google Drive/OneDrive and put the accessible viewer link in `submission/PRESENTATION.md`.

## 12. Demo Video

Add the YouTube/Google Drive link in [submission/DEMO.md](submission/DEMO.md).

## 13. Screenshots / Prototype

<!-- Prototype images go here. Add files to assets/screenshots/ and update the paths below. -->

**Dashboard / reviewer view**

<!-- ![Dashboard overview](assets/screenshots/dashboard-overview.png) -->

&nbsp;

**Overrun prediction result view**

<!-- ![Prediction result](assets/screenshots/prediction-result.png) -->

&nbsp;

**Anomaly detection view**

<!-- ![Anomaly detection dashboard](assets/screenshots/anomaly-detection.png) -->

&nbsp;

**Risk score / LLM summary view**

<!-- ![Risk summary view](assets/screenshots/risk-summary.png) -->

&nbsp;

**Architecture / system diagram**

<!-- ![System architecture](assets/screenshots/architecture-diagram.png) -->

&nbsp;

See [assets/screenshots/README.md](assets/screenshots/README.md) for naming conventions.

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

## Architecture

```
app/
  main.py                 # combined FastAPI app: lifespan loads all three services'
                           # artifacts, mounts all routers, shared error handling
  core.py                 # backward-compat shim only -- see its docstring
  common/
    http.py               # shared error envelope, request-id logging middleware,
                           # capped CSV-upload reading, offloaded-scoring helper
    schemas.py             # shared /predict/batch row-validation schema
    raw_features.py        # shared raw prediction-time schema + feature derivation
                           # (overrun + anomaly import this; classification does not)
  classification/
    config.py             # model paths + OVERRUN_RISK_PROBABILITY_THRESHOLD (default 0.5)
    schemas.py             # ProjectMetadata / ReportingPeriod / batch request+response models
    preprocessing.py       # mirrors the training pipeline exactly (landmark/quarter features)
    inference.py            # OverrunPredictor: loads one ensemble .joblib, predicts probability
    coordinator.py          # orchestrates preprocessing + both predictors + SHAP reasoning
    explainers.py           # SHAP-based human-readable reasoning for "Overrun Risk" predictions
    state.py                # load/unload/get for the coordinator singleton
    router.py               # /classification/health, /predict/batch
    training/                # offline retraining scripts (train_cost_v4.py, train_time_model.py)
  overrun/
    config.py             # paths, column names, training hyperparameters
    core.py                # feature engineering + CombinedOverrunPipeline (used for both targets)
    train.py              # CLI: builds/retrains the two .pkl artifacts from the raw CSV
    state.py              # load/unload/get_cost/get_time for the two loaded pipelines
    router.py             # /overrun/health, /schema, /predict, /predict/batch, /predict/csv
                           # (predict/predict-item already accepts a classifier's
                           # cost_overrun_risk/time_overrun_risk gate flags -- see gateway/)
  anomaly/
    config.py             # frozen validated model configuration
    core.py                # detector/scoring logic (score_all, fit_hybrid_reference, ...)
    raw_features.py       # derives every engineered feature from raw prediction-time fields
    train.py              # offline reference fitting -> models/anomaly_model.joblib
    state.py              # load/unload/get for the loaded artifact
    router.py             # /anomaly/health, /model-info, /predict/batch, /predict/csv
  gateway/
    router.py             # POST /gateway/predict -- chains classification -> overrun/anomaly
                           # using the 0.5 probability gate (see "The gating rule" above).
                           # Only imports each service's existing pieces; never merges them.
data/
  paiman_projects_for_anomaly_detection.csv   # anomaly training data (already engineered schema)
  paiman_projects_landmark_dataset.csv        # overrun training data (raw landmark schema)
models/
  final_cost_overrun_pct_combined_model.pkl    # overrun: cost-overrun pipeline
  final_time_overrun_pct_combined_model.pkl    # overrun: time-overrun pipeline
  cost_overrun_feature_schema.json
  time_overrun_feature_schema.json
  classification/
    paiman_cost_overrun_ensemble.joblib        # classification: cost-overrun ensemble (XGB+HGB)
    paiman_time_overrun_ensemble.joblib        # classification: time-overrun ensemble (XGB+HGB)
Dockerfile
requirements.txt
```

`the anomaly artifact, both overrun artifacts, and both classification artifacts are all shipped already built, so the
service runs out of the box. Retrain any of them any time its dataset
changes -- see below.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train (offline, per service, never inside a request handler)

### 1a. Anomaly detection

```bash
python -m app.anomaly.train
```

Defaults to `--input data/paiman_projects_for_anomaly_detection.csv --output
models/anomaly_model.joblib --cutoff 2023-24`; override any of the three if
needed. The output directory is created automatically if it doesn't exist.
All three services share one top-level `models/` root; anomaly no longer has a separate artifact subdirectory.

### 1b. Cost + time overrun regression

```bash
# Place the raw CSV wherever you like and point OVERRUN_RAW_DATA_PATH at it,
# or use the default location: ./data/paiman_projects_landmark_dataset.csv
export OVERRUN_RAW_DATA_PATH=/path/to/paiman_projects_landmark_dataset.csv

python -m app.overrun.train --target both
```

This writes, into `MODEL_DIR` (default `./models`):
- `final_cost_overrun_pct_combined_model.pkl`
- `final_time_overrun_pct_combined_model.pkl`
- `cost_overrun_feature_schema.json`
- `time_overrun_feature_schema.json`

`--target cost` or `--target time` retrains just one of the two models.
Retrain any time the underlying data changes -- `train.py` is deterministic
given the same CSV (`RANDOM_STATE` in `app/overrun/config.py`).

Both artifacts are fitted offline and reused at inference time; the API
never retrains anything per request.

### 1c. Overrun-risk classification

The two ensemble artifacts under `models/classification/` are shipped
pre-trained. To retrain either one, run the offline scripts directly (they
have no local-package imports, so they run standalone):

```bash
python -m app.classification.training.train_cost_v4
python -m app.classification.training.train_time_model
```

Both classification trainers now default to the repository's
`data/paiman_projects_landmark_dataset.csv` and write directly to
`models/classification/`. Override those paths with
`CLASSIFICATION_DATA_PATH`, `CLASSIFICATION_COST_MODEL_PATH`, or
`CLASSIFICATION_TIME_MODEL_PATH` when needed.

**One shared `models/` root.** All three services now write their trained
artifacts under a single top-level `models/` directory instead of separate
folders per service -- no manual moving of files after training:
```
models/
├── anomaly_model.joblib                            # anomaly (1a)
├── final_cost_overrun_pct_combined_model.pkl      # overrun (1b)
├── final_time_overrun_pct_combined_model.pkl      # overrun (1b)
├── cost_overrun_feature_schema.json               # overrun (1b)
├── time_overrun_feature_schema.json               # overrun (1b)
└── classification/
    ├── paiman_cost_overrun_ensemble.joblib        # classification (1c)
    └── paiman_time_overrun_ensemble.joblib        # classification (1c)
```
Every training script resolves default paths from the repository root and
creates the destination directory automatically. Running training from a
different current directory no longer changes where data is read or models
are written.

## 2. Start the API

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or:

```bash
docker build -t paimana-combined-api .
docker run --rm -p 8000:8000 paimana-combined-api
```

The repository's `models/` directory is copied into the image, so the
container uses the same paths as local inference. To deploy a refreshed set
of models without rebuilding the image, bind-mount the repository models
directory over `/service/models`.

Startup fails fast (with a clear `RuntimeError`) if any service's artifact
is missing, rather than silently starting and returning 503s from every
prediction endpoint until someone notices.

Swagger/OpenAPI: `http://localhost:8000/docs`

### `GET /health`
Aggregate status across all three services:
```json
{
  "status": "ok",
  "services": {
    "overrun_risk_classification": {"model_loaded": true},
    "cost_time_overrun": {"model_loaded": true},
    "anomaly_detection": {"model_loaded": true}
  }
}
```
Each service also has its own `/classification/health` / `/overrun/health` /
`/anomaly/health`.

---

## 3. Overrun-risk classification -- `/classification/*`

**The single entry point for the whole pipeline.** Every endpoint here
takes the same raw, prediction-time schema as `/overrun/*` and
`/anomaly/*` (see `GET /classification/schema`) -- every engineered
feature is derived internally, and the caller supplies each project's data
exactly once. Internally, this classifies overrun risk, then
automatically runs the regression (`/overrun`) and anomaly-detection
(`/anomaly`) services on the same data, gated on the classifier's own
predicted probability (> 0.5 by default).

### `GET /classification/health`
Model version, the 0.5 risk-gating threshold, and which target each loaded
model predicts.

### `POST /classification/predict/batch`
A bare JSON array of raw rows for one or more projects, grouped by
`project_code` internally (a project_code repeated across rows is always
treated as that project's history -- there is no `include_history`
toggle). `preferred_model` is an optional query parameter (defaults to
`model_b`):
```json
[
  {"project_code": "P001", "reporting_quarter": "Q1", "financial_year": "2023-24", "sector": "RAILWAYS", "...": "..."},
  {"project_code": "P001", "reporting_quarter": "Q2", "financial_year": "2023-24", "...": "next quarter"},
  {"project_code": "P002", "...": "a second project's raw row"}
]
```
Response: `predictions`, one entry per project, each with `classification`
(probability + label + SHAP reason per target), `risk_gate` (the threshold
decision), and `overrun`/`anomaly` -- each of which carries the **latest**
quarter's result at the top level AND a `quarterly` list, one entry per
row received for that project tagged with `reporting_period_date`, so a
caller can plot a time-trend graph. `quarterly` is empty when
regression/anomaly was skipped by the risk gate.

### `POST /classification/predict/csv`
Same raw-schema rows via CSV upload (`multipart/form-data`, field `file`).
Same response shape as `/predict/batch`.

### `POST /classification/predict`
A bare JSON array with **exactly one** project row (422 if you send more
than one -- use `/predict/batch` instead):
```json
[
  {"project_code": "P001", "reporting_quarter": "Q1", "financial_year": "2023-24", "sector": "RAILWAYS", "...": "..."}
]
```
Returns that one project's entry directly (not wrapped in `predictions`).

---

## 4. Anomaly detection -- `/anomaly/*`

**Every endpoint takes only the fields actually available when a quarterly
report comes in** -- not the full engineered schema the model was trained
on. `app/anomaly/raw_features.py` derives every engineered feature
internally; callers never precompute anything.

Required raw fields, per row:
```
project_code, agency_name, sector, state, reporting_quarter,
financial_year, original_cost_rs_cr, anticipated_cost_rs_cr,
cumulative_expenditure_rs_cr, approval_date,
original_commissioning_date, anticipated_commissioning_date
```
Optional (omit entirely and nothing errors -- they just come back as
`null`/unused): `project_name`, `project_status`, `revised_cost_rs_cr`,
`revised_commissioning_date`.

For longitudinal scoring, send multiple quarterly rows for the same
`project_code` in one request:
- **3+ observations for a project → Track A** (longitudinal scoring)
- **1-2 observations → Track B** (short-history peer/data-quality scoring)

### `GET /anomaly/model-info`
Model version, fit cutoff, feature families, and the raw input schema.

### `POST /anomaly/predict/batch`
A bare JSON array of raw rows:
```json
[
  {"project_code":"P001","project_name":"...","agency_name":"...","sector":"...",
   "state":"...","project_status":"Ongoing","reporting_quarter":"Q1","financial_year":"2023-24",
   "original_cost_rs_cr":200.0,"anticipated_cost_rs_cr":210.0,"cumulative_expenditure_rs_cr":20.0,
   "approval_date":"2020-04-01","original_commissioning_date":"2024-03-31",
   "anticipated_commissioning_date":"2024-06-30"},
  {"project_code":"P001", "reporting_quarter":"Q2", "...":"same shape, next quarter"},
  {"project_code":"P002", "...":"a second project's raw row"}
]
```
Response: one entry per project, always including both `latest_prediction`
(project_name, project_code, anomaly_score, risk_level, anomaly_reason --
"The project is fine." when risk_level is "Normal") AND
`quarterly_predictions` -- the same shape for every scored quarter, tagged
with `reporting_period_date`, so a caller can plot a time-trend graph.
There is no `include_history` toggle -- a project's multiple rows are
always treated as its history, and every scored quarter is always
returned.

```json
{
  "model_version": "...",
  "projects_received": 2,
  "rows_received": 4,
  "rows_scored": 4,
  "projects": [
    {"project_code": "P001", "history_rows_received": 2,
     "latest_prediction": {"project_name": "...", "project_code": "P001",
                            "anomaly_score": 0.12, "risk_level": "Normal",
                            "anomaly_reason": "The project is fine."},
     "quarterly_predictions": [
       {"project_name": "...", "project_code": "P001", "anomaly_score": 0.05,
        "risk_level": "Normal", "anomaly_reason": "The project is fine.",
        "reporting_period_date": "2023-06-30"},
       {"project_name": "...", "project_code": "P001", "anomaly_score": 0.12,
        "risk_level": "Normal", "anomaly_reason": "The project is fine.",
        "reporting_period_date": "2023-09-30"}
     ]}
  ]
}
```

### `POST /anomaly/predict/csv`
Same raw-schema rows, uploaded as a CSV file (`multipart/form-data`, field
`file`). Same response shape as `/predict/batch` above (grouped by
project, each with `latest_prediction` and `quarterly_predictions`).
Capped at `MAX_CSV_UPLOAD_BYTES` (default 25 MB) -- read in chunks so an
oversized upload is rejected before being fully buffered into memory.

---

## 5. Cost/time overrun regression -- `/overrun/*`

Predicts `final_cost_overrun_pct` and `final_time_overrun_pct` **magnitude**
using two separate models (`CombinedOverrunPipeline` instances, one per
target).

**Every endpoint takes only the same raw, prediction-time fields the
anomaly-detection service takes** -- not the engineered landmark-dataset
schema either model was trained on. `app/common/raw_features.py` (shared
with `/anomaly/*`; see its module docstring) derives every engineered
feature -- `reporting_period_date`, `expenditure_to_cost_pct`,
`project_age_at_report_months`, `planned_duration_months`, `progress_ratio`,
`landmark_index`, `n_landmarks_total`, `horizon_months`, and the
contemporaneous-overrun deltas -- internally; callers never precompute or
supply any of it. Required raw fields, per row:
```
project_code, agency_name, sector, state, reporting_quarter,
financial_year, original_cost_rs_cr, anticipated_cost_rs_cr,
cumulative_expenditure_rs_cr, approval_date,
original_commissioning_date, anticipated_commissioning_date
```
Optional (omit entirely and nothing errors): `project_name`,
`project_status`, `revised_cost_rs_cr`, `revised_commissioning_date`.

`GET /overrun/schema` returns this same required/optional list plus an
example row, and which sector each model's Model B targets.

Every prediction block has this shape:
```json
{"predicted_overrun_pct": 41.32, "model_b_prediction": 41.32,
 "final_model_prediction": 40.6, "preferred_model": "model_b"}
```
`preferred_model` (`"model_b"` or `"final_model"`) selects which ensemble's
value is surfaced as `predicted_overrun_pct`; both are always included.

### `POST /overrun/predict` -- exactly ONE project row
Standalone regression: always predicts both targets, no classifier gating
(that gating lives in `/classification/predict`, which calls this same
regression internally). A bare JSON array with **exactly one** row (422 if
you send more than one -- use `/predict/batch` instead). `preferred_model`
is an optional query parameter:
```json
[
  {
    "project_code": "P001", "project_name": "Doubling of XYZ Rail Line",
    "agency_name": "Ministry of Railways", "sector": "RAILWAYS",
    "state": "MAHARASHTRA", "project_status": "Ongoing",
    "reporting_quarter": "Q1", "financial_year": "2023-24",
    "original_cost_rs_cr": 200.0, "anticipated_cost_rs_cr": 210.0,
    "cumulative_expenditure_rs_cr": 20.0, "approval_date": "2020-04-01",
    "original_commissioning_date": "2024-03-31",
    "anticipated_commissioning_date": "2024-06-30"
  }
]
```

### `POST /overrun/predict/batch`
Bulk equivalent, mirroring the anomaly-detection service's `/predict/batch`
contract: a bare JSON array of raw rows for one or more projects, grouped
by `project_code` internally (never gated by a classifier -- both targets
always run). `preferred_model` is an optional query parameter:
```json
[
  {"project_code": "P001", "reporting_quarter": "Q1", "financial_year": "2023-24", "sector": "RAILWAYS", "...": "..."},
  {"project_code": "P001", "reporting_quarter": "Q2", "financial_year": "2023-24", "...": "next quarter"},
  {"project_code": "P002", "...": "a second project's raw row"}
]
```
Response: always includes both `latest_prediction` AND
`quarterly_predictions` -- the same shape for every supplied quarter,
tagged with `reporting_period_date` (the derived date, not something the
caller supplies), so a caller can plot a time-trend graph. There is no
`include_history` toggle.
```json
{
  "model_version": "...",
  "preferred_model": "model_b",
  "projects_received": 2,
  "rows_received": 3,
  "rows_scored": 3,
  "projects": [
    {"project_code": "P001", "history_rows_received": 2,
     "latest_prediction": {"project_code": "P001",
                            "cost_overrun": {"predicted_overrun_pct": 41.32, "...": "..."},
                            "time_overrun": {"predicted_overrun_pct": 12.05, "...": "..."}},
     "quarterly_predictions": [
       {"project_code": "P001", "reporting_period_date": "2023-06-30",
        "cost_overrun": {"predicted_overrun_pct": 30.1, "...": "..."},
        "time_overrun": {"predicted_overrun_pct": 8.4, "...": "..."}},
       {"project_code": "P001", "reporting_period_date": "2023-09-30",
        "cost_overrun": {"predicted_overrun_pct": 41.32, "...": "..."},
        "time_overrun": {"predicted_overrun_pct": 12.05, "...": "..."}}
     ]}
  ]
}
```

### `POST /overrun/predict/csv`
Same raw-schema rows via CSV upload (`multipart/form-data`, field `file`).
`preferred_model` is a query parameter here instead of a body field; same
response shape as `/predict/batch` (always `latest_prediction` +
`quarterly_predictions`). Same 25 MB chunked-read upload cap as the
anomaly service.

---

## 6. Gated pipeline -- `POST /gateway/predict`

`/gateway/*` is now a thin, backward-compatible alias for
`/classification/predict` and `/classification/predict/batch` -- it
delegates to the exact same pipeline (`app.classification.pipeline`), so
see section 3 above for the request/response shape. New callers should
prefer `/classification/*` directly.

Response shape (single project):
```json
{
  "project_code": "P001",
  "classification": {"cost_overrun": {"probability": 0.31, "...": "..."},
                      "time_overrun": {"probability": 0.23, "...": "..."}},
  "risk_gate": {"threshold": 0.5, "cost_overrun_risk": false,
                "time_overrun_risk": false, "any_overrun_risk": false},
  "overrun": {"cost_overrun": {"reason": "Not flagged as cost-overrun risk by classifier (probability <= 0.5) - regression skipped", "...": "..."},
              "time_overrun": {"...": "..."}, "quarterly": []},
  "anomaly": {"anomaly_reason": "Not flagged as overrun risk by classifier (probability <= 0.5) - anomaly detection skipped", "...": "...", "quarterly": []}
}
```
When `risk_gate.any_overrun_risk` is `true`, `overrun` and `anomaly` contain
real predictions (plus a non-empty `quarterly` trend list) instead of the
skipped placeholders shown above.

`GET /gateway/config` returns the current threshold.

---

## 7. GenAI project summary -- `/summary/*`

Turns prediction output into a short, human-readable project summary using LangChain + LangGraph with Google Gemini as the primary provider and Hugging Face-hosted Qwen as the fallback. The primary model defaults to `gemini-2.5-flash`; the Qwen fallback is configurable.

### Setup

1. Set `GOOGLE_API_KEY` in `.env` for the primary Gemini model.
2. Optionally set `HUGGINGFACEHUB_API_TOKEN` to enable the Qwen fallback.
3. Start the backend normally with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

Without either credential, the prediction services still work and the GenAI endpoints return HTTP 503.

Optional overrides:

| Env var | Default | Purpose |
|---|---|---|
| `SUMMARY_GEMINI_MODEL_ID` | `gemini-2.5-flash` | Primary GenAI model |
| `SUMMARY_QWEN_MODEL_ID` | `Qwen/Qwen3.8-27B` | Hugging Face fallback model |
| `SUMMARY_MAX_NEW_TOKENS` | `350` | Maximum summary output tokens |
| `SUMMARY_TEMPERATURE` | `0.2` | Generation temperature |
| `SUMMARY_REQUEST_TIMEOUT_SECONDS` | `30` | Per-model timeout |
| `SUMMARY_MAX_BATCH_PROJECTS` | `25` | Maximum projects per GenAI batch |

### `GET /summary/health`

Reports whether the OpenRouter key is configured, the primary model, and the fallback model list. It does not call the LLM.

### `GET /summary/schema`

Returns the exact request examples for both GenAI endpoints.

### `POST /summary/generate`

Body:

```json
{
  "prediction": {
    "project_code": "P001",
    "classification": {
      "cost_overrun": {"prediction": "Overrun Risk", "probability": 0.76, "reason": "Anticipated cost is above the original cost."},
      "time_overrun": {"prediction": "Overrun Risk", "probability": 0.68, "reason": "Anticipated commissioning date has moved later."}
    },
    "risk_gate": {"cost_overrun_risk": true, "time_overrun_risk": true, "any_overrun_risk": true},
    "overrun": {},
    "anomaly": {"risk_level": "Review", "anomaly_score": 0.61, "anomaly_reason": "Reporting trend changed materially."}
  },
  "project_name": "Doubling of XYZ Rail Line"
}
```

### `POST /summary/generate/batch`

Body:

```json
{
  "predictions": [
    {"project_code": "P001", "classification": {}, "overrun": {}, "anomaly": {}},
    {"project_code": "P002", "classification": {}, "overrun": {}, "anomaly": {}}
  ]
}
```

The batch endpoint makes one graph invocation per project and automatically falls through the configured model list when a model fails or is unavailable.

### GenAI provider note

The summary service uses Google Gemini first and falls back to Hugging Face-hosted Qwen when the primary provider is unavailable, rate-limited, or otherwise fails with a retryable upstream error.

---
## Error handling & logging

Shared across both services (`app/common/http.py`). Every error response --
a manually raised `HTTPException`, a Pydantic validation failure, or an
unexpected server error -- shares the same JSON shape:
```json
{"detail": "...", "request_id": "..."}
```
`request_id` always matches the `X-Request-Id` response header. A caller may
supply their own `X-Request-Id` request header, echoed back verbatim.

Every request is logged (method, path, status, latency, request ID) at INFO
level by the `paimana.api` logger. Set `LOG_LEVEL` (default `INFO`) to
change verbosity. Request/response bodies are never logged.

Row-level input validation for both `/predict/batch` endpoints runs at the
API boundary, before pandas ever sees the data (`app/common/schemas.py`):
every row must be a non-empty flat mapping of column name to a plain
string/number/boolean/null value -- a nested object or array is rejected
with a 422 naming the row index and field.

## Deployment notes

1. Pin the Python/dependency versions used to benchmark each model.
2. Build both artifacts once offline; never fit models inside request handlers.
3. Keep each artifact and its API model version together.
4. Put authentication/rate limiting/API gateway controls in front of the service.
5. `/predict/csv` uploads (both services) are capped at `MAX_CSV_UPLOAD_BYTES`
   (default 25 MB, overridable via env var) to avoid buffering an unbounded
   file into memory before validation runs.
6. Environment variables of note: `LOG_LEVEL`, `MAX_CSV_UPLOAD_BYTES`,
   `ANOMALY_MODEL_ARTIFACT` (default `models/anomaly_model.joblib`), `DATA_DIR`,
   `MODEL_DIR` (overrun models, default `./models`), `OVERRUN_RAW_DATA_PATH`,
   `DEFAULT_PREFERRED_MODEL` (`model_b` or `final_model`, default `model_b`).
