# NIRMAAN Risk Analysis integration

The Risk Analysis page now uses the combined classification pipeline directly rather than the legacy `/predict-risk` wrapper.

## Prediction endpoints

All prediction requests use the same raw PAIMANA schema.

- `POST /classification/predict` — exactly one raw row as a JSON array.
- `POST /classification/predict/batch` — multiple projects, with multiple rows per `project_code` allowed.
- `POST /classification/predict/csv` — CSV upload; multiple rows per project are treated as its reporting history.

The classification pipeline returns, per project:

```json
{
  "project_code": "P001",
  "rows_used": 2,
  "classification": {
    "cost_overrun": {
      "prediction": "Overrun Risk",
      "probability": 0.76
    },
    "time_overrun": {
      "prediction": "Overrun Risk",
      "probability": 0.68
    }
  },
  "risk_gate": {
    "threshold": 0.5,
    "cost_overrun_risk": true,
    "time_overrun_risk": true,
    "any_overrun_risk": true
  },
  "overrun": {
    "cost_overrun": {
      "predicted_overrun_pct": 7.2,
      "predicted_final_cost_rs_cr": 214.4
    },
    "time_overrun": {
      "predicted_overrun_pct": 8.1,
      "predicted_delay_months": 3.1
    },
    "quarterly": []
  },
  "anomaly": {
    "risk_level": "Review",
    "anomaly_score": 0.61,
    "anomaly_reason": "Reporting trend changed materially.",
    "quarterly": []
  }
}
```

For batch/CSV requests, `overrun.quarterly` and `anomaly.quarterly` contain the row-wise history returned by the backend. The frontend uses those arrays for time-series charts and the quarterly analysis table.

Regression and anomaly detection are intentionally gated by the classifier. When the relevant classification probability is **less than or equal to 0.5**, the corresponding regression is skipped and the backend returns a reason explaining that it was not run. The frontend displays `NA` instead of fabricating an overrun value.

## GenAI summary

- `POST /summary/generate` — one prediction result.
- `POST /summary/generate/batch` — multiple project prediction results.

The frontend sends the complete returned prediction object to the summary service so the generated explanation can incorporate classification, gate decisions, regression, anomaly detection, and warnings.

## Frontend configuration

Set:

```env
VITE_API_BASE_URL=http://localhost:8000
```

The Risk Analysis page calls the API at:

```text
{VITE_API_BASE_URL}/classification/*
{VITE_API_BASE_URL}/summary/*
```

The legacy `/predict-risk` endpoint is not used by the page.

## Financial Quarter

`financial_quarter` / `financialQuarter` is optional in the Risk Analysis form. The live classification pipeline does **not** require this field: it uses `reporting_quarter` together with `financial_year` and the other raw model fields. When Financial Quarter is left blank, the frontend omits it from the prediction payload, so the model input and prediction logic are unchanged.
