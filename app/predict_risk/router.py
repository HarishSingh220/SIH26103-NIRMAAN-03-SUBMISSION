"""
/predict-risk endpoint — bridges the NIRMAAN frontend to the existing
PAIMANA ML pipeline.

The frontend (RiskAnalysis.jsx) sends one POST per project using 15 camelCase
fields (see frontend/RISK_API.md). This router:

  1. Validates and maps the camelCase frontend payload → the backend's snake_case
     raw schema (app.common.raw_features.RAW_REQUIRED_COLS).
  2. Runs the existing full pipeline: classification → gated regression + anomaly
     (app.classification.pipeline.run_pipeline).
  3. Maps the rich pipeline output back to the simple 4-field response the
     frontend understands:
       { costOverrunRisk, timeOverrunRisk, overallRisk, reason }

The frontend's normalizePrediction() also accepts snake_case aliases
(cost_risk, time_risk, overall_risk, explanation) so we emit both forms for
maximum compatibility.

Risk label mapping:
  probability > 0.70  → "High"
  probability > 0.40  → "Medium"
  else                → "Low"

Sector normalization:
  The frontend dropdown uses title-case values; the training data uses
  ALL-CAPS. We normalise known values; unknown values are uppercased.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.classification import state as classification_state
from app.classification.pipeline import run_pipeline
from app.common.http import run_offloaded

logger = logging.getLogger("paimana.api")

router = APIRouter()

# ──────────────────────────────────────────────────────────────────────────────
# Sector normalization map (frontend label → training data value)
# ──────────────────────────────────────────────────────────────────────────────

_SECTOR_MAP: dict[str, str] = {
    "Road Transport & Highways": "ROAD TRANSPORT AND HIGHWAYS",
    "Road Transport and Highways": "ROAD TRANSPORT AND HIGHWAYS",
    "Railways": "RAILWAYS",
    "Petroleum": "PETROLEUM",
    "Power": "POWER",
    "Coal": "COAL",
    "Urban Development": "URBAN DEVELOPMENT",
    "Other": "OTHER",
}


def _normalize_sector(raw: str) -> str:
    return _SECTOR_MAP.get(raw.strip(), raw.strip().upper())


# ──────────────────────────────────────────────────────────────────────────────
# Request schema (matches RISK_API.md exactly)
# ──────────────────────────────────────────────────────────────────────────────

class ProjectInput(BaseModel):
    projectCode: str = Field(..., min_length=1, max_length=50)
    projectName: str = Field(..., min_length=1)
    agencyName: str = Field(..., min_length=1)
    sector: str = Field(..., min_length=1)
    state: str = Field(..., min_length=1)
    projectStatus: str = Field(..., min_length=1)
    reportingQuarter: str = Field(..., pattern=r"^Q[1-4]$")
    financialQuarter: Optional[str] = Field(None, pattern=r"^Q[1-4]$")
    financialYear: str = Field(..., min_length=4)
    originalCost: float = Field(..., gt=0)
    anticipatedCost: float = Field(..., gt=0)
    cumulativeExpenditure: float = Field(..., ge=0)
    approvalDate: str = Field(..., min_length=8)
    originalCommissioningDate: str = Field(..., min_length=8)
    anticipatedCommissioningDate: str = Field(..., min_length=8)

    @field_validator("financialYear")
    @classmethod
    def validate_fy(cls, v: str) -> str:
        v = v.strip()
        # Accept "2026-27" or "2026" — normalise to "YYYY-YY"
        if "-" not in v and len(v) == 4 and v.isdigit():
            yr = int(v)
            v = f"{yr}-{str(yr + 1)[2:]}"
        return v

    @field_validator("state")
    @classmethod
    def uppercase_state(cls, v: str) -> str:
        return v.strip().upper()


class PredictRiskRequest(BaseModel):
    project: ProjectInput


# ──────────────────────────────────────────────────────────────────────────────
# Mapping helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_raw_row(p: ProjectInput) -> dict[str, Any]:
    """Translate frontend camelCase fields → backend raw schema."""
    return {
        "project_code": p.projectCode,
        "project_name": p.projectName,
        "agency_name": p.agencyName,
        "sector": _normalize_sector(p.sector),
        "state": p.state,
        "project_status": p.projectStatus,
        "reporting_quarter": p.reportingQuarter,
        "financial_year": p.financialYear,
        "original_cost_rs_cr": p.originalCost,
        "anticipated_cost_rs_cr": p.anticipatedCost,
        "cumulative_expenditure_rs_cr": p.cumulativeExpenditure,
        "approval_date": p.approvalDate,
        "original_commissioning_date": p.originalCommissioningDate,
        "anticipated_commissioning_date": p.anticipatedCommissioningDate,
    }


def _probability_to_label(prob: float) -> str:
    if prob > 0.70:
        return "High"
    if prob > 0.40:
        return "Medium"
    return "Low"


def _overall_risk(cost_label: str, time_label: str) -> str:
    rank = {"High": 3, "Medium": 2, "Low": 1}
    return cost_label if rank[cost_label] >= rank[time_label] else time_label


def _build_reason(prediction: dict) -> str:
    """Compose a human-readable reason from the pipeline output."""
    parts: list[str] = []

    # 1. Classification summary
    clf = prediction.get("classification", {})
    cost_clf = clf.get("cost_overrun", {})
    time_clf = clf.get("time_overrun", {})
    cost_prob = cost_clf.get("probability")
    time_prob = time_clf.get("probability")
    cost_pred = cost_clf.get("prediction", "")
    time_pred = time_clf.get("prediction", "")

    if cost_prob is not None:
        parts.append(
            f"Cost overrun risk: {cost_pred} (probability {cost_prob:.1%})."
        )
    if time_prob is not None:
        parts.append(
            f"Time overrun risk: {time_pred} (probability {time_prob:.1%})."
        )

    # 2. Anomaly reason (if available and project was flagged)
    anomaly = prediction.get("anomaly", {})
    anomaly_reason = anomaly.get("anomaly_reason", "")
    risk_level = anomaly.get("risk_level", "Normal")
    if anomaly_reason and risk_level != "Normal" and anomaly_reason != "The project is fine.":
        parts.append(f"Anomaly detection: {anomaly_reason}")

    # 3. Overrun magnitude (if regression ran)
    overrun = prediction.get("overrun", {})
    cost_block = overrun.get("cost_overrun", {})
    time_block = overrun.get("time_overrun", {})
    cost_pct = cost_block.get("predicted_overrun_pct")
    time_pct = time_block.get("predicted_overrun_pct")
    if cost_pct is not None:
        parts.append(f"Predicted cost overrun: {cost_pct:.1f}%.")
    if time_pct is not None:
        parts.append(f"Predicted time overrun: {time_pct:.1f}%.")

    # 4. Warnings
    warnings = prediction.get("warnings", [])
    if warnings:
        parts.append("Note: " + " ".join(str(w) for w in warnings))

    return " ".join(parts) if parts else "Risk analysis completed by the ML pipeline."


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "",
    summary="Predict project risk (frontend contract)",
    description=(
        "Accepts one project record in the frontend's camelCase schema "
        "(see RISK_API.md) and returns { costOverrunRisk, timeOverrunRisk, "
        "overallRisk, reason } by running the full PAIMANA classification + "
        "gated regression + anomaly-detection pipeline."
    ),
)
async def predict_risk(
    body: PredictRiskRequest = Body(
        ...,
        openapi_examples={
            "example": {
                "summary": "NH Project",
                "value": {
                    "project": {
                        "projectCode": "NH-001",
                        "projectName": "National Highway Extension",
                        "agencyName": "NHAI",
                        "sector": "Road Transport & Highways",
                        "state": "Maharashtra",
                        "projectStatus": "On Track",
                        "reportingQuarter": "Q1",
                        "financialQuarter": "Q1",
                        "financialYear": "2026-27",
                        "originalCost": 500.0,
                        "anticipatedCost": 530.0,
                        "cumulativeExpenditure": 120.0,
                        "approvalDate": "2020-04-01",
                        "originalCommissioningDate": "2024-03-31",
                        "anticipatedCommissioningDate": "2024-09-30",
                    }
                },
            }
        },
    ),
):
    if not classification_state.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models are not loaded. The backend may still be starting up.",
        )

    raw_row = _to_raw_row(body.project)
    raw_df = pd.DataFrame([raw_row])

    try:
        result = await run_offloaded(run_pipeline, raw_df)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("predict_risk pipeline error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction pipeline error: {exc}",
        )

    predictions = result.get("predictions", [])
    if not predictions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The pipeline returned no predictions for this project.",
        )

    prediction = predictions[0]

    # If the pipeline returned a per-project error, surface it clearly
    if "error" in prediction:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Pipeline error for project '{body.project.projectCode}': {prediction['error']}",
        )

    # Extract classification probabilities
    clf = prediction.get("classification", {})
    cost_prob: Optional[float] = clf.get("cost_overrun", {}).get("probability")
    time_prob: Optional[float] = clf.get("time_overrun", {}).get("probability")

    cost_label = _probability_to_label(cost_prob if cost_prob is not None else 0.0)
    time_label = _probability_to_label(time_prob if time_prob is not None else 0.0)
    overall_label = _overall_risk(cost_label, time_label)
    reason = _build_reason(prediction)

    return {
        # Canonical camelCase fields (preferred by the frontend)
        "costOverrunRisk": cost_label,
        "timeOverrunRisk": time_label,
        "overallRisk": overall_label,
        "reason": reason,
        # snake_case aliases (also accepted by frontend's normalizePrediction)
        "cost_risk": cost_label,
        "time_risk": time_label,
        "overall_risk": overall_label,
        "explanation": reason,
        # Raw probabilities — useful for debugging / future frontend enhancements
        "probabilities": {
            "costOverrun": round(cost_prob, 4) if cost_prob is not None else None,
            "timeOverrun": round(time_prob, 4) if time_prob is not None else None,
        },
    }
