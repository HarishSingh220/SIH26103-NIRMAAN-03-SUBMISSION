"""Pydantic schemas and examples for the GenAI summary endpoints."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


SUMMARY_PREDICTION_EXAMPLE = {
    "project_code": "P001",
    "classification": {
        "cost_overrun": {"prediction": "Overrun Risk", "probability": 0.76, "reason": "Anticipated cost is above the original cost."},
        "time_overrun": {"prediction": "Overrun Risk", "probability": 0.68, "reason": "Anticipated commissioning date has moved later."},
    },
    "risk_gate": {"cost_overrun_risk": True, "time_overrun_risk": True, "any_overrun_risk": True},
    "overrun": {
        "cost_overrun": {"predicted_overrun_pct": 7.2, "predicted_final_cost_rs_cr": 214.4},
        "time_overrun": {"predicted_overrun_pct": 8.1, "predicted_delay_months": 3.1},
    },
    "anomaly": {"risk_level": "Review", "anomaly_score": 0.61, "anomaly_reason": "Reporting trend changed materially."},
}

SUMMARY_BATCH_EXAMPLE = {
    "predictions": [SUMMARY_PREDICTION_EXAMPLE, {**SUMMARY_PREDICTION_EXAMPLE, "project_code": "P002"}],
}


class SummaryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"prediction": SUMMARY_PREDICTION_EXAMPLE, "project_name": "Doubling of XYZ Rail Line"}})

    prediction: dict[str, Any] = Field(..., description="One project's prediction-endpoint output, as-is.")
    project_name: Optional[str] = Field(None, description="Optional display name for the project.")


class SummaryBatchRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": SUMMARY_BATCH_EXAMPLE})

    predictions: list[dict[str, Any]] = Field(..., min_length=1, description="Prediction output for multiple projects.")


class SummaryResponse(BaseModel):
    project_code: Optional[str] = None
    summary: str
    model: str
    warnings: list[str] = Field(default_factory=list)


class SummaryBatchResponse(BaseModel):
    model: str
    summaries: list[SummaryResponse]
