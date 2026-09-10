"""Business-layer orchestration for one-project and batch summaries."""
from __future__ import annotations

from typing import Any, Optional

from .graph import generate_with_graph
from .extractor import extract_project_facts
from .prompt import build_messages


def summarize_project(prediction: dict[str, Any], project_name: Optional[str] = None) -> dict[str, Any]:
    facts = extract_project_facts(prediction)

    if facts.error:
        return {
            "project_code": facts.project_code,
            "summary": f"No summary was generated: this project's prediction failed ({facts.error}).",
            "model": "none",
            "warnings": facts.warnings,
        }

    if not facts.has_any_signal():
        return {
            "project_code": facts.project_code,
            "summary": (
                "No summary was generated: the supplied prediction did not contain any "
                "recognizable cost-overrun, time-overrun, or anomaly result."
            ),
            "model": "none",
            "warnings": facts.warnings,
        }

    project_label = project_name or facts.project_code or "this project"
    messages = build_messages(facts, project_label)
    summary_text, model_used = generate_with_graph(messages)

    return {
        "project_code": facts.project_code,
        "summary": summary_text,
        "model": model_used,
        "warnings": facts.warnings,
    }
