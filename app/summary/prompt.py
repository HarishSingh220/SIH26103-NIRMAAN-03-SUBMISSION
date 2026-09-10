"""
Builds the chat messages sent to the LLM from a `ProjectFacts` object.

Kept deliberately simple: a system prompt that fixes the tone/length/rules,
and a user turn that lists only the facts we actually have (never a field
that was `None`), so the model isn't tempted to invent values for data it
wasn't given.
"""
from __future__ import annotations

from .extractor import ProjectFacts

SYSTEM_PROMPT = (
    "You are a project-monitoring assistant for PAIMANA, a government "
    "infrastructure project tracking system. You are given a project's "
    "cost-overrun risk, time-overrun risk, and reporting-anomaly risk, each "
    "with a short machine-generated reason. Write a brief, plain-language "
    "summary of what is happening with the project for a non-technical "
    "reader (a project officer, not a data scientist).\n\n"
    "Rules:\n"
    "- 3 to 6 sentences, plain prose, no bullet points, no headings.\n"
    "- Base the summary ONLY on the facts given below; never invent numbers, "
    "dates, or causes that were not provided.\n"
    "- Mention cost-overrun risk, time-overrun risk, and the anomaly finding "
    "at least briefly, but do not just restate labels -- explain what they "
    "mean for the project in plain words.\n"
    "- If a risk is low/normal, say so plainly instead of dwelling on it.\n"
    "- Close with one short sentence on what, if anything, deserves "
    "attention next."
)


def _fmt_pct(value) -> str:
    return f"{value:.1f}%" if isinstance(value, (int, float)) else "unknown"


def build_user_prompt(facts: ProjectFacts, project_label: str) -> str:
    lines: list[str] = [f"Project: {project_label}", ""]

    lines.append("Cost overrun risk:")
    if facts.cost_overrun_prediction is not None:
        lines.append(f"- Classifier prediction: {facts.cost_overrun_prediction}")
        if facts.cost_overrun_probability is not None:
            lines.append(f"- Predicted probability of overrun: {_fmt_pct(facts.cost_overrun_probability * 100)}")
        if facts.cost_overrun_reason:
            lines.append(f"- Reason given: {facts.cost_overrun_reason}")
        if facts.cost_overrun_pct is not None:
            lines.append(f"- Estimated cost overrun magnitude: {_fmt_pct(facts.cost_overrun_pct)}")
        if facts.predicted_cost_increase_rs_cr is not None:
            lines.append(f"- Estimated cost increase: Rs {facts.predicted_cost_increase_rs_cr:.2f} crore")
        if facts.predicted_final_cost_rs_cr is not None:
            lines.append(f"- Estimated final cost: Rs {facts.predicted_final_cost_rs_cr:.2f} crore")
    else:
        lines.append("- Not available.")

    lines.append("")
    lines.append("Time overrun risk:")
    if facts.time_overrun_prediction is not None:
        lines.append(f"- Classifier prediction: {facts.time_overrun_prediction}")
        if facts.time_overrun_probability is not None:
            lines.append(f"- Predicted probability of overrun: {_fmt_pct(facts.time_overrun_probability * 100)}")
        if facts.time_overrun_reason:
            lines.append(f"- Reason given: {facts.time_overrun_reason}")
        if facts.predicted_delay_months is not None:
            lines.append(f"- Estimated delay: {facts.predicted_delay_months:.1f} months")
        if facts.predicted_final_commissioning_date:
            lines.append(f"- Estimated final commissioning date: {facts.predicted_final_commissioning_date}")
    else:
        lines.append("- Not available.")

    lines.append("")
    lines.append("Reporting anomaly risk:")
    if facts.anomaly_risk_level is not None:
        lines.append(f"- Risk level: {facts.anomaly_risk_level}")
        if facts.anomaly_score is not None:
            lines.append(f"- Anomaly score: {facts.anomaly_score}")
        if facts.anomaly_reason:
            lines.append(f"- Reason given: {facts.anomaly_reason}")
    else:
        lines.append("- Not available (anomaly detection was not run for this project).")

    if facts.warnings:
        lines.append("")
        lines.append("Data-quality warnings from the pipeline: " + "; ".join(facts.warnings))

    lines.append("")
    lines.append("Write the summary now.")
    return "\n".join(lines)


def build_messages(facts: ProjectFacts, project_label: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(facts, project_label)},
    ]
