#!/usr/bin/env python3
"""
Builds src/data/projects-summary.json from the PAIMANA landmark dataset CSV.

The source CSV is a *panel* dataset: each infrastructure project (project_code)
has one row per reporting "landmark" (landmark_index 1..n_landmarks_total),
capturing how its cost/expenditure/schedule evolved over time. For a
dashboard we need the *current* snapshot of every project, so for each
project_code we keep only the row with the highest landmark_index (its most
recent reported state).

Run: python3 scripts/build_data.py <path-to-csv> <output-json-path>
"""
import csv
import json
import sys
from collections import defaultdict

def f(v):
    """Parse a CSV numeric field; blank/invalid -> None."""
    if v is None:
        return None
    v = v.strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def title_case(s):
    return " ".join(w.capitalize() for w in s.split(" "))


def classify_status(cost_flag, time_flag, cost_pct, time_pct):
    """
    Classify a project's delivery status from its FINAL cost/time overrun
    outcome fields (which the dataset holds constant across a project's
    landmarks once known).
      - reporting:      outcome not yet determined (too early in lifecycle)
      - on-track:       no cost or time overrun recorded
      - minor-overrun:  an overrun exists but stays within +20%
      - at-risk:        overrun exceeds +20% on cost and/or time
    """
    if cost_flag is None and time_flag is None:
        return "reporting"
    cf = cost_flag or 0
    tf = time_flag or 0
    cp = cost_pct or 0
    tp = time_pct or 0
    if cf == 0 and tf == 0:
        return "on-track"
    if max(cp, tp) <= 20:
        return "minor-overrun"
    return "at-risk"


def main():
    csv_path = sys.argv[1]
    out_path = sys.argv[2]

    latest_by_project = {}

    with open(csv_path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            code = row["project_code"]
            li = int(row["landmark_index"] or 0)
            prev = latest_by_project.get(code)
            if prev is None or li > int(prev["landmark_index"] or 0):
                latest_by_project[code] = row

    projects = []
    for code, row in latest_by_project.items():
        state_raw = (row["state"] or "").strip().upper()
        if state_raw == "":
            state = "MISCELLANEOUS / UNSPECIFIED"
        else:
            state = state_raw

        original_cost = f(row["original_cost_rs_cr"]) or 0
        anticipated_cost = f(row["anticipated_cost_rs_cr"])
        cost = anticipated_cost if anticipated_cost is not None else original_cost
        expenditure = f(row["cumulative_expenditure_rs_cr"]) or 0

        progress = None
        if cost:
            progress = max(0, min(100, round((expenditure / cost) * 100, 1)))

        status = classify_status(
            f(row["final_cost_overrun_flag"]),
            f(row["final_time_overrun_flag"]),
            f(row["final_cost_overrun_pct"]),
            f(row["final_time_overrun_pct"]),
        )

        projects.append({
            "code": code,
            "name": (row["project_name"] or "").strip(),
            "sector": (row["sector"] or "").strip().upper(),
            "state": state,
            "agency": (row["agency_name"] or "").strip(),
            "originalCost": round(original_cost, 2),
            "cost": round(cost, 2) if cost is not None else 0,
            "expenditure": round(expenditure, 2),
            "progress": progress,
            "status": status,
            "costOverrunPct": f(row["final_cost_overrun_pct"]),
            "timeOverrunPct": f(row["final_time_overrun_pct"]),
            "isMega": (cost or 0) >= 20000,
        })

    # ---- state-wise aggregates (ALL projects, every state/UT + Multi State + Misc) ----
    state_agg = defaultdict(lambda: {"count": 0, "originalCost": 0.0, "expenditure": 0.0})
    by_state = defaultdict(list)
    for p in projects:
        agg = state_agg[p["state"]]
        agg["count"] += 1
        agg["originalCost"] += p["originalCost"]
        agg["expenditure"] += p["expenditure"]
        by_state[p["state"]].append(p)

    states = [
        {
            "state": s,
            "count": agg["count"],
            "originalCost": round(agg["originalCost"], 2),
            "expenditure": round(agg["expenditure"], 2),
        }
        for s, agg in state_agg.items()
    ]
    states.sort(key=lambda s: s["count"], reverse=True)

    # Keep a generous top-N per state for the expandable "top projects" chips,
    # sorted by current cost so the biggest projects in each state show first.
    by_state_top = {
        s: sorted(plist, key=lambda p: p["cost"], reverse=True)[:12]
        for s, plist in by_state.items()
    }

    # ---- sector / ministry breakdowns (ALL projects) ----
    sector_counts = defaultdict(int)
    for p in projects:
        sector_counts[p["sector"]] += 1
    sector_breakdown = [
        {"sector": s, "label": title_case(s), "count": c}
        for s, c in sector_counts.items()
    ]
    sector_breakdown.sort(key=lambda x: x["count"], reverse=True)

    # ---- high value projects (top by current/anticipated cost) ----
    high_value = sorted(projects, key=lambda p: p["cost"], reverse=True)[:50]

    real_states_count = sum(
        1 for s in states if s["state"] not in ("MULTI STATE", "MISCELLANEOUS / UNSPECIFIED")
    )

    output = {
        "generatedFrom": "paiman_projects_landmark_dataset (latest landmark per project)",
        "totals": {
            "projects": len(projects),
            "originalCost": round(sum(p["originalCost"] for p in projects), 2),
            "expenditure": round(sum(p["expenditure"] for p in projects), 2),
            "states": real_states_count,
        },
        "states": states,
        "byState": by_state_top,
        "sectorBreakdown": sector_breakdown,
        "highValue": high_value,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, separators=(",", ":"))

    print(f"Projects: {len(projects)}")
    print(f"States/UTs (excl. multi-state/misc): {real_states_count}")
    print(f"Total state buckets (incl. Multi State & Misc): {len(states)}")
    print(f"Sectors: {len(sector_breakdown)}")
    print(f"High-value entries: {len(high_value)}")


if __name__ == "__main__":
    main()
