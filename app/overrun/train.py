"""
Train (or retrain) the cost-overrun and/or time-overrun combined pipelines and save them
as plain-pickle `.pkl` files, ready for `main.py` to serve.

Usage:
    python -m app.overrun.train --target cost
    python -m app.overrun.train --target time
    python -m app.overrun.train --target both                 # default
    python -m app.overrun.train --target both --data-path /path/to/data.csv
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import pandas as pd

from . import config as cfg
from . import core


def log(msg):
    print(msg, flush=True)


def load_and_prepare(data_path: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise KeyError(f"'{target_col}' not found in {data_path}")

    df = core.basic_cleaning(df)
    df = core.create_reporting_date(df)
    df = core.create_time_indices(df)
    df = core.map_project_target(df, target_col)
    df = core.prepare_regression_target(df, target_col)
    df = core.create_core_features(df)
    df = core.add_missingness_flags(df)
    df = core.create_temporal_features(df)
    df = core.sanitize_numeric_data(df)
    return df


def build_feature_lists(df: pd.DataFrame, sector_name: str):
    base_numerical = [
        "original_cost_rs_cr", "cumulative_expenditure_rs_cr", "expenditure_to_cost_pct",
        "project_age_at_report_months", "planned_duration_months", "progress_ratio",
        "landmark_index", "n_landmarks_total", "horizon_months",
        "landmark_completion_ratio", "remaining_landmarks", "expenditure_per_landmark_rs_cr",
        "cost_per_landmark_planned_rs_cr", "expenditure_progress_gap", "horizon_to_planned_ratio",
        "fy_start_year", "quarter_num", "expenditure_velocity", "expenditure_acceleration",
        "progress_velocity", "progress_acceleration", "rolling_expenditure_mean",
        "rolling_expenditure_std", "rolling_progress_mean", "rolling_progress_std",
    ] + [c for c in df.columns if c.endswith("_was_missing")]
    base_numerical = [c for c in base_numerical if c in df.columns]
    categorical = [c for c in cfg.CATEGORICAL_FEATURES if c in df.columns]

    final_features = base_numerical + categorical

    sector_cols = core.sector_interaction_columns(sector_name)
    model_b_numerical = list(dict.fromkeys(base_numerical + sector_cols))
    model_b_features = model_b_numerical + categorical

    return {
        "final_numerical": base_numerical, "final_features": final_features,
        "model_b_numerical": model_b_numerical, "model_b_features": model_b_features,
        "categorical": categorical,
    }


def check_leakage(feature_cols, target_col):
    leakage_cols = (
        [target_col, cfg.LOG_TARGET, cfg.PROJECT_COL, "project_name",
         cfg.GLOBAL_TIME_COL, cfg.PROJECT_TIME_COL, cfg.DATE_COL, "financial_year", "quarter"]
        + cfg.LEAKAGE_COLUMNS_BY_TARGET[target_col]
    )
    leaked = [c for c in leakage_cols if c in feature_cols]
    if leaked:
        raise ValueError(f"Leakage columns found in feature set: {leaked}")


def train_target(target_col: str, data_path: str, output_dir: str):
    t0 = time.time()
    log(f"\n{'=' * 70}\nTraining target: {target_col}\n{'=' * 70}")

    df = load_and_prepare(data_path, target_col)
    log(f"Rows after cleaning/target-scoping: {len(df)}  (projects: {df[cfg.PROJECT_COL].nunique()})")

    train_df, val_df, test_df = core.chronological_split(df)
    log(f"Split -> train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    sector_name = core.detect_extreme_sector(train_df, target_col)
    log(f"Auto-detected Model B sector (TRAIN >{cfg.EXTREME_THRESHOLD}% tail): {sector_name}")

    # Sector interaction features must be added to each already-split part separately
    # (not before the split), using the TRAIN-detected sector consistently across all three.
    train_df = core.create_sector_interaction_features(train_df, sector_name)
    val_df = core.create_sector_interaction_features(val_df, sector_name)
    test_df = core.create_sector_interaction_features(test_df, sector_name)

    feats = build_feature_lists(train_df, sector_name)
    check_leakage(feats["final_features"], target_col)
    check_leakage(feats["model_b_features"], target_col)
    log(f"Final Model features: {len(feats['final_features'])}  |  Model B features: {len(feats['model_b_features'])}")

    log("\n--- Training Final Model ---")
    final_model_artifacts = core.train_and_package_model(
        target_col, feats["final_features"], feats["final_numerical"], feats["categorical"],
        train_df, val_df, test_df, log=log,
    )

    log("\n--- Training Model B ---")
    model_b_artifacts = core.train_and_package_model(
        target_col, feats["model_b_features"], feats["model_b_numerical"], feats["categorical"],
        train_df, val_df, test_df, log=log,
    )

    pipeline = core.CombinedOverrunPipeline(
        target_col=target_col,
        model_b=model_b_artifacts,
        final_model=final_model_artifacts,
        sector_name=sector_name,
    )

    os.makedirs(output_dir, exist_ok=True)
    model_filename = (
        "final_cost_overrun_pct_combined_model.pkl"
        if target_col == cfg.COST_TARGET
        else "final_time_overrun_pct_combined_model.pkl"
    )
    schema_filename = (
        "cost_overrun_feature_schema.json"
        if target_col == cfg.COST_TARGET
        else "time_overrun_feature_schema.json"
    )
    output_root = os.path.abspath(output_dir)
    model_path = os.path.join(output_root, model_filename)
    schema_path = os.path.join(output_root, schema_filename)

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    log(f"Saved {model_path} ({os.path.getsize(model_path) / 1e6:.2f} MB)")

    schema = {
        "target": target_col,
        "target_scope": "resolved, strictly positive overruns only",
        "required_raw_input_columns": pipeline.required_raw_cols,
        "models": {
            "model_b": {
                "description": f"XGBoost+LightGBM+CatBoost SLSQP ensemble with a '{sector_name}' sector interaction block.",
                "sector": sector_name,
                "numerical_features": model_b_artifacts["numerical_features"],
                "categorical_features": model_b_artifacts["categorical_features"],
                "ensemble_weights": {"xgboost": float(model_b_artifacts["weights"][0]),
                                      "lightgbm": float(model_b_artifacts["weights"][1]),
                                      "catboost": float(model_b_artifacts["weights"][2])},
                "test_metrics": model_b_artifacts["test_metrics"],
                "output_field": f"predicted_{target_col}_model_b",
            },
            "final_model": {
                "description": "XGBoost+LightGBM+CatBoost SLSQP ensemble on the full feature set, no interaction features.",
                "numerical_features": final_model_artifacts["numerical_features"],
                "categorical_features": final_model_artifacts["categorical_features"],
                "ensemble_weights": {"xgboost": float(final_model_artifacts["weights"][0]),
                                      "lightgbm": float(final_model_artifacts["weights"][1]),
                                      "catboost": float(final_model_artifacts["weights"][2])},
                "test_metrics": final_model_artifacts["test_metrics"],
                "output_field": f"predicted_{target_col}_final_model",
            },
        },
    }
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    log(f"Saved {schema_path}")
    log(f"Done in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["cost", "time", "both"], default="both")
    parser.add_argument("--data-path", default=cfg.RAW_DATA_PATH)
    parser.add_argument("--output-dir", default=cfg.MODEL_DIR)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        log(f"ERROR: data file not found at {args.data_path}")
        log("Set --data-path or the RAW_DATA_PATH environment variable.")
        sys.exit(1)

    targets = []
    if args.target in ("cost", "both"):
        targets.append(cfg.COST_TARGET)
    if args.target in ("time", "both"):
        targets.append(cfg.TIME_TARGET)

    for target_col in targets:
        train_target(target_col, args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
