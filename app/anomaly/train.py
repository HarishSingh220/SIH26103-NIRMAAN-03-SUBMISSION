"""Offline artifact builder.

Run this during deployment/model refresh, never per API request:
    python -m app.anomaly.train
(equivalent to:
    python -m app.anomaly.train --input data/paiman_projects_for_anomaly_detection.csv \
                         --output models/anomaly_model.joblib)
"""
from __future__ import annotations
import argparse
import os
import joblib
import pandas as pd
from .core import fit_hybrid_reference, _fit_track_b_peer_reference, validate_input, coerce_dtypes
from . import config as cfg

DEFAULT_INPUT = str(cfg.DATA_PATH)
DEFAULT_OUTPUT = str(cfg.MODEL_PATH)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help=f"Training CSV (default: {DEFAULT_INPUT})")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help=f"Artifact path to write (default: {DEFAULT_OUTPUT})")
    p.add_argument("--cutoff", default=cfg.FY_HOLDOUT_START)
    args = p.parse_args()
    df = pd.read_csv(args.input)
    validate_input(df)
    df = coerce_dtypes(df)
    hybrid = fit_hybrid_reference(
        df, fit_cutoff=args.cutoff,
        feature_families=["base", "peer", "trajectory"],
        fit_isolation_forest=True,
    )
    track_b = _fit_track_b_peer_reference(df, cutoff=args.cutoff)
    artifact = {
        "model_version": cfg.MODEL_VERSION,
        "fit_cutoff": args.cutoff,
        "hybrid_reference": hybrid,
        "track_b_reference": track_b,
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    print(f"Saved model artifact: {args.output}")


if __name__ == "__main__":
    main()
