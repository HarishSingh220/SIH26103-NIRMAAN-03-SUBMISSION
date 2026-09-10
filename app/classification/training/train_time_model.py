"""
Training Pipeline for Time Overrun Model (V3)
=============================================
V3 Improvements:
1. Group-aware CV (project_code) instead of StratifiedKFold
2. Per-fold SMOTE (no data leakage)
3. F2 + AUC composite metric for Optuna tuning (recall-focused)

Usage:
    python -m app.classification.training.train_time_model
"""

import pandas as pd
import numpy as np
import warnings
import os
import argparse
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import optuna
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5
MAX_LANDMARK_INDEX = 3
BETA = 2
TARGET = 'final_time_overrun_flag'
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "paiman_projects_landmark_dataset.csv"
CSV_PATH = os.environ.get("CLASSIFICATION_DATA_PATH", str(DEFAULT_DATA_PATH))
SMOTE_ENABLED = True
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "models" / "classification" / "paiman_time_overrun_ensemble.joblib"
OUTPUT_PATH = os.environ.get("CLASSIFICATION_TIME_MODEL_PATH", str(DEFAULT_OUTPUT_PATH))

def resolve_path(value: str, default: Path) -> str:
    path = Path(value) if value else default
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


CSV_PATH = resolve_path(CSV_PATH, DEFAULT_DATA_PATH)
OUTPUT_PATH = resolve_path(OUTPUT_PATH, DEFAULT_OUTPUT_PATH)


ENSEMBLE_MODELS = {
    'xgb': {'name': 'XGBoost', 'use': True, 'weight': 0.50, 'optuna_trials': 20},
    'hgb': {'name': 'HistGradientBoosting', 'use': True, 'weight': 0.50, 'optuna_trials': 20},
}

LEAKAGE_COLS = {
    'final_cost_overrun_pct', 'final_time_overrun_pct',
    'contemporaneous_cost_overrun_pct', 'contemporaneous_time_overrun_pct',
    'final_cost_overrun_flag', 'project_name', 'reporting_period_date',
    'anticipated_commissioning_date',
}

INDICATOR_COLS = ['planned_duration_months', 'project_age_at_report_months']
TRAJECTORY_COLS = [
    'expenditure_to_cost_pct', 'progress_ratio',
    'anticipated_cost_change_pct', 'anticipated_date_change_months'
]

def main() -> None:
    global CSV_PATH, OUTPUT_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=CSV_PATH,
        help=f"Training dataset path (default: {CSV_PATH})",
    )
    parser.add_argument(
        "--output-path",
        default=OUTPUT_PATH,
        help=f"Model artifact path (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()
    CSV_PATH = resolve_path(args.data_path, DEFAULT_DATA_PATH)
    OUTPUT_PATH = resolve_path(args.output_path, DEFAULT_OUTPUT_PATH)

    # =============================================================================
    # DATA LOADING
    # =============================================================================
    print("=" * 70)
    print(f"TRAINING: {TARGET} (V3 - Group CV + Per-fold SMOTE + F2+AUC)")
    print("=" * 70)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"Training dataset not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    required_columns = {"project_code", "landmark_index", TARGET}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Training dataset is missing required columns: {missing_columns}")
    df_early = df[df['landmark_index'] <= MAX_LANDMARK_INDEX].copy()

    cols_to_drop = [c for c in LEAKAGE_COLS if c in df_early.columns]
    df_clean = df_early.drop(columns=cols_to_drop, errors='ignore')
    df_clean = df_clean.dropna(subset=[TARGET])

    print(f"  Rows: {len(df_clean)}")
    print(f"  Class 0: {(df_clean[TARGET]==0).sum()}, Class 1: {(df_clean[TARGET]==1).sum()}")

    # Missingness indicators
    for col in INDICATOR_COLS:
        if col in df_clean.columns:
            df_clean[f"{col}_was_missing"] = df_clean[col].isna().astype(int)

    # Sort and project stats
    df_clean = df_clean.sort_values(['project_code', 'landmark_index']).reset_index(drop=True)
    group_sizes = df_clean.groupby('project_code').size()
    df_clean['rows_per_project'] = df_clean['project_code'].map(group_sizes)

    # Trajectory features
    single_row_projects = df_clean['rows_per_project'] == 1
    single_row_project_codes = df_clean.loc[single_row_projects, 'project_code'].unique()

    for col in TRAJECTORY_COLS:
        if col not in df_clean.columns:
            continue

        def first_valid(s): return s.dropna().iloc[0] if len(s.dropna()) > 0 else np.nan
        def last_valid(s): return s.dropna().iloc[-1] if len(s.dropna()) > 0 else np.nan

        extremes = df_clean.groupby('project_code')[col].agg([('first', first_valid), ('last', last_valid)]).reset_index()
        extremes[f'{col}_delta'] = extremes['last'] - extremes['first']
        extremes.loc[extremes['project_code'].isin(single_row_project_codes), f'{col}_delta'] = 0.0

        df_clean = df_clean.merge(extremes[['project_code', f'{col}_delta']], on='project_code', how='left')
        df_clean[f'{col}_delta_missing'] = df_clean[f'{col}_delta'].isna().astype(int)
        df_clean.loc[single_row_projects, f'{col}_delta_missing'] = 0

    # =============================================================================
    # TRAIN / TEST SPLIT (Group-aware)
    # =============================================================================
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    feature_cols = [c for c in df_clean.columns if c not in ['project_code', TARGET]]
    X = df_clean[feature_cols].copy()
    y = df_clean[TARGET].values
    groups = df_clean['project_code'].values

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    project_codes_test = df_clean['project_code'].values[test_idx]

    print(f"\n  Train: {len(X_train)} rows, {len(np.unique(groups_train))} projects")
    print(f"  Test: {len(X_test)} rows")
    print(f"  Features: {len(feature_cols)}")

    # =============================================================================
    # PREPROCESS
    # =============================================================================
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    X_train_proc, X_test_proc = X_train.copy(), X_test.copy()
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        train_vals = X_train[col].fillna('MISSING').astype(str)
        le.fit(train_vals)
        X_train_proc[col] = le.transform(train_vals)
        test_vals = X_test[col].fillna('MISSING').astype(str)
        unseen = ~test_vals.isin(le.classes_)
        if unseen.sum() > 0:
            test_vals = test_vals.replace({v: train_vals.value_counts().index[0] for v in test_vals[unseen].unique()})
        X_test_proc[col] = le.transform(test_vals)
        label_encoders[col] = le

    categorical_modes = {
        col: str(X_train[col].fillna("MISSING").astype(str).mode().iloc[0])
        for col in categorical_cols
    }

    numeric_cols = X_train_proc.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_original = numeric_cols.copy()
    if 'rows_per_project' in numeric_cols:
        numeric_cols.remove('rows_per_project')

    # Clip and impute
    if 'anticipated_cost_change_pct' in X_train_proc.columns:
        for df_ in [X_train_proc, X_test_proc]:
            df_['anticipated_cost_change_pct'] = df_['anticipated_cost_change_pct'].replace([np.inf, -np.inf], np.nan)
        p1, p99 = X_train_proc['anticipated_cost_change_pct'].quantile([0.01, 0.99])
        X_train_proc['anticipated_cost_change_pct'] = X_train_proc['anticipated_cost_change_pct'].clip(p1, p99)
        X_test_proc['anticipated_cost_change_pct'] = X_test_proc['anticipated_cost_change_pct'].clip(p1, p99)
        med = X_train_proc['anticipated_cost_change_pct'].median()
        X_train_proc['anticipated_cost_change_pct'] = X_train_proc['anticipated_cost_change_pct'].fillna(med)
        X_test_proc['anticipated_cost_change_pct'] = X_test_proc['anticipated_cost_change_pct'].fillna(med)

    for col in numeric_cols:
        if col == 'anticipated_cost_change_pct':
            continue
        for df_ in [X_train_proc, X_test_proc]:
            df_[col] = df_[col].replace([np.inf, -np.inf], np.nan)
        med = X_train_proc[col].median()
        X_train_proc[col] = X_train_proc[col].fillna(med)
        X_test_proc[col] = X_test_proc[col].fillna(med)

    # =============================================================================
    # OPTUNA TUNING - Group-aware CV with per-fold SMOTE and F2+AUC metric
    # =============================================================================
    print("\nTuning models (Group CV + Per-fold SMOTE + F2+AUC)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _eval_with_smote(model_class, params, X, y, groups, n_splits=3, use_smote=True):
        """
        Group-aware CV with per-fold SMOTE.
        Returns composite score: 0.5 * F2 + 0.5 * AUC
        """
        gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []

        for tr_idx, va_idx in gkf.split(X, y, groups=groups):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # Per-fold SMOTE (only on training data)
            if use_smote:
                min_class = min((y_tr == 0).sum(), (y_tr == 1).sum())
                k = max(1, min(5, min_class - 1))
                smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

            # Train
            model = model_class(**params)
            if isinstance(model, xgb.XGBClassifier):
                model.fit(X_tr, y_tr, verbose=False)
            else:
                model.fit(X_tr, y_tr)

            # Predict
            probs = model.predict_proba(X_va)[:, 1]

            # Compute metrics
            auc = roc_auc_score(y_va, probs)

            # F2 at best threshold
            best_f2 = -1
            for t in np.arange(0.1, 0.9, 0.05):
                preds = (probs >= t).astype(int)
                f2 = fbeta_score(y_va, preds, beta=BETA, zero_division=0)
                best_f2 = max(best_f2, f2)

            # Composite: 50% F2 + 50% AUC
            fold_scores.append(0.5 * best_f2 + 0.5 * auc)

        return np.mean(fold_scores)


    def xgb_objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbosity': 0,
        }
        return _eval_with_smote(xgb.XGBClassifier, params, X_train_proc, y_train, groups_train, n_splits=3, use_smote=SMOTE_ENABLED)


    def hgb_objective(trial):
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 60),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 40),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 3.0),
            'random_state': RANDOM_STATE,
        }
        return _eval_with_smote(HistGradientBoostingClassifier, params, X_train_proc, y_train, groups_train, n_splits=3, use_smote=SMOTE_ENABLED)


    best_params = {}
    for kind, cfg in ENSEMBLE_MODELS.items():
        print(f"  [{cfg['name']}] Tuning...")
        study = optuna.create_study(direction='maximize')

        if kind == 'xgb':
            study.optimize(xgb_objective, n_trials=cfg['optuna_trials'], show_progress_bar=False)
        else:
            study.optimize(hgb_objective, n_trials=cfg['optuna_trials'], show_progress_bar=False)

        best_params[kind] = study.best_params
        print(f"    Best composite (F2+AUC): {study.best_value:.4f}")

    # =============================================================================
    # 5-FOLD OOF - Group-aware with per-fold SMOTE
    # =============================================================================
    print("\n5-Fold OOF (Group CV + Per-fold SMOTE)...")
    gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = {k: np.zeros(len(X_train_proc)) for k in ENSEMBLE_MODELS}

    for fold, (tr, va) in enumerate(gkf.split(X_train_proc, y_train, groups=groups_train), 1):
        X_tr, X_va = X_train_proc.iloc[tr], X_train_proc.iloc[va]
        y_tr, y_va = y_train[tr], y_train[va]

        # Per-fold SMOTE
        if SMOTE_ENABLED:
            min_class = min((y_tr == 0).sum(), (y_tr == 1).sum())
            k = max(1, min(5, min_class - 1))
            smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
            X_tr_smote, y_tr_smote = smote.fit_resample(X_tr, y_tr)
        else:
            X_tr_smote, y_tr_smote = X_tr, y_tr

        for kind in ENSEMBLE_MODELS:
            params = best_params[kind].copy()
            if kind == 'xgb':
                params.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0})
            else:
                params.update({'random_state': RANDOM_STATE})

            model = xgb.XGBClassifier(**params) if kind == 'xgb' else HistGradientBoostingClassifier(**params)
            if kind == 'xgb':
                model.fit(X_tr_smote, y_tr_smote, verbose=False)
            else:
                model.fit(X_tr_smote, y_tr_smote)

            oof_probs[kind][va] = model.predict_proba(X_va)[:, 1]

    ensemble_oof = sum(cfg['weight'] * oof_probs[k] for k, cfg in ENSEMBLE_MODELS.items())
    ensemble_auc = roc_auc_score(y_train, ensemble_oof)
    ensemble_pr = average_precision_score(y_train, ensemble_oof)
    print(f"  OOF AUC: {ensemble_auc:.4f}, PR-AUC: {ensemble_pr:.4f}")

    # =============================================================================
    # THRESHOLD SWEEP (on real OOF predictions, no SMOTE leakage)
    # =============================================================================
    print("\nFinding optimal threshold on real OOF predictions...")
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_f2, best_threshold = -1, 0.5

    for t in thresholds:
        preds = (ensemble_oof >= t).astype(int)
        f2 = fbeta_score(y_train, preds, beta=BETA, zero_division=0)
        if f2 > best_f2:
            best_f2, best_threshold = f2, t

    print(f"  Best F2 threshold: {best_threshold:.3f} (F2={best_f2:.4f})")
    final_threshold = best_threshold

    # =============================================================================
    # RETRAIN ON FULL TRAINING DATA (with SMOTE)
    # =============================================================================
    print("\nRetraining on full training data with SMOTE...")
    if SMOTE_ENABLED:
        min_class = min((y_train == 0).sum(), (y_train == 1).sum())
        k = max(1, min(5, min_class - 1))
        smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
        X_smote, y_smote = smote.fit_resample(X_train_proc, y_train)
    else:
        X_smote, y_smote = X_train_proc, y_train

    print(f"  Training samples: {len(X_smote)} (real: {len(X_train_proc)})")

    final_models = {}
    for kind, cfg in ENSEMBLE_MODELS.items():
        params = best_params[kind].copy()
        if kind == 'xgb':
            params.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': 0})
            model = xgb.XGBClassifier(**params)
            model.fit(X_smote, y_smote, verbose=False)
        else:
            params.update({'random_state': RANDOM_STATE})
            model = HistGradientBoostingClassifier(**params)
            model.fit(X_smote, y_smote)
        final_models[kind] = model
        print(f"  [{cfg['name']}] trained")

    # Save
    artifact = {
        'models': final_models,
        'label_encoders': label_encoders,
        'categorical_modes': categorical_modes,
        'clip_bounds': {'anticipated_cost_change_pct': {
            'p1': float(X_train_proc['anticipated_cost_change_pct'].quantile(0.01)),
            'p99': float(X_train_proc['anticipated_cost_change_pct'].quantile(0.99)),
            'median': float(X_train_proc['anticipated_cost_change_pct'].median()),
        }},
        'numeric_medians': {col: float(X_train_proc[col].median()) for col in numeric_cols_original},
        'feature_columns': feature_cols,
        'threshold': float(final_threshold),
        'ensemble_weights': {k: float(v['weight']) for k, v in ENSEMBLE_MODELS.items()},
        'target': TARGET,
        'max_landmark_index': MAX_LANDMARK_INDEX,
        'training_timestamp': datetime.now(timezone.utc).isoformat(),
        'smote_applied': SMOTE_ENABLED,
        'version': 'V3',
        'train_metrics': {
            'oof_auc': float(ensemble_auc),
            'oof_pr_auc': float(ensemble_pr),
            'best_f2_threshold': float(best_threshold),
            'final_threshold': float(final_threshold),
            'n_train_rows': len(X_train_proc),
            'n_features': len(feature_cols),
        },
    }

    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    joblib.dump(artifact, OUTPUT_PATH)
    print(f"\nSaved to: {OUTPUT_PATH}")

    # =============================================================================
    # EVALUATE
    # =============================================================================
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)

    test_probs = sum(cfg['weight'] * final_models[k].predict_proba(X_test_proc)[:, 1] for k, cfg in ENSEMBLE_MODELS.items())
    test_preds = (test_probs >= final_threshold).astype(int)

    print(f"  ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")
    print(f"  PR-AUC: {average_precision_score(y_test, test_probs):.4f}")

    cm = confusion_matrix(y_test, test_preds)
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"              No Overrun | Overrun")
    print(f"  Actual No Overrun:  {cm[0,0]:5d}  |  {cm[0,1]:5d}")
    print(f"  Actual Overrun:     {cm[1,0]:5d}  |  {cm[1,1]:5d}")

    tp, fp, fn = cm[1,1], cm[0,1], cm[1,0]
    print(f"\n  Precision: {tp/(tp+fp):.4f}")
    print(f"  Recall: {tp/(tp+fn):.4f}")
    print(f"  F2: {fbeta_score(y_test, test_preds, beta=2, zero_division=0):.4f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - V3")
    print("=" * 70)


if __name__ == "__main__":
    main()
