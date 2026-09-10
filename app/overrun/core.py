"""
Shared feature engineering, training helpers, and the deployable prediction pipeline.

This is the one place these things are defined. `train.py` imports it to build and pickle
the fitted pipelines; `main.py` imports it (for the class definition) to unpickle and serve
them. Because both sides import the SAME module instead of relying on a notebook's
in-memory `__main__` namespace, plain `pickle` is safe here — no cloudpickle "serialize
code by value" fragility, and no risk of the corruption that showed up when a
notebook-cloudpickle artifact got mangled in transit.

One pipeline class (`CombinedOverrunPipeline`) is used for both cost and time overrun —
they are structurally identical (same feature engineering, same Model B / Final Model
ensembling), differing only in target column and which sector Model B targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.optimize import minimize

from . import config as cfg


# =============================================================================
# Feature engineering (shared by both targets)
# =============================================================================
def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    a = numerator.to_numpy(dtype=float)
    b = denominator.to_numpy(dtype=float)
    result = np.full(len(a), np.nan, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b) & (b != 0)
    result[valid] = a[valid] / b[valid]
    return result


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    df = df.loc[:, ~df.columns.duplicated()]

    df[cfg.PROJECT_COL] = df[cfg.PROJECT_COL].astype(str).str.strip()
    df.loc[df[cfg.PROJECT_COL].isin(["", "nan", "none", "null"]), cfg.PROJECT_COL] = np.nan

    for c in cfg.RAW_TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].str.lower().isin(["", "nan", "none", "null"]), c] = np.nan
    return df


def create_reporting_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL], errors="coerce")
    return df


def create_time_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    unique_dates = df[cfg.DATE_COL].dropna().drop_duplicates().sort_values().tolist()
    date_to_index = {date: i for i, date in enumerate(unique_dates)}
    df[cfg.GLOBAL_TIME_COL] = df[cfg.DATE_COL].map(date_to_index)

    df = df.sort_values([cfg.PROJECT_COL, cfg.DATE_COL]).reset_index(drop=True)
    df[cfg.PROJECT_TIME_COL] = df.groupby(cfg.PROJECT_COL).cumcount()
    return df


def create_core_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in cfg.RAW_NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["expenditure_to_cost_pct"] = safe_divide(df["cumulative_expenditure_rs_cr"], df["original_cost_rs_cr"]) * 100
    df["landmark_completion_ratio"] = safe_divide(df["landmark_index"], df["n_landmarks_total"])
    df["remaining_landmarks"] = df["n_landmarks_total"] - df["landmark_index"]
    df["expenditure_per_landmark_rs_cr"] = safe_divide(df["cumulative_expenditure_rs_cr"], df["landmark_index"])
    df["cost_per_landmark_planned_rs_cr"] = safe_divide(df["original_cost_rs_cr"], df["n_landmarks_total"])
    df["expenditure_progress_gap"] = df["expenditure_to_cost_pct"] - (df["progress_ratio"] * 100)
    df["horizon_to_planned_ratio"] = safe_divide(df["horizon_months"], df["planned_duration_months"])
    df["fy_start_year"] = pd.to_numeric(df["financial_year"].astype(str).str.slice(0, 4), errors="coerce")
    df["quarter_num"] = df["quarter"].astype(str).str.extract(r"(\d)").astype(float)
    return df


def add_missingness_flags(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    df = df.copy()
    for c in (cols or cfg.MISSINGNESS_SOURCE_COLS):
        if c in df.columns:
            df[f"{c}_was_missing"] = df[c].isna().astype(int)
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([cfg.PROJECT_COL, cfg.DATE_COL])

    g_exp = df.groupby(cfg.PROJECT_COL)["cumulative_expenditure_rs_cr"]
    df["expenditure_velocity"] = g_exp.diff()
    df["expenditure_acceleration"] = df.groupby(cfg.PROJECT_COL)["expenditure_velocity"].diff()

    g_prog = df.groupby(cfg.PROJECT_COL)["landmark_completion_ratio"]
    df["progress_velocity"] = g_prog.diff()
    df["progress_acceleration"] = df.groupby(cfg.PROJECT_COL)["progress_velocity"].diff()

    df["rolling_expenditure_mean"] = g_exp.transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    df["rolling_expenditure_std"] = g_exp.transform(lambda x: x.shift(1).rolling(window=3, min_periods=2).std())
    df["rolling_progress_mean"] = g_prog.transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    df["rolling_progress_std"] = g_prog.transform(lambda x: x.shift(1).rolling(window=3, min_periods=2).std())
    return df


def sanitize_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df


def sector_key_for(sector_name: str) -> str:
    return (
        str(sector_name).strip().lower()
        .replace("&", "and").replace(" ", "_").replace("-", "_")
    )


def create_sector_interaction_features(df: pd.DataFrame, sector_name: str) -> pd.DataFrame:
    """Adds a sector flag + 4 derived interaction columns for the given sector name."""
    df = df.copy()
    key = sector_key_for(sector_name)

    sector_values = df["sector"].astype(str).str.strip().str.upper()
    flag_col = f"{key}_flag"
    df[flag_col] = (sector_values == str(sector_name).strip().upper()).astype(int)

    interaction_map = {
        f"{key}_expenditure_gap": "expenditure_progress_gap",
        f"{key}_maturity": "progress_ratio",
        f"{key}_expenditure_velocity": "expenditure_velocity",
        f"{key}_progress_velocity": "progress_velocity",
    }
    for new_col, source_col in interaction_map.items():
        if source_col in df.columns:
            df[new_col] = df[flag_col] * pd.to_numeric(df[source_col], errors="coerce")
        else:
            df[new_col] = np.nan
    return df


def sector_interaction_columns(sector_name: str) -> list[str]:
    key = sector_key_for(sector_name)
    return [f"{key}_flag", f"{key}_expenditure_gap", f"{key}_maturity",
            f"{key}_expenditure_velocity", f"{key}_progress_velocity"]


# =============================================================================
# Target handling (shared logic, target column passed in)
# =============================================================================
def map_project_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    known = df.dropna(subset=[cfg.PROJECT_COL, target_col])
    target_map = known.groupby(cfg.PROJECT_COL)[target_col].median().to_dict()
    df[target_col] = df[target_col].fillna(df[cfg.PROJECT_COL].map(target_map))
    return df


def prepare_regression_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).copy()
    df = df[df[target_col] > 0].copy()
    df[cfg.LOG_TARGET] = np.log1p(df[target_col])
    return df


def chronological_split(df: pd.DataFrame, train_ratio=cfg.TRAIN_RATIO, val_ratio=cfg.VAL_RATIO):
    df = df.sort_values(cfg.GLOBAL_TIME_COL).reset_index(drop=True)
    unique_times = np.sort(df[cfg.GLOBAL_TIME_COL].dropna().unique())
    n_times = len(unique_times)

    train_end = int(n_times * train_ratio)
    val_end = int(n_times * (train_ratio + val_ratio))

    train_times = unique_times[:train_end]
    val_times = unique_times[train_end:val_end]
    test_times = unique_times[val_end:]

    train_df = df[df[cfg.GLOBAL_TIME_COL].isin(train_times)].copy()
    val_df = df[df[cfg.GLOBAL_TIME_COL].isin(val_times)].copy()
    test_df = df[df[cfg.GLOBAL_TIME_COL].isin(test_times)].copy()
    return train_df, val_df, test_df


def detect_extreme_sector(train_df: pd.DataFrame, target_col: str, threshold: int = cfg.EXTREME_THRESHOLD) -> str:
    """The sector making up the largest share of the >threshold% tail, from TRAIN only."""
    extreme = train_df[pd.to_numeric(train_df[target_col], errors="coerce") > threshold]
    if extreme.empty:
        raise ValueError(f"No TRAIN observations above {threshold}% - cannot auto-detect a sector.")
    return extreme["sector"].fillna("UNKNOWN").value_counts().index[0]


# =============================================================================
# Modeling
# =============================================================================
def build_preprocessor(numerical_features, categorical_features) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numerical_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_features),
        ],
        remainder="drop",
    )


def prepare_catboost_data(df, feature_cols, categorical_features):
    X = df[feature_cols].copy()
    for col in categorical_features:
        X[col] = X[col].fillna("UNKNOWN").astype(str)
    return X


def inverse_log_prediction(pred):
    return np.expm1(np.clip(np.asarray(pred), -20, 20))


def evaluate_original_scale(y_true, prediction):
    return {
        "MAE": mean_absolute_error(y_true, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_true, prediction)),
        "R2": r2_score(y_true, prediction),
    }


def create_xgb(random_state=cfg.RANDOM_STATE):
    return XGBRegressor(random_state=random_state, **cfg.XGB_PARAMS)


def create_lgbm(random_state=cfg.RANDOM_STATE):
    return LGBMRegressor(random_state=random_state, **cfg.LGBM_PARAMS)


def create_catboost(random_state=cfg.RANDOM_STATE):
    return CatBoostRegressor(random_seed=random_state, **cfg.CATBOOST_PARAMS)


def create_expanding_folds(train_df, n_folds=cfg.N_OOF_FOLDS, initial_fraction=0.50):
    unique_times = np.sort(train_df[cfg.GLOBAL_TIME_COL].unique())
    n_times = len(unique_times)
    initial_size = max(2, int(n_times * initial_fraction))
    remaining = n_times - initial_size
    fold_size = max(1, remaining // n_folds)

    folds = []
    for i in range(n_folds):
        train_end = initial_size + i * fold_size
        val_end = min(initial_size + (i + 1) * fold_size, n_times)
        if train_end >= val_end:
            break
        folds.append((unique_times[:train_end], unique_times[train_end:val_end]))
    return folds


def generate_oof_predictions(train_df, feature_cols, numerical_features, categorical_features, n_folds=cfg.N_OOF_FOLDS):
    folds = create_expanding_folds(train_df, n_folds=n_folds)
    oof_xgb = np.full(len(train_df), np.nan)
    oof_lgbm = np.full(len(train_df), np.nan)
    oof_cat = np.full(len(train_df), np.nan)
    index_to_position = {idx: pos for pos, idx in enumerate(train_df.index)}

    for fold_no, (train_times, val_times) in enumerate(folds, 1):
        fold_train = train_df[train_df[cfg.GLOBAL_TIME_COL].isin(train_times)].copy()
        fold_val = train_df[train_df[cfg.GLOBAL_TIME_COL].isin(val_times)].copy()

        fold_preprocessor = build_preprocessor(numerical_features, categorical_features)
        X_tr_processed = fold_preprocessor.fit_transform(fold_train[feature_cols])
        X_va_processed = fold_preprocessor.transform(fold_val[feature_cols])
        y_tr = fold_train[cfg.LOG_TARGET]

        xgb = create_xgb(cfg.RANDOM_STATE + fold_no)
        xgb.fit(X_tr_processed, y_tr)
        lgbm = create_lgbm(cfg.RANDOM_STATE + fold_no)
        lgbm.fit(X_tr_processed, y_tr)

        cat_train = prepare_catboost_data(fold_train, feature_cols, categorical_features)
        cat_val = prepare_catboost_data(fold_val, feature_cols, categorical_features)
        cat = create_catboost(cfg.RANDOM_STATE + fold_no)
        cat.fit(cat_train, y_tr, cat_features=categorical_features)

        pred_xgb = inverse_log_prediction(xgb.predict(X_va_processed))
        pred_lgbm = inverse_log_prediction(lgbm.predict(X_va_processed))
        pred_cat = inverse_log_prediction(cat.predict(cat_val))

        for idx, px, pl, pc in zip(fold_val.index, pred_xgb, pred_lgbm, pred_cat):
            pos = index_to_position[idx]
            oof_xgb[pos], oof_lgbm[pos], oof_cat[pos] = px, pl, pc

    return oof_xgb, oof_lgbm, oof_cat


def optimize_slsqp_weights(y_true, predictions):
    def objective(weights):
        ensemble = weights[0] * predictions[:, 0] + weights[1] * predictions[:, 1] + weights[2] * predictions[:, 2]
        return mean_absolute_error(y_true, ensemble)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1), (0, 1), (0, 1)]
    result = minimize(objective, np.array([1 / 3, 1 / 3, 1 / 3]), method="SLSQP",
                       bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-9})
    return result.x


def train_and_package_model(target_col, feature_cols, numerical_features, categorical_features,
                             train_df, val_df, test_df, log=print):
    """Full OOF -> SLSQP weights -> production-fit -> TEST-eval pipeline for one feature set."""
    oof_xgb, oof_lgbm, oof_cat = generate_oof_predictions(train_df, feature_cols, numerical_features, categorical_features)
    oof_true = train_df[target_col].to_numpy()
    oof_mask = np.isfinite(oof_xgb) & np.isfinite(oof_lgbm) & np.isfinite(oof_cat)

    weights = optimize_slsqp_weights(
        oof_true[oof_mask], np.column_stack([oof_xgb[oof_mask], oof_lgbm[oof_mask], oof_cat[oof_mask]])
    )

    dev_df = pd.concat([train_df, val_df], axis=0).sort_values(cfg.GLOBAL_TIME_COL).copy()
    dev_preprocessor = build_preprocessor(numerical_features, categorical_features)
    X_dev_processed = dev_preprocessor.fit_transform(dev_df[feature_cols])
    X_test_processed = dev_preprocessor.transform(test_df[feature_cols])
    y_dev = dev_df[cfg.LOG_TARGET]

    production_xgb = create_xgb()
    production_xgb.fit(X_dev_processed, y_dev)
    production_lgbm = create_lgbm()
    production_lgbm.fit(X_dev_processed, y_dev)

    cat_X_dev = prepare_catboost_data(dev_df, feature_cols, categorical_features)
    cat_X_test = prepare_catboost_data(test_df, feature_cols, categorical_features)
    production_cat = create_catboost()
    production_cat.fit(cat_X_dev, y_dev, cat_features=categorical_features)

    test_pred = (
        weights[0] * inverse_log_prediction(production_xgb.predict(X_test_processed))
        + weights[1] * inverse_log_prediction(production_lgbm.predict(X_test_processed))
        + weights[2] * inverse_log_prediction(production_cat.predict(cat_X_test))
    )
    test_metrics = evaluate_original_scale(test_df[target_col].to_numpy(), test_pred)
    log(f"  TEST metrics: MAE={test_metrics['MAE']:.3f} RMSE={test_metrics['RMSE']:.3f} R2={test_metrics['R2']:.4f}")

    return {
        "feature_cols": feature_cols,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "preprocessor": dev_preprocessor,
        "model_xgb": production_xgb,
        "model_lgbm": production_lgbm,
        "model_catboost": production_cat,
        "weights": weights,
        "test_metrics": test_metrics,
    }


# =============================================================================
# Deployable pipeline (used for BOTH cost and time overrun)
# =============================================================================
class CombinedOverrunPipeline:
    """
    Self-contained prediction pipeline for one overrun target (cost or time), bundling:
      - Model B      : XGBoost + LightGBM + CatBoost ensemble with a sector interaction
                        feature block (auto-detected from TRAIN as the dominant contributor
                        to the >EXTREME_THRESHOLD% tail).
      - Final Model  : XGBoost + LightGBM + CatBoost ensemble on the full feature set, no
                        interaction features.

    Because this class lives in `app.overrun.core` (a real, versioned module) rather than a
    notebook's `__main__`, it's picklable with plain `pickle` — `train.py` and `main.py`
    `import app.overrun.core`, so unpickling always finds the same class definition. No
    cloudpickle "serialize code by value" needed, and no risk of the bytecode-corruption
    failure mode that comes with shipping notebook-cloudpickle artifacts around by hand.
    """

    def __init__(self, target_col: str, model_b: dict, final_model: dict, sector_name: str):
        self.target_col = target_col
        self.model_b = model_b
        self.final_model = final_model
        self.sector_name = sector_name
        self.sector_key = sector_key_for(sector_name)

    @property
    def required_raw_cols(self) -> list[str]:
        return list(dict.fromkeys([cfg.PROJECT_COL, cfg.DATE_COL] + cfg.RAW_TEXT_COLS + cfg.RAW_NUMERIC_COLS))

    def _ensure_raw_columns(self, df):
        df = df.copy()
        for c in self.required_raw_cols:
            if c not in df.columns:
                df[c] = np.nan
        return df

    def engineer_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Raw CSV-schema dataframe -> full model-ready feature frame (row order preserved)."""
        df = raw_df.copy()
        df["_row_order"] = np.arange(len(df))

        df = self._ensure_raw_columns(df)
        df = basic_cleaning(df)
        df = create_reporting_date(df)
        df = create_core_features(df)
        df = add_missingness_flags(df)
        df = create_temporal_features(df)
        df = create_sector_interaction_features(df, self.sector_name)
        df = sanitize_numeric_data(df)

        df = df.sort_values("_row_order").reset_index(drop=True).drop(columns=["_row_order"])
        return df

    def _predict_one(self, engineered: pd.DataFrame, model_dict: dict) -> np.ndarray:
        feature_cols = model_dict["feature_cols"]
        categorical_features = model_dict["categorical_features"]

        X = engineered.copy()
        for c in feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        X = X[feature_cols]

        X_processed = model_dict["preprocessor"].transform(X)
        # LightGBM artifacts may have been fitted with named transformed
        # features. Preserve those names when the preprocessor exposes them;
        # otherwise keep the original transformed array unchanged.
        lgbm_input = X_processed
        try:
            feature_names = model_dict["preprocessor"].get_feature_names_out()
            if len(feature_names) == X_processed.shape[1]:
                lgbm_input = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        except (AttributeError, TypeError, ValueError):
            pass
        cat_X = X.copy()
        for c in categorical_features:
            cat_X[c] = cat_X[c].fillna("UNKNOWN").astype(str)

        pred_xgb = inverse_log_prediction(model_dict["model_xgb"].predict(X_processed))
        pred_lgbm = inverse_log_prediction(model_dict["model_lgbm"].predict(lgbm_input))
        pred_cat = inverse_log_prediction(model_dict["model_catboost"].predict(cat_X))

        w = model_dict["weights"]
        ensemble = w[0] * pred_xgb + w[1] * pred_lgbm + w[2] * pred_cat
        return np.clip(ensemble, a_min=0.0, a_max=None)

    def predict(self, raw_df: pd.DataFrame, return_frame: bool = True):
        """
        raw_df : DataFrame with the source CSV's raw columns.
        return_frame : if True, returns `raw_df` with two prediction columns appended
                       (`predicted_{target}_model_b`, `predicted_{target}_final_model`);
                       if False, returns {"model_b": array, "final_model": array}.
        """
        engineered = self.engineer_features(raw_df)
        pred_model_b = self._predict_one(engineered, self.model_b)
        pred_final = self._predict_one(engineered, self.final_model)

        if return_frame:
            out = raw_df.reset_index(drop=True).copy()
            out[f"predicted_{self.target_col}_model_b"] = pred_model_b
            out[f"predicted_{self.target_col}_final_model"] = pred_final
            return out
        return {"model_b": pred_model_b, "final_model": pred_final}
