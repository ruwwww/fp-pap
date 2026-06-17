#!/usr/bin/env python3
"""ElasticNet forecasting template for daily USDIDR time series.

This script is intentionally self-contained and focused on one model only.
It follows the design choices discussed earlier:
- forecast USDIDR *changes* (deltas) rather than raw levels
- use lagged USDIDR context (lag1 / lag5 / lag21 / lag63)
- add macro feature transforms (logs, differences, spreads)
- preserve time order (no random split)
- recursively forecast the test period one day at a time

Expected CSV layout
-------------------
Train CSV:
    contains feature columns and target column `USDIDR`
    optionally contains a date column (e.g. `Date`)

Test CSV:
    contains feature columns only (no USDIDR)
    optionally contains the same date column

Output
------
- submission CSV with a `USDIDR` column (Kaggle-style)
- optional benchmark RMSE if you provide a CSV with actual test USDIDR

Example
-------
python elasticnet_forecast_template.py \
    --train_csv data_train.csv \
    --test_csv data_test.csv \
    --submission_csv submission.csv

Optional benchmark against real test labels:
python elasticnet_forecast_template.py \
    --train_csv data_train.csv \
    --test_csv data_test.csv \
    --actual_test_csv actual_test.csv \
    --submission_csv submission.csv
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------
# Configuration
# -----------------------------
TARGET_COL = "USDIDR"
DEFAULT_LAGS = [1, 5, 21, 63]
DEFAULT_ROLL_WINDOWS = [5, 21, 63]
DEFAULT_TARGET_DIFF_LAGS = [1, 5, 21]
RANDOM_STATE = 42


# -----------------------------
# Utility functions
# -----------------------------

def infer_date_col(df: pd.DataFrame) -> Optional[str]:
    """Infer a date column if one exists."""
    candidates = [c for c in df.columns if c.lower() in {"date", "datetime", "ds", "timestamp"}]
    if candidates:
        return candidates[0]

    # Try the first column if it can be parsed as dates.
    first_col = df.columns[0]
    try:
        parsed = pd.to_datetime(df[first_col], errors="coerce")
        if parsed.notna().mean() > 0.8:
            return first_col
    except Exception:
        pass
    return None


def sort_by_date_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    date_col = infer_date_col(df)
    out = df.copy()
    if date_col is not None:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.sort_values(date_col).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def safe_log(series: pd.Series) -> pd.Series:
    """Log transform only on strictly positive values."""
    s = pd.to_numeric(series, errors="coerce")
    return np.log(s.where(s > 0))


def one_step_diff(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").diff()


def cyclical_encoding(values: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
    theta = 2.0 * np.pi * values / period
    return np.sin(theta), np.cos(theta)


def get_existing_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


# -----------------------------
# Exogenous feature engineering
# -----------------------------

def engineer_exogenous_features(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    """Create features from known-at-prediction-time exogenous variables.

    These features are safe because they depend only on current/past exogenous
    values, not on the target USDIDR.
    """
    out = df.copy()

    # Ensure numeric where possible.
    for c in out.columns:
        if c == date_col:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Time index across the concatenated train+test horizon.
    out["time_index"] = np.arange(len(out), dtype=float)

    # Calendar features, if a date column exists.
    if date_col is not None:
        dt = pd.to_datetime(out[date_col], errors="coerce")
        out["dow"] = dt.dt.dayofweek.astype(float)
        out["month"] = dt.dt.month.astype(float)
        out["doy"] = dt.dt.dayofyear.astype(float)
        dow_sin, dow_cos = cyclical_encoding(out["dow"], 7)
        doy_sin, doy_cos = cyclical_encoding(out["doy"], 365.25)
        out["dow_sin"] = dow_sin
        out["dow_cos"] = dow_cos
        out["doy_sin"] = doy_sin
        out["doy_cos"] = doy_cos

    # Core spread feature.
    if "BI_rate" in out.columns and "US_rate" in out.columns:
        out["rate_spread"] = out["BI_rate"] - out["US_rate"]

    # Log transforms for long-run growers.
    for c in ["GOLD", "SP500", "IHSG"]:
        if c in out.columns:
            out[f"log_{c}"] = safe_log(out[c])

    # Simple first differences for exogenous series.
    diff_cols = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]
    for c in get_existing_cols(out, diff_cols):
        out[f"diff_{c}"] = one_step_diff(out[c])

    # Log returns for positive macro series.
    for c in ["GOLD", "SP500", "IHSG"]:
        if c in out.columns:
            out[f"logret_{c}"] = np.log(out[c] / out[c].shift(1))

    # A few lag features for exogenous variables. These are safe because future
    # test exogenous values are already known in the test CSV.
    lag_cols = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate", "rate_spread"]
    for c in get_existing_cols(out, lag_cols):
        for lag in [1, 5, 21]:
            out[f"{c}_lag{lag}"] = out[c].shift(lag)

    return out


# -----------------------------
# Target-context feature engineering
# -----------------------------

def build_target_context_features(
    levels_history: Sequence[float],
    windows: Sequence[int] = DEFAULT_ROLL_WINDOWS,
    lag_levels: Sequence[int] = DEFAULT_LAGS,
    lag_diffs: Sequence[int] = DEFAULT_TARGET_DIFF_LAGS,
) -> Dict[str, float]:
    """Features based on past USDIDR values only.

    This function is used in two places:
    - training: build features for each supervised row using actual history
    - forecasting: build features row-by-row using actual + predicted history
    """
    hist = pd.Series(levels_history, dtype="float64")
    feats: Dict[str, float] = {}

    # Lags of the level series.
    for lag in lag_levels:
        feats[f"usdidr_lag{lag}"] = hist.iloc[-lag] if len(hist) >= lag else np.nan

    # Recent differences / momentum.
    if len(hist) >= 2:
        diffs = hist.diff().dropna()
        feats["usdidr_diff_1"] = diffs.iloc[-1]

        for lag in lag_diffs:
            feats[f"usdidr_diff_lag{lag}"] = diffs.iloc[-lag] if len(diffs) >= lag else np.nan

        for w in windows:
            recent_diffs = diffs.iloc[-w:] if len(diffs) >= w else diffs
            feats[f"usdidr_diff_mean_{w}"] = recent_diffs.mean() if len(recent_diffs) else np.nan
            feats[f"usdidr_diff_std_{w}"] = recent_diffs.std(ddof=0) if len(recent_diffs) else np.nan
            feats[f"usdidr_trend_{w}"] = hist.iloc[-1] - hist.iloc[-(w + 1)] if len(hist) >= (w + 1) else np.nan

    # Extra ratio-like context that is often useful in FX.
    if len(hist) >= 22:
        feats["usdidr_pct_change_21"] = hist.iloc[-1] / hist.iloc[-22] - 1.0
    else:
        feats["usdidr_pct_change_21"] = np.nan

    return feats


# -----------------------------
# Supervised dataset construction
# -----------------------------

def build_training_table(
    train_df: pd.DataFrame,
    exog_df: pd.DataFrame,
    target_col: str = TARGET_COL,
    min_history: int = 64,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Convert the training series into a supervised learning table.

    Predict the *daily change* (delta) in USDIDR, not the raw level.
    The label at time t is:
        delta_t = USDIDR_t - USDIDR_{t-1}

    Features at time t use only information available up to t-1.
    """
    if target_col not in train_df.columns:
        raise ValueError(f"Training dataframe must contain target column '{target_col}'.")

    train_levels = pd.to_numeric(train_df[target_col], errors="coerce").to_numpy(dtype=float)
    exog_cols = [c for c in exog_df.columns if c != target_col]

    rows: List[Dict[str, float]] = []
    y: List[float] = []

    # We need at least one past observation to define the delta; and enough
    # history for lag features.
    for t in range(max(min_history, 2), len(train_df)):
        history_levels = train_levels[:t]  # available up to t-1
        target_feats = build_target_context_features(history_levels)

        row: Dict[str, float] = {}
        # Exogenous features for the current time step t.
        for c in exog_cols:
            row[c] = exog_df.iloc[t][c] if c in exog_df.columns else np.nan

        row.update(target_feats)
        rows.append(row)
        y.append(np.log(train_levels[t]) - np.log(train_levels[t - 1]))

    X = pd.DataFrame(rows)
    y_series = pd.Series(y, name="delta_usdidr")
    return X, y_series


# -----------------------------
# Recursive forecasting
# -----------------------------

def forecast_test_period(
    model: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    combined_exog: pd.DataFrame,
    target_col: str = TARGET_COL,
    min_history: int = 64,
) -> np.ndarray:
    """Forecast test USDIDR recursively day-by-day."""
    if target_col not in train_df.columns:
        raise ValueError(f"Training dataframe must contain target column '{target_col}'.")

    history_levels = pd.to_numeric(train_df[target_col], errors="coerce").astype(float).tolist()
    exog_cols = [c for c in combined_exog.columns if c != target_col]

    preds: List[float] = []

    start_idx = len(train_df)
    for i in range(len(test_df)):
        idx = start_idx + i
        # Current exogenous features at the forecast date.
        row_exog = {c: combined_exog.iloc[idx][c] if c in combined_exog.columns else np.nan for c in exog_cols}
        target_feats = build_target_context_features(history_levels)
        feat_row = {**row_exog, **target_feats}
        X_row = pd.DataFrame([feat_row])

        # Align columns to training columns.
        X_row = X_row.reindex(columns=model.feature_names_in_, fill_value=np.nan)
        # Drop any datetime columns that can't be processed by the pipeline.
        X_row = X_row.select_dtypes(exclude=["datetime64", "datetimetz"])
        delta_pred = float(model.predict(X_row)[0])
        level_pred = history_levels[-1] * np.exp(delta_pred)
        preds.append(level_pred)
        history_levels.append(level_pred)

    return np.array(preds, dtype=float)


# -----------------------------
# Modeling
# -----------------------------

def make_model() -> GridSearchCV:
    """ElasticNet pipeline + small time-series-aware hyperparameter grid."""
    base_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=50000, random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {
        "model__alpha": np.logspace(-4, 1, 12),
        "model__l1_ratio": [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.95],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    search = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    return search


# -----------------------------
# Main runner
# -----------------------------

def run_pipeline(
    train_csv: Path,
    test_csv: Path,
    submission_csv: Path,
    actual_test_csv: Optional[Path] = None,
) -> None:
    # Load data.
    train_raw = pd.read_csv(train_csv)
    test_raw = pd.read_csv(test_csv)
    actual_test_raw = pd.read_csv(actual_test_csv) if actual_test_csv is not None else None

    # Sort by date if possible.
    train_raw = sort_by_date_if_possible(train_raw)
    test_raw = sort_by_date_if_possible(test_raw)
    if actual_test_raw is not None:
        actual_test_raw = sort_by_date_if_possible(actual_test_raw)

    train_date_col = infer_date_col(train_raw)
    test_date_col = infer_date_col(test_raw)

    # Align date columns / drop target from exogenous feature set.
    train_exog = train_raw.drop(columns=[TARGET_COL], errors="ignore")
    test_exog = test_raw.copy()

    # Combine train+test exogenous data so lagged exogenous features at the
    # start of the test period can use the last train observations.
    combined_exog = pd.concat([train_exog, test_exog], ignore_index=True)
    effective_date_col = train_date_col if train_date_col == test_date_col else train_date_col
    combined_exog = engineer_exogenous_features(combined_exog, date_col=effective_date_col)

    # Drop all datetime columns after feature engineering (calendar features are already extracted).
    datetime_cols = combined_exog.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    if datetime_cols:
        combined_exog = combined_exog.drop(columns=datetime_cols)

    # Build supervised training table.
    X_train, y_train = build_training_table(
        train_df=train_raw,
        exog_df=combined_exog.iloc[: len(train_raw)].reset_index(drop=True),
        target_col=TARGET_COL,
        min_history=max(DEFAULT_LAGS + [64]),
    )

    # Keep only numeric columns and enforce consistent ordering.
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    feature_cols = X_train.columns.tolist()

    print(f"Training rows used: {len(X_train)}")
    print(f"Feature count: {len(feature_cols)}")

    # Hyperparameter search.
    model_search = make_model()
    model_search.fit(X_train, y_train)
    best_model: Pipeline = model_search.best_estimator_
    print("Best params:", model_search.best_params_)
    print(f"Best CV RMSE: {-model_search.best_score_:.4f}")

    # Recursive forecasting for the test period.
    test_preds = forecast_test_period(
        model=best_model,
        train_df=train_raw,
        test_df=test_raw,
        combined_exog=combined_exog.reset_index(drop=True),
        target_col=TARGET_COL,
        min_history=max(DEFAULT_LAGS + [64]),
    )

    # Save submission.
    submission = pd.DataFrame({TARGET_COL: test_preds})
    if test_date_col and test_date_col in test_raw.columns:
        submission.insert(0, "Date", test_raw[test_date_col].values)
    submission_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_csv, index=False)
    print(f"Saved submission to: {submission_csv}")

    # Optional benchmark if actual test labels are available.
    if actual_test_raw is not None:
        if TARGET_COL not in actual_test_raw.columns:
            raise ValueError(f"actual_test_csv must contain target column '{TARGET_COL}'.")
        actual = pd.to_numeric(actual_test_raw[TARGET_COL], errors="coerce").to_numpy(dtype=float)
        if len(actual) != len(test_preds):
            raise ValueError(
                f"Length mismatch: predictions={len(test_preds)}, actual={len(actual)}"
            )
        rmse = mean_squared_error(actual, test_preds, squared=False)
        print(f"True test RMSE: {rmse:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="USDIDR ElasticNet forecasting template")
    p.add_argument("--train_csv", type=Path, required=True, help="Path to training CSV")
    p.add_argument("--test_csv", type=Path, required=True, help="Path to test CSV")
    p.add_argument("--submission_csv", type=Path, required=True, help="Output submission CSV")
    p.add_argument(
        "--actual_test_csv",
        type=Path,
        default=None,
        help="Optional CSV with actual test USDIDR for honest benchmarking",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        submission_csv=args.submission_csv,
        actual_test_csv=args.actual_test_csv,
    )
