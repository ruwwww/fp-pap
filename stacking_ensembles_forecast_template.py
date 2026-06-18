#!/usr/bin/env python3
"""Time-Series Stacking Ensemble pipeline for daily USDIDR forecasting.

This script implements a robust ensemble strategy combining:
1.  Linear Anchor (ElasticNet): Captures baseline macro relationships and
    mean-reversion without overfitting to noise.
2.  Non-Linear Mapper (Constrained XGBoost): Shallow trees (max_depth=3) with
    strong L1/L2 regularization to capture only high-confidence regime shifts
    or threshold effects.
3.  Meta-Learner (Ridge Regression): Blends the two models based on their
    Out-of-Fold (OOF) predictions using TimeSeriesSplit to prevent data leakage.

Forecasting Scheme:
- Predicts daily log-returns (deltas) and reconstructs levels recursively.
- Recursive walk-forward inference feeds predicted levels back into history
  for dynamic lag/EMA/RSI feature generation.

Example
-------
python ensemble_usdidr_forecast.py \
    --train_csv data_train.csv \
    --test_csv data_test.csv \
    --submission_csv submission.csv

Optional benchmark:
python ensemble_usdidr_forecast.py \
    --train_csv data_train.csv \
    --test_csv data_test.csv \
    --actual_test_csv actual_test.csv \
    --submission_csv submission.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------
# Configuration
# -----------------------------
TARGET_COL = "USDIDR"
DEFAULT_LAGS = [1, 5, 21, 63]
DEFAULT_ROLL_WINDOWS = [5, 21, 63]
DEFAULT_TARGET_DIFF_LAGS = [1, 5, 21]
EMA_SPANS = [5, 21]
RSI_WINDOWS = [7, 14]
RANDOM_STATE = 42
N_SPLITS = 5


# -----------------------------
# Utility functions
# -----------------------------

def infer_date_col(df: pd.DataFrame) -> Optional[str]:
    """Infer a date column if one exists in the dataframe."""
    candidates = [c for c in df.columns if c.lower() in {"date", "datetime", "ds", "timestamp"}]
    if candidates:
        return candidates[0]

    first_col = df.columns[0]
    try:
        parsed = pd.to_datetime(df[first_col], errors="coerce")
        if parsed.notna().mean() > 0.8:
            return first_col
    except Exception:
        pass
    return None


def sort_by_date_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    """Sort dataframe by its date column if one is detected."""
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


def compute_rsi(levels: pd.Series, window: int) -> float:
    """Compute the most recent RSI value from a series of price levels."""
    if len(levels) < window + 1:
        return np.nan
    diffs = levels.diff().dropna()
    if len(diffs) < window:
        return np.nan
    gains = diffs.clip(lower=0.0)
    losses = (-diffs.clip(upper=0.0))
    avg_gain = gains.iloc[-window:].mean()
    avg_loss = losses.iloc[-window:].mean()
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_ema_diff(levels: pd.Series, span: int) -> float:
    """EMA of the daily differences of a level series."""
    diffs = levels.diff().dropna()
    if len(diffs) < span:
        return np.nan
    return float(diffs.ewm(span=span, adjust=False).mean().iloc[-1])


# -----------------------------
# Feature Engineering
# -----------------------------

def engineer_exogenous_features(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    """Create features from known-at-prediction-time exogenous variables."""
    out = df.copy()

    for c in out.columns:
        if c == date_col:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["time_index"] = np.arange(len(out), dtype=float)

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

    if "BI_rate" in out.columns and "US_rate" in out.columns:
        out["rate_spread"] = out["BI_rate"] - out["US_rate"]

    for c in ["GOLD", "SP500", "IHSG"]:
        if c in out.columns:
            out[f"log_{c}"] = safe_log(out[c])

    diff_cols = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]
    for c in get_existing_cols(out, diff_cols):
        out[f"diff_{c}"] = one_step_diff(out[c])

    for c in ["GOLD", "SP500", "IHSG"]:
        if c in out.columns:
            out[f"logret_{c}"] = np.log(out[c] / out[c].shift(1))

    lag_cols = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate", "rate_spread"]
    for c in get_existing_cols(out, lag_cols):
        for lag in [1, 5, 21]:
            out[f"{c}_lag{lag}"] = out[c].shift(lag)

    for c in ["OIL", "GOLD", "VIX", "SP500", "IHSG"]:
        if c in out.columns:
            for w in [5, 21]:
                out[f"{c}_rollmean_{w}"] = out[c].rolling(w).mean()
                out[f"{c}_rollstd_{w}"] = out[c].rolling(w).std(ddof=0)

    # Cross / interaction features
    if "rate_spread" in out.columns and "VIX" in out.columns:
        out["rate_spread_x_vix"] = out["rate_spread"] * out["VIX"]
        if "GOLD" in out.columns:
            gold_safe = out["GOLD"].where(out["GOLD"] > 0, np.nan)
            out["rate_spread_div_gold"] = out["rate_spread"] / gold_safe

    if "GOLD" in out.columns and "VIX" in out.columns:
        out["gold_x_vix"] = out["GOLD"] * out["VIX"]

    if "OIL" in out.columns and "US_rate" in out.columns:
        out["oil_x_us_rate"] = out["OIL"] * out["US_rate"]

    if "IHSG" in out.columns and "VIX" in out.columns:
        out["ihsg_x_vix"] = out["IHSG"] * out["VIX"]

    if "diff_VIX" in out.columns and "diff_GOLD" in out.columns:
        out["diff_vix_x_diff_gold"] = out["diff_VIX"] * out["diff_GOLD"]

    return out


def build_target_context_features(
    levels_history: Sequence[float],
    windows: Sequence[int] = DEFAULT_ROLL_WINDOWS,
    lag_levels: Sequence[int] = DEFAULT_LAGS,
    lag_diffs: Sequence[int] = DEFAULT_TARGET_DIFF_LAGS,
) -> Dict[str, float]:
    """Features based on past USDIDR values only (Lags, EMA, RSI)."""
    hist = pd.Series(levels_history, dtype="float64")
    feats: Dict[str, float] = {}

    for lag in lag_levels:
        feats[f"usdidr_lag{lag}"] = hist.iloc[-lag] if len(hist) >= lag else np.nan

    if len(hist) >= 2:
        diffs = hist.diff().dropna()
        feats["usdidr_diff_1"] = diffs.iloc[-1]

        for lag in lag_diffs:
            feats[f"usdidr_diff_lag{lag}"] = diffs.iloc[-lag] if len(diffs) >= lag else np.nan

        for w in windows:
            recent_diffs = diffs.iloc[-w:] if len(diffs) >= w else diffs
            feats[f"usdidr_diff_mean_{w}"] = recent_diffs.mean() if len(recent_diffs) else np.nan
            feats[f"usdidr_diff_std_{w}"] = recent_diffs.std(ddof=0) if len(recent_diffs) else np.nan
            feats[f"usdidr_trend_{w}"] = (
                hist.iloc[-1] - hist.iloc[-(w + 1)] if len(hist) >= (w + 1) else np.nan
            )

        for span in EMA_SPANS:
            feats[f"usdidr_diff_ema_{span}"] = compute_ema_diff(hist, span)

        if not np.isnan(feats.get("usdidr_diff_ema_5", np.nan)):
            ema5 = feats["usdidr_diff_ema_5"]
            if ema5 != 0:
                feats["usdidr_diff_over_ema5"] = feats["usdidr_diff_1"] / ema5
            else:
                feats["usdidr_diff_over_ema5"] = np.nan
        else:
            feats["usdidr_diff_over_ema5"] = np.nan

    for w in RSI_WINDOWS:
        feats[f"usdidr_rsi_{w}"] = compute_rsi(hist, w)

    if len(hist) >= 22:
        feats["usdidr_pct_change_21"] = hist.iloc[-1] / hist.iloc[-22] - 1.0
    else:
        feats["usdidr_pct_change_21"] = np.nan

    return feats


def build_training_table(
    train_df: pd.DataFrame,
    exog_df: pd.DataFrame,
    target_col: str = TARGET_COL,
    min_history: int = 64,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Convert the training series into a supervised learning table."""
    if target_col not in train_df.columns:
        raise ValueError(f"Training dataframe must contain target column '{target_col}'.")

    train_levels = pd.to_numeric(train_df[target_col], errors="coerce").to_numpy(dtype=float)
    exog_cols = [c for c in exog_df.columns if c != target_col]

    rows: List[Dict[str, float]] = []
    y: List[float] = []

    for t in range(max(min_history, 2), len(train_df)):
        history_levels = train_levels[:t]
        target_feats = build_target_context_features(history_levels)

        row: Dict[str, float] = {}
        for c in exog_cols:
            row[c] = exog_df.iloc[t][c] if c in exog_df.columns else np.nan

        row.update(target_feats)
        rows.append(row)
        y.append(np.log(train_levels[t]) - np.log(train_levels[t - 1]))

    X = pd.DataFrame(rows)
    y_series = pd.Series(y, name="delta_usdidr")
    return X, y_series


# -----------------------------
# Stacking Ensemble Components
# -----------------------------

def make_linear_model() -> Pipeline:
    """ElasticNet pipeline as the Linear Anchor."""
    # Fixed reasonable hyperparameters for financial time-series
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=50000, random_state=RANDOM_STATE))
    ])


def make_nonlinear_model() -> Pipeline:
    """Constrained XGBoost as the Non-Linear Mapper.
    
    Kept extremely shallow (max_depth=3) with high regularization to prevent
    it from memorizing noise, allowing it only to capture high-confidence
    regime shifts.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            n_estimators=200,
            max_depth=3,          # Strictly shallow
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=10,  # High resistance to noise
            reg_alpha=1.0,        # L1
            reg_lambda=5.0,       # L2
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0
        ))
    ])


def train_stacking_ensemble(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Tuple[Pipeline, Pipeline, Ridge]:
    """Trains the stacking ensemble using OOF predictions.
    
    Returns the trained linear model, nonlinear model, and meta-learner.
    """
    print("Generating Out-of-Fold (OOF) predictions for stacking...")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    oof_preds_linear = np.zeros(len(X_train))
    oof_preds_nonlinear = np.zeros(len(X_train))
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Clone models for this fold
        linear_model = make_linear_model()
        nonlinear_model = make_nonlinear_model()
        
        linear_model.fit(X_tr, y_tr)
        nonlinear_model.fit(X_tr, y_tr)
        
        oof_preds_linear[val_idx] = linear_model.predict(X_val)
        oof_preds_nonlinear[val_idx] = nonlinear_model.predict(X_val)
        print(f"  Fold {fold+1}/{N_SPLITS} completed.")
        
    # Prepare Meta-Features
    X_meta_train = np.column_stack((oof_preds_linear, oof_preds_nonlinear))
    
    # Train Meta-Learner (Ridge with non-negative weights constraint)
    # positive=True enforces that the blender can only assign positive weights,
    # preventing it from shorting a model which is unstable in finance.
    meta_model = Ridge(alpha=1.0, positive=True, random_state=RANDOM_STATE)
    meta_model.fit(X_meta_train, y_train)
    
    print(f"Meta-Learner Coefficients (Linear, NonLinear): {meta_model.coef_}")
    
    # Retrain base models on the FULL training set for test inference
    print("Retraining base models on full dataset...")
    final_linear_model = make_linear_model()
    final_nonlinear_model = make_nonlinear_model()
    
    final_linear_model.fit(X_train, y_train)
    final_nonlinear_model.fit(X_train, y_train)
    
    return final_linear_model, final_nonlinear_model, meta_model


# -----------------------------
# Recursive forecasting
# -----------------------------

def forecast_test_period(
    linear_model: Pipeline,
    nonlinear_model: Pipeline,
    meta_model: Ridge,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    combined_exog: pd.DataFrame,
    feature_names: List[str],
    target_col: str = TARGET_COL,
    min_history: int = 64,
) -> np.ndarray:
    """Forecast test USDIDR recursively using the Stacking Ensemble."""
    if target_col not in train_df.columns:
        raise ValueError(f"Training dataframe must contain target column '{target_col}'.")

    history_levels = pd.to_numeric(train_df[target_col], errors="coerce").astype(float).tolist()
    exog_cols = [c for c in combined_exog.columns if c != target_col]

    preds: List[float] = []
    start_idx = len(train_df)

    for i in range(len(test_df)):
        idx = start_idx + i
        row_exog = {
            c: combined_exog.iloc[idx][c] if c in combined_exog.columns else np.nan
            for c in exog_cols
        }
        target_feats = build_target_context_features(history_levels)
        feat_row = {**row_exog, **target_feats}
        X_row = pd.DataFrame([feat_row])

        X_row = X_row.reindex(columns=feature_names, fill_value=np.nan)
        X_row = X_row.select_dtypes(exclude=["datetime64", "datetimetz"])

        # Predict deltas from both base models
        delta_linear = float(linear_model.predict(X_row)[0])
        delta_nonlinear = float(nonlinear_model.predict(X_row)[0])
        
        # Blend using Meta-Learner
        X_meta_row = np.array([[delta_linear, delta_nonlinear]])
        delta_final = float(meta_model.predict(X_meta_row)[0])
        
        # Reconstruct level
        level_pred = history_levels[-1] * np.exp(delta_final)
        preds.append(level_pred)
        history_levels.append(level_pred)

    return np.array(preds, dtype=float)


# -----------------------------
# Main runner
# -----------------------------

def run_pipeline(
    train_csv: Path,
    test_csv: Path,
    submission_csv: Path,
    actual_test_csv: Optional[Path] = None,
) -> None:
    """Run the full Stacking Ensemble USDIDR forecasting pipeline end-to-end."""
    train_raw = pd.read_csv(train_csv)
    test_raw = pd.read_csv(test_csv)
    actual_test_raw = pd.read_csv(actual_test_csv) if actual_test_csv is not None else None

    train_raw = sort_by_date_if_possible(train_raw)
    test_raw = sort_by_date_if_possible(test_raw)
    if actual_test_raw is not None:
        actual_test_raw = sort_by_date_if_possible(actual_test_raw)

    train_date_col = infer_date_col(train_raw)
    test_date_col = infer_date_col(test_raw)

    train_exog = train_raw.drop(columns=[TARGET_COL], errors="ignore")
    test_exog = test_raw.copy()

    combined_exog = pd.concat([train_exog, test_exog], ignore_index=True)
    effective_date_col = train_date_col if train_date_col == test_date_col else train_date_col
    combined_exog = engineer_exogenous_features(combined_exog, date_col=effective_date_col)

    datetime_cols = combined_exog.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    if datetime_cols:
        combined_exog = combined_exog.drop(columns=datetime_cols)

    X_train, y_train = build_training_table(
        train_df=train_raw,
        exog_df=combined_exog.iloc[: len(train_raw)].reset_index(drop=True),
        target_col=TARGET_COL,
        min_history=max(DEFAULT_LAGS + [64]),
    )

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    feature_cols = X_train.columns.tolist()

    print(f"Training rows used: {len(X_train)}")
    print(f"Feature count: {len(feature_cols)}")

    # Train the Ensemble
    linear_model, nonlinear_model, meta_model = train_stacking_ensemble(X_train, y_train)
    print("=" * 60)
    print("Stacking Ensemble training complete.")
    print("=" * 60)

    # Forecast
    test_preds = forecast_test_period(
        linear_model=linear_model,
        nonlinear_model=nonlinear_model,
        meta_model=meta_model,
        train_df=train_raw,
        test_df=test_raw,
        combined_exog=combined_exog.reset_index(drop=True),
        feature_names=feature_cols,
        target_col=TARGET_COL,
        min_history=max(DEFAULT_LAGS + [64]),
    )

    # Save submission
    submission = pd.DataFrame({TARGET_COL: test_preds})
    if test_date_col and test_date_col in test_raw.columns:
        submission.insert(0, "Date", test_raw[test_date_col].values)
    submission_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_csv, index=False)
    print(f"Saved submission to: {submission_csv}")

    # Optional benchmark
    if actual_test_raw is not None:
        if TARGET_COL not in actual_test_raw.columns:
            raise ValueError(f"actual_test_csv must contain target column '{TARGET_COL}'.")
        actual = pd.to_numeric(actual_test_raw[TARGET_COL], errors="coerce").to_numpy(dtype=float)
        if len(actual) != len(test_preds):
            raise ValueError(
                f"Length mismatch: predictions={len(test_preds)}, actual={len(actual)}"
            )
        rmse = float(np.sqrt(mean_squared_error(actual, test_preds)))
        print(f"True test RMSE (USDIDR level): {rmse:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="USDIDR Stacking Ensemble forecasting pipeline")
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