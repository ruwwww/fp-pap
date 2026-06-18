#!/usr/bin/env python3
"""Strict Causal (t-1) ElasticNet pipeline for daily USDIDR time series.

Strict Causality Design:
- Zero look-ahead bias. All stochastic exogenous variables (GOLD, VIX, etc.)
  are strictly shifted to t-1 or older before being used to predict time t.
- Deterministic calendar features (day of week, month) are kept at time t
  because they are known in advance.
- Target (USDIDR) features are built causally using history up to t-1.
- Recursive walk-forward inference ensures predictions are fed back strictly.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# -----------------------------
# Configuration
# -----------------------------
TARGET_COL = "USDIDR"
DEFAULT_LAGS = [1, 5, 21, 63, 126, 252]
DEFAULT_ROLL_WINDOWS = [5, 21, 63]
DEFAULT_TARGET_DIFF_LAGS = [1, 5, 21]
RANDOM_STATE = 42
N_SPLITS = 5
MAX_DAILY_LOG_RETURN = 0.03

# -----------------------------
# Utility functions
# -----------------------------
def infer_date_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in {"date", "datetime", "ds", "timestamp"}]
    if candidates: return candidates[0]
    first_col = df.columns[0]
    try:
        parsed = pd.to_datetime(df[first_col], errors="coerce")
        if parsed.notna().mean() > 0.8: return first_col
    except Exception: pass
    return None

def sort_by_date_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    date_col = infer_date_col(df)
    out = df.copy()
    if date_col is not None:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.sort_values(date_col).reset_index(drop=True)
    else: out = out.reset_index(drop=True)
    return out

def safe_log(series: pd.Series) -> pd.Series:
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
# Strict Causal Feature Engineering
# -----------------------------
def engineer_exogenous_features(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == date_col: continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        
    out["time_index"] = np.arange(len(out), dtype=float)
    
    # 1. Deterministic Calendar Features (Safe to use at time t)
    if date_col is not None:
        dt = pd.to_datetime(out[date_col], errors="coerce")
        out["dow"] = dt.dt.dayofweek.astype(float)
        out["month"] = dt.dt.month.astype(float)
        out["doy"] = dt.dt.dayofyear.astype(float)
        dow_sin, dow_cos = cyclical_encoding(out["dow"], 7)
        doy_sin, doy_cos = cyclical_encoding(out["doy"], 365.25)
        out["dow_sin"], out["dow_cos"], out["doy_sin"], out["doy_cos"] = dow_sin, dow_cos, doy_sin, doy_cos

    raw_exog = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]
    
    if "BI_rate" in out.columns and "US_rate" in out.columns: 
        out["rate_spread"] = out["BI_rate"] - out["US_rate"]
        
    # 2. Explicit Lags (Strictly t-1, t-5, t-21)
    for c in raw_exog + ["rate_spread"]:
        if c in out.columns:
            for lag in [1, 5, 21]:
                out[f"{c}_lag{lag}"] = out[c].shift(lag)
                
    # 3. Shifted Diffs & LogRets (Representing momentum ending at t-1)
    for c in get_existing_cols(out, raw_exog):
        # diff at t is X_t - X_{t-1}. Shift(1) makes it X_{t-1} - X_{t-2}
        out[f"diff_{c}"] = one_step_diff(out[c]).shift(1)
        
    for c in ["GOLD", "SP500", "IHSG"]:
        if c in out.columns:
            # Log return shifted to t-1
            out[f"logret_{c}"] = np.log(out[c] / out[c].shift(1)).shift(1)
            # Log level at t-1
            out[f"log_{c}"] = safe_log(out[c].shift(1))
            
    # 4. Drop the raw t-0 values to enforce strict causality
    out = out.drop(columns=get_existing_cols(out, raw_exog), errors="ignore")
    if "rate_spread" in out.columns:
        out = out.drop(columns=["rate_spread"], errors="ignore")
            
    out = out.copy()
    return out

def build_target_context_features(levels_history: Sequence[float]) -> Dict[str, float]:
    """Fitur lag dan momentum USDIDR murni (history up to t-1)."""
    hist = pd.Series(levels_history, dtype="float64")
    feats: Dict[str, float] = {}
    
    for lag in DEFAULT_LAGS: 
        feats[f"usdidr_lag{lag}"] = hist.iloc[-lag] if len(hist) >= lag else np.nan
        
    if len(hist) >= 2:
        diffs = hist.diff().dropna()
        feats["usdidr_diff_1"] = diffs.iloc[-1]
        for lag in DEFAULT_TARGET_DIFF_LAGS: 
            feats[f"usdidr_diff_lag{lag}"] = diffs.iloc[-lag] if len(diffs) >= lag else np.nan
        for w in DEFAULT_ROLL_WINDOWS:
            recent_diffs = diffs.iloc[-w:] if len(diffs) >= w else diffs
            feats[f"usdidr_diff_mean_{w}"] = recent_diffs.mean() if len(recent_diffs) else np.nan
            feats[f"usdidr_diff_std_{w}"] = recent_diffs.std(ddof=0) if len(recent_diffs) else np.nan
            feats[f"usdidr_trend_{w}"] = hist.iloc[-1] - hist.iloc[-(w + 1)] if len(hist) >= (w + 1) else np.nan

    return feats

def build_training_table(train_df: pd.DataFrame, exog_df: pd.DataFrame, target_col: str = TARGET_COL, min_history: int = 253) -> Tuple[pd.DataFrame, pd.Series]:
    train_levels = pd.to_numeric(train_df[target_col], errors="coerce").to_numpy(dtype=float)
    exog_cols = [c for c in exog_df.columns if c != target_col]
    rows, y = [], []
    
    for t in range(max(min_history, 2), len(train_df)):
        history_levels = train_levels[:t]
        target_feats = build_target_context_features(history_levels)
        row = {c: exog_df.iloc[t][c] if c in exog_df.columns else np.nan for c in exog_cols}
        row.update(target_feats)
        rows.append(row)
        y.append(np.log(train_levels[t]) - np.log(train_levels[t - 1]))
    return pd.DataFrame(rows), pd.Series(y, name="delta_usdidr")

# -----------------------------
# Tuning & Training
# -----------------------------
def tune_and_train_elasticnet(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    print("Hyperparameter Tuning for Strict Causal ElasticNet...")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(max_iter=100000, random_state=RANDOM_STATE))
    ])
    
    param_grid = {
        "model__alpha": [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.5, 1.0],
        "model__l1_ratio": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    }
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    search = GridSearchCV(pipe, param_grid, scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    
    print(f"Best Params: {search.best_params_}")
    print(f"Best CV RMSE: {-search.best_score_:.6f}")
    return search.best_estimator_

# -----------------------------
# Recursive forecasting
# -----------------------------
def forecast_test_period(
    model: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame, combined_exog: pd.DataFrame,
    feature_names: List[str], target_col: str = TARGET_COL, min_history: int = 253,
) -> np.ndarray:
    history_levels = pd.to_numeric(train_df[target_col], errors="coerce").astype(float).tolist()
    exog_cols = [c for c in combined_exog.columns if c != target_col]
    preds = []
    start_idx = len(train_df)
    print(feature_names)
    for i in range(len(test_df)):
        idx = start_idx + i
        row_exog = {c: combined_exog.iloc[idx][c] if c in combined_exog.columns else np.nan for c in exog_cols}
        target_feats = build_target_context_features(history_levels)
        X_row = pd.DataFrame([{**row_exog, **target_feats}])
        X_row = X_row.reindex(columns=feature_names, fill_value=np.nan)
        X_row = X_row.select_dtypes(exclude=["datetime64", "datetimetz"])

        delta_pred = float(model.predict(X_row)[0])
        delta_pred = np.clip(delta_pred, -MAX_DAILY_LOG_RETURN, MAX_DAILY_LOG_RETURN)
        
        level_pred = history_levels[-1] * np.exp(delta_pred)
        preds.append(level_pred)
        history_levels.append(level_pred)

    return np.array(preds, dtype=float)

# -----------------------------
# Main runner
# -----------------------------
def run_pipeline(train_csv: Path, test_csv: Path, submission_csv: Path, actual_test_csv: Optional[Path] = None) -> None:
    train_raw = pd.read_csv(train_csv)
    test_raw = pd.read_csv(test_csv)
    actual_test_raw = pd.read_csv(actual_test_csv) if actual_test_csv is not None else None

    train_raw = sort_by_date_if_possible(train_raw)
    test_raw = sort_by_date_if_possible(test_raw)
    if actual_test_raw is not None: 
        actual_test_raw = sort_by_date_if_possible(actual_test_raw)

    train_date_col = infer_date_col(train_raw)
    test_date_col = infer_date_col(test_raw)
    effective_date_col = train_date_col if train_date_col == test_date_col else train_date_col

    train_exog = train_raw.drop(columns=[TARGET_COL], errors="ignore")
    test_exog = test_raw.copy()
    combined_exog = pd.concat([train_exog, test_exog], ignore_index=True)
    
    # Explicit chronological sorting BEFORE feature engineering
    if effective_date_col and effective_date_col in combined_exog.columns:
        combined_exog[effective_date_col] = pd.to_datetime(combined_exog[effective_date_col], errors="coerce")
        combined_exog = combined_exog.sort_values(effective_date_col).reset_index(drop=True)
    else:
        combined_exog = combined_exog.reset_index(drop=True)

    combined_exog = engineer_exogenous_features(combined_exog, date_col=effective_date_col)
    datetime_cols = combined_exog.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    if datetime_cols: combined_exog = combined_exog.drop(columns=datetime_cols)

    X_train, y_train = build_training_table(
        train_df=train_raw, 
        exog_df=combined_exog.iloc[: len(train_raw)].reset_index(drop=True)
    )
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    feature_cols = X_train.columns.tolist()

    print(f"Training rows used: {len(X_train)}")
    print(f"Feature count: {len(feature_cols)}\n")

    model = tune_and_train_elasticnet(X_train, y_train)
    print("\n" + "="*60 + "\nStrict Causal ElasticNet training complete.\n" + "="*60 + "\n")

    test_preds = forecast_test_period(
        model=model, train_df=train_raw, test_df=test_raw, combined_exog=combined_exog.reset_index(drop=True),
        feature_names=feature_cols
    )

    submission = pd.DataFrame({TARGET_COL: test_preds})
    if test_date_col and test_date_col in test_raw.columns:
        submission.insert(0, "Date", test_raw[test_date_col].values)
    submission_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_csv, index=False)
    print(f"Saved submission to: {submission_csv}")

    if actual_test_raw is not None:
        if TARGET_COL not in actual_test_raw.columns: 
            raise ValueError(f"actual_test_csv must contain '{TARGET_COL}'.")
        actual = pd.to_numeric(actual_test_raw[TARGET_COL], errors="coerce").to_numpy(dtype=float)
        if len(actual) != len(test_preds): 
            raise ValueError("Length mismatch between predictions and actuals.")
        rmse = float(np.sqrt(mean_squared_error(actual, test_preds)))
        print(f"True test RMSE (USDIDR level): {rmse:.4f}")

    coef = pd.Series(
        model.named_steps["model"].coef_,
        index=feature_cols
    ).sort_values(key=np.abs, ascending=False)

    print(coef.head(20))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="USDIDR Strict Causal ElasticNet Pipeline")
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--submission_csv", type=Path, required=True)
    p.add_argument("--actual_test_csv", type=Path, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.train_csv, args.test_csv, args.submission_csv, args.actual_test_csv)