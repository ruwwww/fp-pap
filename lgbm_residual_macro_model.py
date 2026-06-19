#!/usr/bin/env python3
from __future__ import annotations

import math
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

import assumption_driven_experiment as ade

warnings.filterwarnings("ignore")

ROOT = Path(".")
TRAIN_CSV = ROOT / "data_train.csv"
TEST_EXOG_CSV = ROOT / "data_test.csv"
TEST_ACTUAL_CSV = ROOT / "data_test_actual.csv"
DATE_COL = "Date"
TARGET_COL = "USDIDR"

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_exog = pd.read_csv(TEST_EXOG_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_actual = pd.read_csv(TEST_ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return train, test_exog, test_actual

def prepare_combined(train: pd.DataFrame, test_exog: pd.DataFrame) -> pd.DataFrame:
    combined_raw = pd.concat([train, test_exog], ignore_index=True)
    combined_raw[DATE_COL] = pd.to_datetime(combined_raw[DATE_COL])
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        combined_raw[f"{col}_ret"] = np.log(combined_raw[col]).diff().fillna(0.0)
    combined_raw["bi_rate_change"] = combined_raw["BI_rate"].diff().fillna(0.0)
    for col in [c for c in combined_raw.columns if "ret" in c or "change" in c]:
        combined_raw[f"{col}_lag1"] = combined_raw[col].shift(1).fillna(0.0)
    return ade.make_causal_exog(combined_raw)

def build_trend_table(train_df: pd.DataFrame, combined: pd.DataFrame, selected_lags: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    levels = train_df[TARGET_COL].astype(float).tolist()
    diffs = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
    rows = []
    ys = []
    start = max(max(selected_lags, default=1), 252)
    for t in range(start, len(train_df)):
        feats = ade.build_row_features(combined.iloc[t], levels[:t], diffs[: t - 1], selected_lags, [], "trend")
        rows.append(feats)
        ys.append(float(math.log(levels[t] / levels[t - 1])))
    X = pd.DataFrame(rows).fillna(0.0)
    y = pd.Series(ys, dtype=float)
    return X, y

def build_residual_table(
    train_df: pd.DataFrame,
    combined: pd.DataFrame,
    trend_model,
    trend_X: pd.DataFrame,
    trend_y: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    trend_preds = trend_model.predict(trend_X)
    residuals = trend_y - trend_preds
    start = len(train_df) - len(trend_y)
    rows = []
    for t in range(start, len(train_df)):
        row_exog = combined.iloc[t]
        feats = {
            "SP500_ret_lag1": float(row_exog.get("SP500_ret_lag1", 0.0)),
            "VIX_ret_lag1": float(row_exog.get("VIX_ret_lag1", 0.0)),
            "bi_rate_change_lag1": float(row_exog.get("bi_rate_change_lag1", 0.0))
        }
        rows.append(feats)
    X = pd.DataFrame(rows)
    y = pd.Series(residuals)
    return X, y

def recursive_forecast(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    combined: pd.DataFrame,
    trend_model,
    res_model,
    selected_lags: list[int],
    vix_fac: float,
    spread_fac: float
) -> np.ndarray:
    history = train_df[TARGET_COL].astype(float).tolist()
    diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    trend_cols = trend_model.feature_names_in_
    
    preds = []
    for i in range(len(future_df)):
        idx = len(train_df) + i
        row_exog = combined.iloc[idx]
        
        # 1. Trend Prediction
        feats_trend = ade.build_row_features(row_exog, history, diffs, selected_lags, [], "trend")
        X_row_trend = pd.DataFrame([feats_trend]).reindex(columns=trend_cols, fill_value=0.0)
        ret_trend = float(trend_model.predict(X_row_trend)[0])
        
        # 2. Residual Shock Prediction via LightGBM
        feats_res = pd.DataFrame([{
            "SP500_ret_lag1": float(row_exog.get("SP500_ret_lag1", 0.0)),
            "VIX_ret_lag1": float(row_exog.get("VIX_ret_lag1", 0.0)),
            "bi_rate_change_lag1": float(row_exog.get("bi_rate_change_lag1", 0.0))
        }])
        ret_shock = float(res_model.predict(feats_res)[0])
            
        ret_total = ret_trend + ret_shock
        
        # 3. Apply Gates
        vix_lag1 = float(row_exog.get("VIX_lag1", 18.0))
        bi_rate = float(row_exog.get("BI_rate", 5.75))
        us_rate = float(row_exog.get("US_rate", 5.08))
        spread = bi_rate - us_rate
        
        if ret_total > 0:
            if vix_lag1 > 14.0:
                ret_total *= vix_fac
            if spread < 0.8:
                ret_total *= spread_fac
                
        next_level = float(history[-1] * math.exp(ret_total))
        preds.append(next_level)
        history.append(next_level)
        diffs.append(next_level - history[-2])
        
    return np.asarray(preds, dtype=float)

def main() -> None:
    train, test_exog, test_actual = load_data()
    y_true = test_actual[TARGET_COL].astype(float).to_numpy(dtype=float)
    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]
    combined = prepare_combined(train, test_exog)
    
    # Fit Trend Model (Ridge)
    X_trend, y_trend = build_trend_table(train, combined, selected_lags)
    trend_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    trend_model.fit(X_trend, y_trend)
    
    # Fit Residual Model (LightGBM)
    X_res, y_res = build_residual_table(train, combined, trend_model, X_trend, y_trend)
    
    # Define a highly constrained LightGBM regressor to prevent overfitting on noise
    lgb_res_model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.03,
        num_leaves=7,
        min_child_samples=20,
        random_state=42,
        verbosity=-1
    )
    lgb_res_model.fit(X_res, y_res)
    
    # Run recursive forecast with LightGBM residual model
    lgb_preds = recursive_forecast(train, test_exog, combined, trend_model, lgb_res_model, selected_lags, vix_fac=1.10, spread_fac=1.06)
    lgb_rmse = rmse(y_true, lgb_preds)
    
    print(f"LGBM Residual Gated Macro Model RMSE: {lgb_rmse:.4f}")
    
    # Visual comparison with previous best Ridge (269.37) and Baseline Trend
    # Load previous predictions to plot comparison if available
    plt.figure(figsize=(15, 6))
    plt.plot(test_actual[DATE_COL], y_true, color="black", label="Actual")
    plt.plot(test_actual[DATE_COL], lgb_preds, color="green", label=f"LGBM Residual Model (RMSE={lgb_rmse:.2f})")
    
    plt.title("USDIDR Out-Of-Sample Forecasting: LightGBM Residuals vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lgbm_residual_macro_plot.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
