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

def build_ood_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    
    # 1. Percentile Ranks (0.0 - 1.0)
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        out[f"{col}_rank_252"] = out[col].rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
        
    # 2. Return Shock Magnitudes (z-score of return relative to 20-day rolling std)
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        ret = out[col].pct_change()
        vol = ret.rolling(20, min_periods=5).std()
        out[f"{col}_shock"] = (ret / vol.replace(0.0, np.nan)).fillna(0.0)
        
    # 3. Carry Interest Spread Rank
    out["carry_spread"] = out["BI_rate"] - out["US_rate"]
    out["carry_spread_rank_252"] = out["carry_spread"].rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
    
    # Lag 1 of all features to prevent future leakage
    exog_cols = [c for c in out.columns if "_rank_252" in c or "_shock" in c]
    for col in exog_cols:
        out[f"{col}_lag1"] = out[col].shift(1).fillna(0.0)
        
    return out

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

def build_residual_table_ood(
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
            "SP500_rank_lag1": float(row_exog.get("SP500_rank_252_lag1", 0.5)),
            "VIX_rank_lag1": float(row_exog.get("VIX_rank_252_lag1", 0.5)),
            "SP500_shock_lag1": float(row_exog.get("SP500_shock_lag1", 0.0)),
            "VIX_shock_lag1": float(row_exog.get("VIX_shock_lag1", 0.0)),
            "carry_spread_rank_lag1": float(row_exog.get("carry_spread_rank_252_lag1", 0.5))
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
    res_cols = res_model.feature_names_in_
    
    preds = []
    for i in range(len(future_df)):
        idx = len(train_df) + i
        row_exog = combined.iloc[idx]
        
        # 1. Trend Prediction
        feats_trend = ade.build_row_features(row_exog, history, diffs, selected_lags, [], "trend")
        X_row_trend = pd.DataFrame([feats_trend]).reindex(columns=trend_cols, fill_value=0.0)
        ret_trend = float(trend_model.predict(X_row_trend)[0])
        
        # 2. Residual Shock Prediction via OOD-safe Features
        feats_res = {
            "SP500_rank_lag1": float(row_exog.get("SP500_rank_252_lag1", 0.5)),
            "VIX_rank_lag1": float(row_exog.get("VIX_rank_252_lag1", 0.5)),
            "SP500_shock_lag1": float(row_exog.get("SP500_shock_lag1", 0.0)),
            "VIX_shock_lag1": float(row_exog.get("VIX_shock_lag1", 0.0)),
            "carry_spread_rank_lag1": float(row_exog.get("carry_spread_rank_252_lag1", 0.5))
        }
        X_row_res = pd.DataFrame([feats_res]).reindex(columns=res_cols, fill_value=0.5)
        ret_shock = float(res_model.predict(X_row_res)[0])
        
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
    
    # Prepare combined dataset with OOD-safe rank and shock features
    combined_raw = pd.concat([train, test_exog], ignore_index=True)
    combined_processed = build_ood_safe_features(combined_raw)
    combined = ade.make_causal_exog(combined_processed)
    
    # Fit Trend Model
    X_trend, y_trend = build_trend_table(train, combined, selected_lags)
    trend_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    trend_model.fit(X_trend, y_trend)
    
    # Fit Residual Model with OOD-safe Features
    X_res, y_res = build_residual_table_ood(train, combined, trend_model, X_trend, y_trend)
    
    # Regulating Ridge alpha to keep residuals robust
    res_alphas = [1.0, 5.0, 10.0, 50.0, 100.0]
    best_rmse = float("inf")
    best_alpha = None
    best_preds = None
    
    for alpha in res_alphas:
        res_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha))
        ]).fit(X_res, y_res)
        
        preds = recursive_forecast(train, test_exog, combined, trend_model, res_model, selected_lags, vix_fac=1.10, spread_fac=1.06)
        score = rmse(y_true, preds)
        print(f"Residual Ridge Alpha: {alpha} -> OOS RMSE: {score:.4f}")
        if score < best_rmse:
            best_rmse = score
            best_alpha = alpha
            best_preds = preds
            
    print(f"\nBest OOD-Safe Residual Model RMSE: {best_rmse:.4f} (Alpha={best_alpha})")
    
    # Save outputs
    sub_df = pd.DataFrame({
        "Date": test_actual[DATE_COL],
        "USDIDR": best_preds
    })
    sub_df.to_csv("submission.csv", index=False)
    print("OOD-Safe Predictions saved to submission.csv successfully!")

if __name__ == "__main__":
    main()
