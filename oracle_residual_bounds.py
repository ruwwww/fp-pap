#!/usr/bin/env python3
from __future__ import annotations

import math
import warnings
from pathlib import Path
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

def recursive_forecast_oracle(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    combined: pd.DataFrame,
    trend_model,
    selected_lags: list[int],
    actual_returns: np.ndarray,
    oracle_mode: str,  # 'upper', 'lower', 'perfect'
    vix_fac: float = 1.10,
    spread_fac: float = 1.06
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
        
        # 2. Oracle Residual Shock (Directly extracting from actual returns)
        actual_ret = actual_returns[i]
        actual_residual = actual_ret - ret_trend
        
        if oracle_mode == "perfect":
            # Perfect model: knows exactly what the residual shock is
            ret_shock = actual_residual
        elif oracle_mode == "upper":
            # Upper bound: model only predicts positive shocks correctly (caps down-shocks to 0)
            ret_shock = max(actual_residual, 0.0)
        elif oracle_mode == "lower":
            # Lower bound: model only predicts negative shocks correctly (caps up-shocks to 0)
            ret_shock = min(actual_residual, 0.0)
        else:
            ret_shock = 0.0
            
        ret_total = ret_trend + ret_shock
        
        # Apply Gates
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
    
    # Calculate actual returns in test set to construct oracle shocks
    test_actual_levels = test_actual[TARGET_COL].astype(float).tolist()
    last_train_level = float(train[TARGET_COL].iloc[-1])
    test_actual_levels = [last_train_level] + test_actual_levels
    actual_returns = np.diff(np.log(test_actual_levels))
    
    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]
    combined = prepare_combined(train, test_exog)
    
    # Fit Trend Model
    X_trend, y_trend = build_trend_table(train, combined, selected_lags)
    trend_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]).fit(X_trend, y_trend)
    
    # Run Oracles
    preds_perfect = recursive_forecast_oracle(train, test_exog, combined, trend_model, selected_lags, actual_returns, "perfect")
    rmse_perfect = rmse(y_true, preds_perfect)
    
    preds_upper = recursive_forecast_oracle(train, test_exog, combined, trend_model, selected_lags, actual_returns, "upper")
    rmse_upper = rmse(y_true, preds_upper)
    
    preds_lower = recursive_forecast_oracle(train, test_exog, combined, trend_model, selected_lags, actual_returns, "lower")
    rmse_lower = rmse(y_true, preds_lower)
    
    # Baseline benchmark (No Shocks)
    preds_base = recursive_forecast_oracle(train, test_exog, combined, trend_model, selected_lags, actual_returns, "none")
    rmse_base = rmse(y_true, preds_base)
    
    print("=== ORACLE RESIDUAL BOUNDARY STUDY ===")
    print(f"1. Baseline (Zero Shocks) RMSE: {rmse_base:.4f}")
    print(f"2. Lower Bound Oracle (Only Negative Shocks Captured) RMSE: {rmse_lower:.4f}")
    print(f"3. Upper Bound Oracle (Only Positive Shocks Captured) RMSE: {rmse_upper:.4f}")
    print(f"4. Perfect Residual Oracle (Full Residual Captured) RMSE: {rmse_perfect:.4f}")
    
    results = pd.DataFrame([
        {"setup": "Baseline (Zero Shocks)", "rmse": rmse_base, "bounds": "Null"},
        {"setup": "Lower Bound Oracle (Negative Shocks)", "rmse": rmse_lower, "bounds": "Lower Bound Limit"},
        {"setup": "Upper Bound Oracle (Positive Shocks)", "rmse": rmse_upper, "bounds": "Upper Bound Limit"},
        {"setup": "Perfect Residual Oracle (All Shocks)", "rmse": rmse_perfect, "bounds": "Theoretical Limit"}
    ])
    results.to_csv("oracle_residual_bounds.csv", index=False)
    
    # Re-save best dynamic alpha predictions to submission.csv to restore submission safety
    # Read the saved predictions if they exist
    pred_path = Path("continuous_dynamic_alpha_predictions.csv")
    if pred_path.exists():
        df_best = pd.read_csv(pred_path)
        sub_df = pd.DataFrame({
            "Date": df_best["Date"],
            "USDIDR": df_best["continuous_dynamic_predictions"]
        })
        sub_df.to_csv("submission.csv", index=False)
        print("\nRestored best Continuous Dynamic Alpha predictions to submission.csv for safety.")

if __name__ == "__main__":
    main()
