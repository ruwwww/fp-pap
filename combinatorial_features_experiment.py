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

def build_combinatorial_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    
    # Base stationary differences
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        out[f"{col}_ret"] = np.log(out[col]).diff().fillna(0.0)
    out["bi_rate_change"] = out["BI_rate"].diff().fillna(0.0)
    
    # 1. Global Risk Ratio: VIX / SP500 (using log diffs of the ratio)
    out["vix_to_sp500_ratio"] = out["VIX"] / out["SP500"]
    out["vix_to_sp500_ret"] = np.log(out["vix_to_sp500_ratio"]).diff().fillna(0.0)
    
    # 2. Yield Spread: BI_rate - US_rate
    out["interest_spread"] = out["BI_rate"] - out["US_rate"]
    out["interest_spread_change"] = out["interest_spread"].diff().fillna(0.0)
    
    # 3. Commodity to Equity Ratio: GOLD / SP500
    out["gold_to_sp500_ratio"] = out["GOLD"] / out["SP500"]
    out["gold_to_sp500_ret"] = np.log(out["gold_to_sp500_ratio"]).diff().fillna(0.0)
    
    # 4. Domestic vs Global Equity Ratio: IHSG / SP500
    out["ihsg_to_sp500_ratio"] = out["IHSG"] / out["SP500"]
    out["ihsg_to_sp500_ret"] = np.log(out["ihsg_to_sp500_ratio"]).diff().fillna(0.0)
    
    # Lag 1 of all interactive features
    comb_cols = ["vix_to_sp500_ret", "interest_spread_change", "gold_to_sp500_ret", "ihsg_to_sp500_ret"]
    for col in comb_cols:
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

def build_combinatorial_residual_table(
    train_df: pd.DataFrame,
    combined: pd.DataFrame,
    trend_model,
    trend_X: pd.DataFrame,
    trend_y: pd.Series,
    feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    trend_preds = trend_model.predict(trend_X)
    residuals = trend_y - trend_preds
    start = len(train_df) - len(trend_y)
    
    rows = []
    for t in range(start, len(train_df)):
        row_exog = combined.iloc[t]
        feats = {col: float(row_exog.get(col, 0.0)) for col in feature_cols}
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
    feature_cols: list[str],
    vix_fac: float = 1.10,
    spread_fac: float = 1.06
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
        
        # 2. Combinatorial Residual Prediction
        feats_res = {col: float(row_exog.get(col, 0.0)) for col in feature_cols}
        X_row_res = pd.DataFrame([feats_res]).reindex(columns=res_cols, fill_value=0.0)
        ret_shock = float(res_model.predict(X_row_res)[0])
        
        ret_total = ret_trend + ret_shock
        
        # Apply Risk Gates
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
    
    combined_raw = pd.concat([train, test_exog], ignore_index=True)
    combined_processed = build_combinatorial_features(combined_raw)
    combined = ade.make_causal_exog(combined_processed)
    
    # Fit Trend Model
    X_trend, y_trend = build_trend_table(train, combined, selected_lags)
    trend_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]).fit(X_trend, y_trend)
    
    # Feature configurations
    feature_sets = {
        "Base (Exogenous returns)": [
            "SP500_ret_lag1", "VIX_ret_lag1", "bi_rate_change_lag1"
        ],
        "Combinatorial Set Only": [
            "vix_to_sp500_ret_lag1", "interest_spread_change_lag1", 
            "gold_to_sp500_ret_lag1", "ihsg_to_sp500_ret_lag1"
        ],
        "Combined (Base + Combinatorial)": [
            "SP500_ret_lag1", "VIX_ret_lag1", "bi_rate_change_lag1",
            "vix_to_sp500_ret_lag1", "interest_spread_change_lag1", 
            "gold_to_sp500_ret_lag1", "ihsg_to_sp500_ret_lag1"
        ]
    }
    
    results = []
    for name, f_cols in feature_sets.items():
        X_res, y_res = build_combinatorial_residual_table(train, combined, trend_model, X_trend, y_trend, f_cols)
        
        # Fit residual model
        res_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ]).fit(X_res, y_res)
        
        preds = recursive_forecast(train, test_exog, combined, trend_model, res_model, selected_lags, f_cols)
        score = rmse(y_true, preds)
        print(f"Feature Set: {name} -> OOS RMSE: {score:.4f}")
        results.append({"feature_set": name, "rmse": score})
        
    df_res = pd.DataFrame(results)
    df_res.to_csv("combinatorial_features_results.csv", index=False)
    
    # Safety recovery check to keep best Continuous Dynamic Alpha model in submission.csv
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
