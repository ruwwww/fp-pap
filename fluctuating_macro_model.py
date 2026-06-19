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


def build_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes purely stationary difference and shock features.
    NO level features are kept to prevent OOD drift propagation.
    """
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    
    # 1. First differences (returns) of exogenous variables - these are stationary
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        out[f"{col}_diff1"] = out[col].diff().fillna(0.0)
        out[f"{col}_ret1"] = np.log(out[col]).diff().fillna(0.0)
        
    # Rate changes (discrete policy shocks)
    out["bi_rate_change"] = out["BI_rate"].diff().fillna(0.0)
    out["us_rate_change"] = out["US_rate"].diff().fillna(0.0)
    
    # 2. Return shocks (volatility-scaled)
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        ret = out[f"{col}_ret1"]
        vol = ret.rolling(20, min_periods=5).std().replace(0.0, np.nan)
        out[f"{col}_shock"] = (ret / vol).fillna(0.0)
        
    # Create lag 1 of all stationary features
    stat_cols = [c for c in out.columns if "_diff1" in c or "_ret1" in c or "change" in c or "_shock" in c]
    for col in stat_cols:
        out[f"{col}_lag1"] = out[col].shift(1).fillna(0.0)
        
    return out


def build_row_features(
    combined_row: pd.Series,
    level_hist: list[float],
    diff_hist: list[float],
    selected_lags: list[int],
    include_exog: bool
) -> dict[str, float]:
    feats = ade.build_row_features(combined_row, level_hist, diff_hist, selected_lags, [], "trend")
    
    if include_exog:
        # Stationary exogenous features (Lag 1)
        feats["SP500_ret"] = float(combined_row.get("SP500_ret1_lag1", 0.0))
        feats["GOLD_ret"] = float(combined_row.get("GOLD_ret1_lag1", 0.0))
        feats["OIL_ret"] = float(combined_row.get("OIL_ret1_lag1", 0.0))
        feats["IHSG_ret"] = float(combined_row.get("IHSG_ret1_lag1", 0.0))
        feats["VIX_ret"] = float(combined_row.get("VIX_ret1_lag1", 0.0))
        
        feats["bi_rate_change"] = float(combined_row.get("bi_rate_change_lag1", 0.0))
        feats["us_rate_change"] = float(combined_row.get("us_rate_change_lag1", 0.0))
        
        feats["SP500_shock"] = float(combined_row.get("SP500_shock_lag1", 0.0))
        feats["VIX_shock"] = float(combined_row.get("VIX_shock_lag1", 0.0))
        
    return feats


def build_train_table(
    train_exog: pd.DataFrame,
    combined: pd.DataFrame,
    selected_lags: list[int],
    include_exog: bool
) -> tuple[pd.DataFrame, pd.Series]:
    levels = train_exog[TARGET_COL].astype(float).tolist()
    diffs = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
    rows = []
    ys = []
    
    start = max(max(selected_lags, default=1), 252)
    for t in range(start, len(train_exog)):
        feats = build_row_features(combined.iloc[t], levels[:t], diffs[: t - 1], selected_lags, include_exog)
        rows.append(feats)
        ys.append(float(math.log(levels[t] / levels[t - 1])))
        
    X = pd.DataFrame(rows).fillna(0.0)
    y = pd.Series(ys, dtype=float)
    return X, y


def recursive_forecast(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    combined: pd.DataFrame,
    model,
    selected_lags: list[int],
    include_exog: bool
) -> np.ndarray:
    history = train_df[TARGET_COL].astype(float).tolist()
    diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    cols = model.feature_names_in_
    
    preds = []
    for i in range(len(future_df)):
        current_idx = len(train_df) + i
        row_exog_test = combined.iloc[current_idx]
        
        feats_test = build_row_features(row_exog_test, history, diffs, selected_lags, include_exog)
        X_test = pd.DataFrame([feats_test]).reindex(columns=cols, fill_value=0.0)
        
        ret_pred = float(model.predict(X_test)[0])
        
        # Policy shock logic:
        # On days when BI rates change, inject the shock directly to forecast to create fluctuations
        bi_change_lag1 = float(row_exog_test.get("bi_rate_change_lag1", 0.0))
        if include_exog and bi_change_lag1 != 0.0:
            # A hike (positive) strengthens Rupiah -> drops USDIDR -> negative return shock
            # A cut (negative) weakens Rupiah -> hikes USDIDR -> positive return shock
            ret_pred -= 0.005 * bi_change_lag1 # 50 bps change corresponds to 0.25% return impact
            
        next_level = float(history[-1] * math.exp(ret_pred))
        preds.append(next_level)
        
        history.append(next_level)
        diffs.append(next_level - history[-2])
        
    return np.asarray(preds, dtype=float)


def main() -> None:
    train, test_exog, test_actual = load_data()
    y_true = test_actual[TARGET_COL].astype(float).to_numpy(dtype=float)
    
    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]
    
    # Precompute stationary features on the combined dataset
    combined_raw = pd.concat([train, test_exog], ignore_index=True)
    combined_stat = build_stationary_features(combined_raw)
    combined = ade.make_causal_exog(combined_stat)
    
    # 1. Fit Fluctuating Exogenous Model on full train
    X_adv, y_adv = build_train_table(train, combined, selected_lags, include_exog=True)
    adv_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)) # Alpha=1.0 for more sensitivity to fluctuations
    ])
    adv_model.fit(X_adv, y_adv)
    
    # Print coefficients to verify macro variables are learned
    m_coefs = adv_model.named_steps["model"].coef_
    m_feats = adv_model.feature_names_in_
    print("Model Coefficients:")
    for f, c in zip(m_feats, m_coefs):
        if "_" in f:
            print(f"{f}: {c:.6f}")
            
    # Forecast OOS
    adv_preds = recursive_forecast(train, test_exog, combined, adv_model, selected_lags, include_exog=True)
    adv_rmse = rmse(y_true, adv_preds)
    print(f"\nFluctuating Exogenous Model OOS RMSE: {adv_rmse:.4f}")
    
    # 2. Baseline: Pure Trend Model without Exogenous features (standard smooth trend)
    X_base, y_base = build_train_table(train, combined, selected_lags, include_exog=False)
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    base_model.fit(X_base, y_base)
    
    base_preds = recursive_forecast(train, test_exog, combined, base_model, selected_lags, include_exog=False)
    base_rmse = rmse(y_true, base_preds)
    print(f"Static Pure Trend Baseline RMSE: {base_rmse:.4f}")
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(test_actual[DATE_COL], y_true, color="black", label="Actual")
    plt.plot(test_actual[DATE_COL], base_preds, color="blue", alpha=0.7, label=f"Static Smooth Trend (RMSE={base_rmse:.2f})")
    plt.plot(test_actual[DATE_COL], adv_preds, color="red", label=f"Fluctuating Exogenous Model (RMSE={adv_rmse:.2f})")
    plt.title("USDIDR Out-Of-Sample Forecasting: Fluctuating Exogenous vs Smooth Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fluctuating_macro_plot.png", dpi=150)
    plt.close()
    
    # Save outputs
    pred_df = pd.DataFrame({
        "Date": test_actual[DATE_COL],
        "actual": y_true,
        "pure_trend": base_preds,
        "fluctuating_predictions": adv_preds
    })
    pred_df.to_csv("fluctuating_predictions.csv", index=False)
    
    results = pd.DataFrame([
        {"model": "Static Pure Trend Model", "rmse": base_rmse},
        {"model": "Fluctuating Exogenous Model", "rmse": adv_rmse}
    ])
    results.to_csv("fluctuating_results.csv", index=False)


if __name__ == "__main__":
    main()
