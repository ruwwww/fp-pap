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


def build_rolling_zscores(df: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """Computes rolling Z-scores to ensure exogenous features are stationary and scale-invariant."""
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    
    for col in ["VIX", "SP500", "IHSG", "US_rate", "BI_rate"]:
        # Rolling mean and std on historical data only
        r = out[col].rolling(window=window, min_periods=22)
        mean = r.mean()
        std = r.std(ddof=0)
        # Lag by 1 to prevent lookahead bias
        out[f"{col}_z_lag1"] = ((out[col] - mean) / std.replace(0.0, np.nan)).shift(1).fillna(0.0)
        
    # Interest rate spread rolling Z-score
    spread = out["BI_rate"] - out["US_rate"]
    r_spread = spread.rolling(window=window, min_periods=22)
    out["Spread_z_lag1"] = ((spread - r_spread.mean()) / r_spread.std(ddof=0).replace(0.0, np.nan)).shift(1).fillna(0.0)
    
    return out


def build_train_table(
    train_exog: pd.DataFrame,
    combined: pd.DataFrame,
    selected_lags: list[int],
    use_z_exog: bool
) -> tuple[pd.DataFrame, pd.Series]:
    levels = train_exog[TARGET_COL].astype(float).tolist()
    diffs = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
    rows = []
    ys = []
    
    start = max(max(selected_lags, default=1), 252)
    for t in range(start, len(train_exog)):
        row_exog = combined.iloc[t]
        # Base trend features
        feats = ade.build_row_features(row_exog, levels[:t], diffs[: t - 1], selected_lags, [], "trend")
        
        if use_z_exog:
            # Scale-invariant rolling Z-scores
            vix_z = float(row_exog["VIX_z_lag1"])
            spread_z = float(row_exog["Spread_z_lag1"])
            sp500_z = float(row_exog["SP500_z_lag1"])
            
            feats["VIX_z_lag1"] = vix_z
            feats["Spread_z_lag1"] = spread_z
            feats["SP500_z_lag1"] = sp500_z
            
            # Macro-conditioned AR interactions
            feats["ret_lag1_x_VIX_z"] = feats["diff_lag1"] * vix_z
            feats["ret_lag1_x_Spread_z"] = feats["diff_lag1"] * spread_z
            
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
    use_z_exog: bool
) -> np.ndarray:
    history = train_df[TARGET_COL].astype(float).tolist()
    diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    cols = model.feature_names_in_
    
    preds = []
    for i in range(len(future_df)):
        idx = len(train_df) + i
        row_exog = combined.iloc[idx]
        
        feats = ade.build_row_features(row_exog, history, diffs, selected_lags, [], "trend")
        
        if use_z_exog:
            vix_z = float(row_exog["VIX_z_lag1"])
            spread_z = float(row_exog["Spread_z_lag1"])
            sp500_z = float(row_exog["SP500_z_lag1"])
            
            feats["VIX_z_lag1"] = vix_z
            feats["Spread_z_lag1"] = spread_z
            feats["SP500_z_lag1"] = sp500_z
            
            feats["ret_lag1_x_VIX_z"] = feats["diff_lag1"] * vix_z
            feats["ret_lag1_x_Spread_z"] = feats["diff_lag1"] * spread_z
            
        X_row = pd.DataFrame([feats]).reindex(columns=cols, fill_value=0.0)
        ret_pred = float(model.predict(X_row)[0])
        next_level = float(history[-1] * math.exp(ret_pred))
        preds.append(next_level)
        
        history.append(next_level)
        diffs.append(next_level - history[-2])
        
    return np.asarray(preds, dtype=float)


def main() -> None:
    train, test_exog, test_actual = load_data()
    y_true = test_actual[TARGET_COL].astype(float).to_numpy(dtype=float)
    
    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]
    
    # 1. Build combined dataset and compute rolling Z-scores
    combined_raw = pd.concat([train, test_exog], ignore_index=True)
    combined = build_rolling_zscores(combined_raw, window=90)
    
    train_exog = combined.iloc[:len(train)].reset_index(drop=True)
    test_exog_processed = combined.iloc[len(train):].reset_index(drop=True)
    
    # 2. Fit models on full train
    # Baseline Trend Model
    X_base, y_base = build_train_table(train_exog, combined, selected_lags, use_z_exog=False)
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    base_model.fit(X_base, y_base)
    base_preds = recursive_forecast(train, test_exog, combined, base_model, selected_lags, use_z_exog=False)
    base_rmse = rmse(y_true, base_preds)
    print(f"Baseline Trend Model OOS RMSE: {base_rmse:.4f}")
    
    # Advanced Model with Stationary Rolling Z-Scores and Interactions
    X_adv, y_adv = build_train_table(train_exog, combined, selected_lags, use_z_exog=True)
    adv_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0)) # Higher regularization to keep it robust
    ])
    adv_model.fit(X_adv, y_adv)
    adv_preds = recursive_forecast(train, test_exog, combined, adv_model, selected_lags, use_z_exog=True)
    adv_rmse = rmse(y_true, adv_preds)
    print(f"Advanced Z-Score Exog Interaction Model OOS RMSE: {adv_rmse:.4f}")
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(test_actual[DATE_COL], y_true, color="black", label="Actual")
    plt.plot(test_actual[DATE_COL], base_preds, color="blue", alpha=0.7, label=f"Baseline Trend Model (RMSE={base_rmse:.2f})")
    plt.plot(test_actual[DATE_COL], adv_preds, color="red", label=f"Stationary Z-Score Model (RMSE={adv_rmse:.2f})")
    plt.title("USDIDR Out-Of-Sample Forecasting: Stationary Rolling Z-Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rolling_zscore_plot.png", dpi=150)
    plt.close()
    
    # Save outputs
    pred_df = pd.DataFrame({
        "Date": test_actual[DATE_COL],
        "actual": y_true,
        "pure_trend": base_preds,
        "zscore_exogenous": adv_preds
    })
    pred_df.to_csv("rolling_zscore_predictions.csv", index=False)
    
    results = pd.DataFrame([
        {"model": "Pure Trend Model", "rmse": base_rmse},
        {"model": "Stationary Z-Score Model", "rmse": adv_rmse}
    ])
    results.to_csv("rolling_zscore_results.csv", index=False)


if __name__ == "__main__":
    main()
