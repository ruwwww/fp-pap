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

warnings.filterwarnings("ignore")

ROOT = Path(".")
TRAIN_CSV = ROOT / "data_train.csv"
TEST_CSV = ROOT / "data_test.csv"
ACTUAL_CSV = ROOT / "data_test_actual.csv"
DATE_COL = "Date"
TARGET_COL = "USDIDR"


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test = pd.read_csv(TEST_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    actual = pd.read_csv(ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return train, test, actual


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-computes daily lag/diff features for exogenous variables in a causal manner."""
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    
    # Exogenous variables log returns and diffs
    out["VIX_lag1"] = out["VIX"].shift(1)
    out["SP500_ret_lag1"] = np.log(out["SP500"]).diff().shift(1)
    out["IHSG_ret_lag1"] = np.log(out["IHSG"]).diff().shift(1)
    out["OIL_ret_lag1"] = np.log(out["OIL"]).diff().shift(1)
    out["GOLD_ret_lag1"] = np.log(out["GOLD"]).diff().shift(1)
    
    # Rates and spreads
    out["Spread_lag1"] = (out["BI_rate"] - out["US_rate"]).shift(1)
    out["Spread_change_lag1"] = out["Spread_lag1"].diff()
    
    return out


def get_target_features(levels: list[float], selected_lags: list[int]) -> dict[str, float]:
    feats = {}
    
    # Target log returns
    rets = np.diff(np.log(np.asarray(levels, dtype=float)))
    
    for lag in selected_lags:
        if len(rets) >= lag:
            feats[f"ret_lag{lag}"] = float(rets[-lag])
        else:
            feats[f"ret_lag{lag}"] = 0.0
            
    # Trend deviation (Z-score from 90-day MA)
    if len(levels) >= 90:
        arr = np.asarray(levels[-90:], dtype=float)
        ma90 = float(arr.mean())
        sd90 = float(arr.std(ddof=0))
        if sd90 > 0:
            z = (levels[-1] - ma90) / sd90
            feats["trend_z"] = z
            # Asymmetric threshold pulls
            feats["intervention_pull_1.5"] = float((z - 1.5) * (levels[-1] - ma90)) if z > 1.5 else 0.0
            feats["intervention_pull_2.0"] = float((z - 2.0) * (levels[-1] - ma90)) if z > 2.0 else 0.0
        else:
            feats["trend_z"] = 0.0
            feats["intervention_pull_1.5"] = 0.0
            feats["intervention_pull_2.0"] = 0.0
    else:
        feats["trend_z"] = 0.0
        feats["intervention_pull_1.5"] = 0.0
        feats["intervention_pull_2.0"] = 0.0
        
    return feats


def build_train_table(
    train_exog: pd.DataFrame, 
    selected_lags: list[int],
    use_interactions: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    levels = train_exog[TARGET_COL].astype(float).tolist()
    rows = []
    ys = []
    
    # We need enough history for 252-day trend or lags
    start = 252
    for t in range(start, len(train_exog)):
        row_exog = train_exog.iloc[t]
        feats = get_target_features(levels[:t], selected_lags)
        
        # Add exogenous variables
        feats["VIX_lag1"] = float(row_exog["VIX_lag1"])
        feats["SP500_ret_lag1"] = float(row_exog["SP500_ret_lag1"])
        feats["IHSG_ret_lag1"] = float(row_exog["IHSG_ret_lag1"])
        feats["Spread_lag1"] = float(row_exog["Spread_lag1"])
        
        if use_interactions:
            # Macro-conditioned AR dynamics
            feats["ret_lag1_x_VIX"] = feats["ret_lag1"] * feats["VIX_lag1"]
            feats["ret_lag1_x_Spread"] = feats["ret_lag1"] * feats["Spread_lag1"]
            feats["ret_lag1_x_SP500"] = feats["ret_lag1"] * feats["SP500_ret_lag1"]
            
        rows.append(feats)
        ys.append(float(math.log(levels[t] / levels[t - 1])))
        
    X = pd.DataFrame(rows).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    y = pd.Series(ys, dtype=float)
    return X, y


def recursive_forecast(
    train_df: pd.DataFrame,
    test_exog: pd.DataFrame,
    model,
    selected_lags: list[int],
    use_interactions: bool = True
) -> np.ndarray:
    history = train_df[TARGET_COL].astype(float).tolist()
    cols = model.feature_names_in_
    
    preds = []
    for i in range(len(test_exog)):
        row_exog = test_exog.iloc[i]
        feats = get_target_features(history, selected_lags)
        
        # Add exogenous variables
        feats["VIX_lag1"] = float(row_exog["VIX_lag1"])
        feats["SP500_ret_lag1"] = float(row_exog["SP500_ret_lag1"])
        feats["IHSG_ret_lag1"] = float(row_exog["IHSG_ret_lag1"])
        feats["Spread_lag1"] = float(row_exog["Spread_lag1"])
        
        if use_interactions:
            feats["ret_lag1_x_VIX"] = feats["ret_lag1"] * feats["VIX_lag1"]
            feats["ret_lag1_x_Spread"] = feats["ret_lag1"] * feats["Spread_lag1"]
            feats["ret_lag1_x_SP500"] = feats["ret_lag1"] * feats["SP500_ret_lag1"]
            
        X_row = pd.DataFrame([feats]).reindex(columns=cols, fill_value=0.0)
        ret_pred = float(model.predict(X_row)[0])
        next_level = float(history[-1] * math.exp(ret_pred))
        preds.append(next_level)
        history.append(next_level)
        
    return np.asarray(preds, dtype=float)


def main() -> None:
    train, test, actual = load_data()
    y_true = actual[TARGET_COL].astype(float).to_numpy(dtype=float)
    
    # Precompute lag/diff variables causally
    combined_raw = pd.concat([train, test], ignore_index=True)
    combined_exog = build_base_features(combined_raw)
    
    train_exog = combined_exog.iloc[:len(train)].reset_index(drop=True)
    test_exog = combined_exog.iloc[len(train):].reset_index(drop=True)
    
    # PACF Lags
    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]
    
    # 1. Validation split (honest parameter tuning)
    split_start = pd.Timestamp("2022-01-01")
    split_end = pd.Timestamp("2023-05-31")
    train_fold = train_exog[train_exog[DATE_COL] < split_start].reset_index(drop=True)
    valid_fold = train_exog[(train_exog[DATE_COL] >= split_start) & (train_exog[DATE_COL] <= split_end)].reset_index(drop=True)
    
    # Test different alphas and models
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = None
    best_val_rmse = float("inf")
    best_use_interactions = None
    
    for use_int in [True, False]:
        for alpha in alphas:
            X_tr, y_tr = build_train_table(train_fold, selected_lags, use_interactions=use_int)
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha))
            ])
            model.fit(X_tr, y_tr)
            
            # Predict validation set recursively
            preds = recursive_forecast(train_fold, valid_fold, model, selected_lags, use_interactions=use_int)
            score = rmse(valid_fold[TARGET_COL], preds)
            print(f"Validation [Interactions={use_int}, Alpha={alpha}] -> RMSE: {score:.4f}")
            
            if score < best_val_rmse:
                best_val_rmse = score
                best_alpha = alpha
                best_use_interactions = use_int
                
    print(f"\nBest Validation Settings: Interactions={best_use_interactions}, Alpha={best_alpha} with RMSE: {best_val_rmse:.4f}")
    
    # 2. Fit on full training set using best parameters
    X_full, y_full = build_train_table(train_exog, selected_lags, use_interactions=best_use_interactions)
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=best_alpha))
    ])
    final_model.fit(X_full, y_full)
    
    # 3. Predict out-of-sample recursively
    oos_preds = recursive_forecast(train_exog, test_exog, final_model, selected_lags, use_interactions=best_use_interactions)
    oos_rmse = rmse(y_true, oos_preds)
    print(f"\nFinal Test OOS RMSE: {oos_rmse:.4f}")
    
    # Compare with naive and baseline trend model
    # Baseline trend-only model:
    X_tr_trend, y_tr_trend = build_train_table(train_exog, selected_lags, use_interactions=False)
    # Remove exogenous columns from X_tr_trend
    trend_cols = [c for c in X_tr_trend.columns if "lag" in c or "pull" in c or "z" in c]
    X_tr_trend = X_tr_trend[trend_cols]
    trend_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    trend_model.fit(X_tr_trend, y_tr_trend)
    
    trend_preds = []
    history = train_exog[TARGET_COL].astype(float).tolist()
    for i in range(len(test_exog)):
        feats = get_target_features(history, selected_lags)
        X_row = pd.DataFrame([feats]).reindex(columns=trend_model.feature_names_in_, fill_value=0.0)
        ret_pred = float(trend_model.predict(X_row)[0])
        next_level = float(history[-1] * math.exp(ret_pred))
        trend_preds.append(next_level)
        history.append(next_level)
        
    trend_rmse = rmse(y_true, trend_preds)
    print(f"Baseline Trend-only OOS RMSE: {trend_rmse:.4f}")
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(actual[DATE_COL], y_true, color="black", label="Actual")
    plt.plot(actual[DATE_COL], trend_preds, color="blue", alpha=0.7, label="Pure Trend Model")
    plt.plot(actual[DATE_COL], oos_preds, color="red", label="Advanced Exogenous Interaction Model")
    plt.title("USDIDR Out-Of-Sample Forecasting: Exogenous Interactions vs Pure Trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig("robust_exogenous_plot.png", dpi=150)
    plt.close()
    
    # Save outputs
    pred_df = pd.DataFrame({
        "Date": actual[DATE_COL],
        "actual": y_true,
        "pure_trend": trend_preds,
        "exogenous_interaction": oos_preds
    })
    pred_df.to_csv("robust_exogenous_predictions.csv", index=False)
    
    results = pd.DataFrame([
        {"model": "Pure Trend Model", "rmse": trend_rmse},
        {"model": "Advanced Exogenous Interaction Model", "rmse": oos_rmse}
    ])
    results.to_csv("robust_exogenous_results.csv", index=False)


if __name__ == "__main__":
    main()
