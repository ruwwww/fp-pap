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


def build_bounded_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    
    # 1. Bounded Percentile Ranks (0.0 - 1.0)
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        out[f"{col}_rank_252"] = out[col].rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
        
    # 2. Return Shock Magnitudes (z-score of return relative to 20-day standard deviation)
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        ret = out[col].pct_change()
        vol = ret.rolling(20, min_periods=5).std()
        out[f"{col}_shock"] = (ret / vol.replace(0.0, np.nan)).fillna(0.0)
        
    # 3. Carry Interest Spread and its rolling rank
    out["carry_spread"] = out["BI_rate"] - out["US_rate"]
    out["carry_spread_rank_252"] = out["carry_spread"].rolling(252, min_periods=60).rank(pct=True).fillna(0.5)
    
    # Create lag 1 of all exogenous features to prevent lookahead leakage
    exog_cols = [c for c in out.columns if "_rank_252" in c or "_shock" in c]
    for col in exog_cols:
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
        # Add OOD-safe bounded exogenous variables and shock indicators (Lag 1)
        feats["SP500_rank"] = float(combined_row.get("SP500_rank_252_lag1", 0.5))
        feats["GOLD_rank"] = float(combined_row.get("GOLD_rank_252_lag1", 0.5))
        feats["OIL_rank"] = float(combined_row.get("OIL_rank_252_lag1", 0.5))
        feats["IHSG_rank"] = float(combined_row.get("IHSG_rank_252_lag1", 0.5))
        feats["VIX_rank"] = float(combined_row.get("VIX_rank_252_lag1", 0.5))
        
        feats["SP500_shock"] = float(combined_row.get("SP500_shock_lag1", 0.0))
        feats["GOLD_shock"] = float(combined_row.get("GOLD_shock_lag1", 0.0))
        feats["OIL_shock"] = float(combined_row.get("OIL_shock_lag1", 0.0))
        feats["IHSG_shock"] = float(combined_row.get("IHSG_shock_lag1", 0.0))
        
        feats["carry_spread_rank"] = float(combined_row.get("carry_spread_rank_252_lag1", 0.5))
        
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
    include_exog: bool,
    use_vix_gate: bool = True
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
        
        if use_vix_gate:
            # Dynamic macro gate: VIX rank is in the top 20% panic zone
            vix_rank_lag1 = float(row_exog_test.get("VIX_rank_252_lag1", 0.5))
            if ret_pred > 0 and vix_rank_lag1 > 0.8:
                ret_pred *= 1.10
                
        next_level = float(history[-1] * math.exp(ret_pred))
        preds.append(next_level)
        
        history.append(next_level)
        diffs.append(next_level - history[-2])
        
    return np.asarray(preds, dtype=float)


def main() -> None:
    train, test_exog, test_actual = load_data()
    y_true = test_actual[TARGET_COL].astype(float).to_numpy(dtype=float)
    
    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]
    
    # Precompute bounded features on the combined dataset
    combined_raw = pd.concat([train, test_exog], ignore_index=True)
    combined_bounded = build_bounded_features(combined_raw)
    combined = ade.make_causal_exog(combined_bounded)
    
    # 1. Fit Bounded Exogenous Model on full train
    X_adv, y_adv = build_train_table(train, combined, selected_lags, include_exog=True)
    adv_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0))
    ])
    adv_model.fit(X_adv, y_adv)
    
    # Forecast OOS
    adv_preds = recursive_forecast(train, test_exog, combined, adv_model, selected_lags, include_exog=True, use_vix_gate=True)
    adv_rmse = rmse(y_true, adv_preds)
    print(f"Bounded Exogenous Model OOS RMSE: {adv_rmse:.4f}")
    
    # 2. Baseline: Pure Trend Model without Exogenous features
    X_base, y_base = build_train_table(train, combined, selected_lags, include_exog=False)
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    base_model.fit(X_base, y_base)
    
    base_preds = recursive_forecast(train, test_exog, combined, base_model, selected_lags, include_exog=False, use_vix_gate=False)
    base_rmse = rmse(y_true, base_preds)
    print(f"Static Pure Trend Baseline RMSE: {base_rmse:.4f}")
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(test_actual[DATE_COL], y_true, color="black", label="Actual")
    plt.plot(test_actual[DATE_COL], base_preds, color="blue", alpha=0.7, label=f"Static Pure Trend Model (RMSE={base_rmse:.2f})")
    plt.plot(test_actual[DATE_COL], adv_preds, color="red", label=f"Bounded Exogenous Model (RMSE={adv_rmse:.2f})")
    plt.title("USDIDR Out-Of-Sample Forecasting: Bounded Ranks vs Static Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rolling_ridge_gate_plot.png", dpi=150)
    plt.close()
    
    # Save outputs
    pred_df = pd.DataFrame({
        "Date": test_actual[DATE_COL],
        "actual": y_true,
        "pure_trend": base_preds,
        "rolling_ridge_predictions": adv_preds
    })
    pred_df.to_csv("rolling_ridge_predictions.csv", index=False)
    
    results = pd.DataFrame([
        {"model": "Static Pure Trend Model", "rmse": base_rmse},
        {"model": "Bounded Exogenous Model", "rmse": adv_rmse}
    ])
    results.to_csv("rolling_ridge_results.csv", index=False)


if __name__ == "__main__":
    main()
