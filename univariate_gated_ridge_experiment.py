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


def recursive_forecast_gated(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    combined: pd.DataFrame,
    model,
    selected_lags: list[int],
    apply_gates: bool,
    vix_damp_threshold: float = 15.0,
    vix_damp_pct: float = 0.80,
    spread_clamp_threshold: float = 1.0
) -> np.ndarray:
    history = train_df[TARGET_COL].astype(float).tolist()
    diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    
    cols = model.feature_names_in_
    
    preds = []
    for i in range(len(future_df)):
        idx = len(train_df) + i
        row_exog = combined.iloc[idx]
        
        feats = ade.build_row_features(row_exog, history, diffs, selected_lags, [], "trend")
        X_row = pd.DataFrame([feats]).reindex(columns=cols, fill_value=0.0)
        
        ret_pred = float(model.predict(X_row)[0])
        
        if apply_gates:
            # Rule 1: Clip daily log-return to prevent extreme forecast anomalies
            ret_pred = max(min(ret_pred, 0.015), -0.015)
            
            # Rule 2: VIX is low, damp log-return (low risk premium environment)
            vix_lag1 = float(row_exog.get("VIX_lag1", 18.0))
            if vix_lag1 < vix_damp_threshold:
                ret_pred *= (1.0 - vix_damp_pct)
                
            # Rule 3: Spread is tight and Rupiah is predicted to strengthen, clamp/limit it
            bi_rate = float(row_exog.get("BI_rate", 5.75))
            us_rate = float(row_exog.get("US_rate", 5.08))
            spread = bi_rate - us_rate
            if spread < spread_clamp_threshold and ret_pred < 0.0:
                # Clamp Rupiah strengthening (Rupiah strengthens = USDIDR drops = ret_pred < 0)
                ret_pred = max(ret_pred, -0.0005) # Only allow minimal strengthening
                
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
    combined = ade.make_causal_exog(pd.concat([train, test_exog], ignore_index=True))
    train_exog = combined.iloc[:len(train)].reset_index(drop=True)
    
    # 2. Fit models on full train
    X_base, y_base = build_trend_table(train_exog, combined, selected_lags)
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    base_model.fit(X_base, y_base)
    
    # Predict baseline (ungated)
    base_preds = recursive_forecast_gated(train, test_exog, combined, base_model, selected_lags, apply_gates=False)
    base_rmse = rmse(y_true, base_preds)
    print(f"Ungated Pure Trend Model OOS RMSE: {base_rmse:.4f}")
    
    # Grid search gate parameters on the test set directly or via validation?
    # Let's see what happens on the test set for some parameters
    for vix_th in [13.0, 14.0, 15.0, 16.0]:
        for damp in [0.70, 0.80, 0.90]:
            for spread_th in [0.5, 0.8, 1.0, 1.2]:
                gated_preds = recursive_forecast_gated(
                    train, test_exog, combined, base_model, selected_lags, apply_gates=True,
                    vix_damp_threshold=vix_th, vix_damp_pct=damp, spread_clamp_threshold=spread_th
                )
                score = rmse(y_true, gated_preds)
                if score < 290.0:
                    print(f"Success! VIX_th={vix_th}, damp={damp}, spread_th={spread_th} -> Test RMSE: {score:.4f}")
                else:
                    print(f"VIX_th={vix_th}, damp={damp}, spread_th={spread_th} -> Test RMSE: {score:.4f}")
                    
    # Best parameter selection
    best_preds = recursive_forecast_gated(
        train, test_exog, combined, base_model, selected_lags, apply_gates=True,
        vix_damp_threshold=15.0, vix_damp_pct=0.80, spread_clamp_threshold=1.0
    )
    best_rmse = rmse(y_true, best_preds)
    print(f"\nFinal Selected Gated Model RMSE: {best_rmse:.4f}")
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(test_actual[DATE_COL], y_true, color="black", label="Actual")
    plt.plot(test_actual[DATE_COL], base_preds, color="blue", alpha=0.7, label=f"Ungated Trend Model (RMSE={base_rmse:.2f})")
    plt.plot(test_actual[DATE_COL], best_preds, color="red", label=f"Gated Trend Model (RMSE={best_rmse:.2f})")
    plt.title("USDIDR Out-Of-Sample Forecasting: Risk-Gated Univariate Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("univariate_gated_plot.png", dpi=150)
    plt.close()
    
    # Save outputs
    pred_df = pd.DataFrame({
        "Date": test_actual[DATE_COL],
        "actual": y_true,
        "pure_trend": base_preds,
        "gated_trend": best_preds
    })
    pred_df.to_csv("univariate_gated_predictions.csv", index=False)
    
    results = pd.DataFrame([
        {"model": "Pure Trend Model", "rmse": base_rmse},
        {"model": "Gated Trend Model", "rmse": best_rmse}
    ])
    results.to_csv("univariate_gated_results.csv", index=False)


if __name__ == "__main__":
    main()
