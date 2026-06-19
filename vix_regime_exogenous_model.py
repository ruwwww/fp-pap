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
from sklearn.linear_model import Ridge, ElasticNet
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

FULL_EXOG_COLS = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_exog = pd.read_csv(TEST_EXOG_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_actual = pd.read_csv(TEST_ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return train, test_exog, test_actual


def prepare_combined(train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    train_cols = [DATE_COL, TARGET_COL] + [c for c in future.columns if c not in {DATE_COL, TARGET_COL}]
    future_cols = [DATE_COL] + [c for c in future.columns if c not in {DATE_COL, TARGET_COL}]
    return ade.make_causal_exog(pd.concat([train[train_cols], future[future_cols]], ignore_index=True))


def build_custom_row_features(
    combined_row: pd.Series,
    level_hist: list[float],
    diff_hist: list[float],
    selected_lags: list[int],
    feature_mode: str,
    exog_setup: str
) -> dict[str, float]:
    # Build core trend features
    feats = ade.build_row_features(combined_row, level_hist, diff_hist, selected_lags, [], feature_mode)
    
    # Advanced exogenous features & interactions
    vix = float(combined_row.get("VIX_lag1", np.nan))
    spread = float(combined_row.get("BI_rate_lag1", 0.0) - combined_row.get("US_rate_lag1", 0.0))
    sp500_ret = float(combined_row.get("SP500_diff1", 0.0) / combined_row.get("SP500_lag1", 1.0) if combined_row.get("SP500_lag1", 0.0) != 0 else 0.0)
    
    if exog_setup == "none":
        pass
    elif exog_setup == "basic":
        feats["VIX_lag1"] = vix
        feats["Spread_lag1"] = spread
        feats["SP500_ret_lag1"] = sp500_ret
    elif exog_setup == "interactions":
        feats["VIX_lag1"] = vix
        feats["Spread_lag1"] = spread
        
        # Macro-conditioned trend dynamics
        if "gap_from_trend" in feats:
            feats["gap_x_VIX"] = feats["gap_from_trend"] * vix
            feats["gap_x_Spread"] = feats["gap_from_trend"] * spread
        if "rolling_mean_diff_252" in feats:
            feats["drift_x_VIX"] = feats["rolling_mean_diff_252"] * vix
            feats["drift_x_Spread"] = feats["rolling_mean_diff_252"] * spread
            
    return feats


def build_train_table(
    train_exog: pd.DataFrame,
    combined: pd.DataFrame,
    selected_lags: list[int],
    feature_mode: str,
    exog_setup: str
) -> tuple[pd.DataFrame, pd.Series]:
    levels = train_exog[TARGET_COL].astype(float).tolist()
    diffs = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
    rows = []
    ys = []
    
    start = max(max(selected_lags, default=1), 252)
    for t in range(start, len(train_exog)):
        row_exog = combined.iloc[t]
        feats = build_custom_row_features(row_exog, levels[:t], diffs[: t - 1], selected_lags, feature_mode, exog_setup)
        rows.append(feats)
        ys.append(float(math.log(levels[t] / levels[t - 1])))
        
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.Series(ys, dtype=float)
    return X, y


def recursive_forecast(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    combined: pd.DataFrame,
    model,
    selected_lags: list[int],
    feature_mode: str,
    exog_setup: str
) -> np.ndarray:
    history = train_df[TARGET_COL].astype(float).tolist()
    diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    cols = model.feature_names_in_
    
    preds = []
    for i in range(len(future_df)):
        idx = len(train_df) + i
        row_exog = combined.iloc[idx]
        
        feats = build_custom_row_features(row_exog, history, diffs, selected_lags, feature_mode, exog_setup)
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
    
    # PACF Lags
    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]
    
    combined = prepare_combined(train, test_exog)
    
    # 1. Validation split for selecting best hyperparameters and features
    split_start = pd.Timestamp("2022-01-01")
    split_end = pd.Timestamp("2023-05-31")
    train_fold = train[train[DATE_COL] < split_start].reset_index(drop=True)
    valid_fold = train[(train[DATE_COL] >= split_start) & (train[DATE_COL] <= split_end)].reset_index(drop=True)
    
    # Search grid
    models = ["ridge", "elasticnet"]
    alphas = [0.1, 1.0, 10.0, 50.0, 100.0]
    exog_setups = ["none", "basic", "interactions"]
    
    best_score = float("inf")
    best_config = {}
    
    # We use validation fold target labels for evaluating performance
    y_val = valid_fold[TARGET_COL].astype(float).to_numpy(dtype=float)
    
    print("Starting validation sweep...")
    for model_name in models:
        for alpha in alphas:
            for exog_setup in exog_setups:
                X_tr, y_tr = build_train_table(train_fold, combined, selected_lags, "trend", exog_setup)
                
                if model_name == "ridge":
                    m = Ridge(alpha=alpha)
                else:
                    m = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
                    
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", m)
                ])
                pipeline.fit(X_tr, y_tr)
                
                preds = recursive_forecast(train_fold, valid_fold, combined, pipeline, selected_lags, "trend", exog_setup)
                score = rmse(y_val, preds)
                print(f"Model: {model_name}, Alpha: {alpha}, Exog: {exog_setup} -> Val RMSE: {score:.4f}")
                
                if score < best_score:
                    best_score = score
                    best_config = {"model": model_name, "alpha": alpha, "exog_setup": exog_setup}
                    
    print(f"\nBest Config: {best_config} with Val RMSE: {best_score:.4f}")
    
    # 2. Final fit on full training set using best configuration
    X_full, y_full = build_train_table(train, combined, selected_lags, "trend", best_config["exog_setup"])
    
    if best_config["model"] == "ridge":
        m = Ridge(alpha=best_config["alpha"])
    else:
        m = ElasticNet(alpha=best_config["alpha"], l1_ratio=0.5, max_iter=10000)
        
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", m)
    ])
    final_pipeline.fit(X_full, y_full)
    
    # 3. Forecast OOS recursively
    test_preds = recursive_forecast(train, test_exog, combined, final_pipeline, selected_lags, "trend", best_config["exog_setup"])
    test_rmse = rmse(y_true, test_preds)
    print(f"\nFinal Test OOS RMSE: {test_rmse:.4f}")
    
    # 4. Compare with Baseline Trend-Only model (none)
    X_base, y_base = build_train_table(train, combined, selected_lags, "trend", "none")
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0))
    ])
    base_pipeline.fit(X_base, y_base)
    base_preds = recursive_forecast(train, test_exog, combined, base_pipeline, selected_lags, "trend", "none")
    base_rmse = rmse(y_true, base_preds)
    print(f"Baseline Trend-only (no exog) OOS RMSE: {base_rmse:.4f}")
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(test_actual[DATE_COL], y_true, color="black", label="Actual")
    plt.plot(test_actual[DATE_COL], base_preds, color="blue", alpha=0.7, label=f"Baseline Trend Model (RMSE={base_rmse:.2f})")
    plt.plot(test_actual[DATE_COL], test_preds, color="red", label=f"Best Gated/Exog Model (RMSE={test_rmse:.2f})")
    plt.title("USDIDR Out-Of-Sample Forecasting: Advanced Macro-Conditioned vs Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig("vix_regime_exogenous_plot.png", dpi=150)
    plt.close()
    
    # Save outputs
    pred_df = pd.DataFrame({
        "Date": test_actual[DATE_COL],
        "actual": y_true,
        "pure_trend": base_preds,
        "gated_exogenous": test_preds
    })
    pred_df.to_csv("vix_regime_exogenous_predictions.csv", index=False)
    
    results = pd.DataFrame([
        {"model": "Pure Trend Model", "rmse": base_rmse},
        {"model": "Gated Exogenous Model", "rmse": test_rmse}
    ])
    results.to_csv("vix_regime_exogenous_results.csv", index=False)


if __name__ == "__main__":
    main()
