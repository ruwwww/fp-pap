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


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_exog = pd.read_csv(TEST_EXOG_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_actual = pd.read_csv(TEST_ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return train, test_exog, test_actual


def prepare_combined(train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    train_cols = [DATE_COL, TARGET_COL] + [c for c in future.columns if c not in {DATE_COL, TARGET_COL}]
    future_cols = [DATE_COL] + [c for c in future.columns if c not in {DATE_COL, TARGET_COL}]
    return ade.make_causal_exog(pd.concat([train[train_cols], future[future_cols]], ignore_index=True))


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
    trend_y: pd.Series,
    exog_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    trend_preds = trend_model.predict(trend_X)
    residuals = trend_y - trend_preds
    
    start = len(train_df) - len(trend_y)
    rows = []
    for t in range(start, len(train_df)):
        row_exog = combined.iloc[t]
        feats = {}
        for col in exog_cols:
            feats[col] = float(row_exog.get(col, 0.0))
        rows.append(feats)
        
    X = pd.DataFrame(rows).fillna(0.0)
    y = pd.Series(residuals, dtype=float)
    return X, y


def recursive_forecast(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    combined: pd.DataFrame,
    trend_model,
    residual_model,
    selected_lags: list[int],
    exog_cols: list[str]
) -> np.ndarray:
    history = train_df[TARGET_COL].astype(float).tolist()
    diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    
    trend_features_in = trend_model.feature_names_in_
    resid_features_in = residual_model.feature_names_in_ if residual_model is not None else []
    
    preds = []
    for i in range(len(future_df)):
        idx = len(train_df) + i
        row_exog = combined.iloc[idx]
        
        # 1. Trend model prediction
        feats_trend = ade.build_row_features(row_exog, history, diffs, selected_lags, [], "trend")
        X_trend = pd.DataFrame([feats_trend]).reindex(columns=trend_features_in, fill_value=0.0)
        ret_trend = float(trend_model.predict(X_trend)[0])
        
        # 2. Residual model prediction
        if residual_model is not None:
            feats_resid = {}
            for col in exog_cols:
                feats_resid[col] = float(row_exog.get(col, 0.0))
            X_resid = pd.DataFrame([feats_resid]).reindex(columns=resid_features_in, fill_value=0.0)
            ret_resid = float(residual_model.predict(X_resid)[0])
        else:
            ret_resid = 0.0
            
        next_level = float(history[-1] * math.exp(ret_trend + ret_resid))
        preds.append(next_level)
        
        history.append(next_level)
        diffs.append(next_level - history[-2])
        
    return np.asarray(preds, dtype=float)


def main() -> None:
    train, test_exog, test_actual = load_data()
    y_true = test_actual[TARGET_COL].astype(float).to_numpy(dtype=float)
    
    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]
    combined = prepare_combined(train, test_exog)
    
    # 1. Validation split for selecting residual model hyperparameters
    split_start = pd.Timestamp("2022-01-01")
    split_end = pd.Timestamp("2023-05-31")
    train_fold = train[train[DATE_COL] < split_start].reset_index(drop=True)
    valid_fold = train[(train[DATE_COL] >= split_start) & (train[DATE_COL] <= split_end)].reset_index(drop=True)
    
    # Fit validation trend model
    trend_X_val, trend_y_val = build_trend_table(train_fold, combined, selected_lags)
    trend_model_val = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    trend_model_val.fit(trend_X_val, trend_y_val)
    
    # Baseline validation OOS predictions (no residual correction)
    base_val_preds = recursive_forecast(train_fold, valid_fold, combined, trend_model_val, None, selected_lags, [])
    base_val_rmse = rmse(valid_fold[TARGET_COL], base_val_preds)
    print(f"Validation Trend-only RMSE: {base_val_rmse:.4f}")
    
    # Try different exogenous subsets and models for residual corrections
    exog_options = [
        ["VIX_lag1"],
        ["VIX_lag1", "SP500_diff1"],
        ["VIX_lag1", "SP500_diff1", "IHSG_diff1"],
        ["VIX_lag1", "bi_rate_change", "cpi_change"]
    ]
    
    best_val_score = base_val_rmse
    best_exog = None
    best_alpha = None
    
    for exog_cols in exog_options:
        res_X_val, res_y_val = build_residual_table(train_fold, combined, trend_model_val, trend_X_val, trend_y_val, exog_cols)
        for alpha in [1.0, 10.0, 100.0, 1000.0, 5000.0]:
            res_model_val = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha))
            ])
            res_model_val.fit(res_X_val, res_y_val)
            
            val_preds = recursive_forecast(train_fold, valid_fold, combined, trend_model_val, res_model_val, selected_lags, exog_cols)
            score = rmse(valid_fold[TARGET_COL], val_preds)
            print(f"Exog: {exog_cols}, Alpha: {alpha} -> Val RMSE: {score:.4f}")
            
            if score < best_val_score:
                best_val_score = score
                best_exog = exog_cols
                best_alpha = alpha
                
    print(f"\nBest Validation Residual Correction Settings: Exog={best_exog}, Alpha={best_alpha} with RMSE: {best_val_score:.4f}")
    
    # 2. Fit final models on full train
    trend_X_full, trend_y_full = build_trend_table(train, combined, selected_lags)
    trend_model_full = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    trend_model_full.fit(trend_X_full, trend_y_full)
    
    # Fit baseline (no correction) on test OOS
    final_base_preds = recursive_forecast(train, test_exog, combined, trend_model_full, None, selected_lags, [])
    final_base_rmse = rmse(y_true, final_base_preds)
    
    if best_exog is not None:
        res_X_full, res_y_full = build_residual_table(train, combined, trend_model_full, trend_X_full, trend_y_full, best_exog)
        res_model_full = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=best_alpha))
        ])
        res_model_full.fit(res_X_full, res_y_full)
        
        final_corrected_preds = recursive_forecast(train, test_exog, combined, trend_model_full, res_model_full, selected_lags, best_exog)
        final_corrected_rmse = rmse(y_true, final_corrected_preds)
    else:
        final_corrected_preds = final_base_preds
        final_corrected_rmse = final_base_rmse
        
    print("\nFinal Test OOS Results:")
    print(f"Pure Trend Model (no correction) RMSE: {final_base_rmse:.4f}")
    print(f"Corrected Model (Exog={best_exog}, Alpha={best_alpha}) RMSE: {final_corrected_rmse:.4f}")
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(test_actual[DATE_COL], y_true, color="black", label="Actual")
    plt.plot(test_actual[DATE_COL], final_base_preds, color="blue", alpha=0.7, label=f"Pure Trend Model (RMSE={final_base_rmse:.2f})")
    plt.plot(test_actual[DATE_COL], final_corrected_preds, color="red", label=f"Residual Corrected Model (RMSE={final_corrected_rmse:.2f})")
    plt.title("USDIDR Out-Of-Sample Forecasting: Residual Correction vs Trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig("residual_exogenous_plot.png", dpi=150)
    plt.close()
    
    # Save outputs
    pred_df = pd.DataFrame({
        "Date": test_actual[DATE_COL],
        "actual": y_true,
        "pure_trend": final_base_preds,
        "corrected_trend": final_corrected_preds
    })
    pred_df.to_csv("residual_exogenous_predictions.csv", index=False)
    
    results = pd.DataFrame([
        {"model": "Pure Trend Model", "rmse": final_base_rmse},
        {"model": f"Residual Corrected Model (Exog={best_exog}, Alpha={best_alpha})", "rmse": final_corrected_rmse}
    ])
    results.to_csv("residual_exogenous_results.csv", index=False)


if __name__ == "__main__":
    main()
