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

import assumption_driven_experiment as ade

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


def build_intervention_feat(levels: list[float], threshold: float | None) -> float:
    if threshold is None or len(levels) < 90:
        return 0.0
    arr = np.asarray(levels[-90:], dtype=float)
    ma90 = float(arr.mean())
    sd90 = float(arr.std(ddof=0))
    if sd90 <= 0:
        return 0.0
    z = (levels[-1] - ma90) / sd90
    if z <= threshold:
        return 0.0
    return float((z - threshold) * (levels[-1] - ma90))


def build_train_table(train: pd.DataFrame, selected_lags: list[int], threshold: float | None) -> tuple[pd.DataFrame, pd.Series]:
    levels = train[TARGET_COL].astype(float).tolist()
    rows = []
    ys = []
    start = max(max(selected_lags, default=1), 252)
    for t in range(start, len(train)):
        hist = levels[:t]
        row = ade.build_row_features(pd.Series(dtype=float), hist, [hist[i] - hist[i - 1] for i in range(1, len(hist))], selected_lags, [], "trend")
        row["intervention_pull"] = build_intervention_feat(hist, threshold)
        rows.append(row)
        ys.append(float(math.log(levels[t] / levels[t - 1])))
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    y = pd.Series(ys, dtype=float)
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def fit_model(X: pd.DataFrame, y: pd.Series):
    return ade.fit_model(X, y, kind="ridge")


def gated_recursive_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    trend_model,
    threshold_model,
    threshold_val: float,
    vix_gate_threshold: float,
    selected_lags: list[int]
) -> np.ndarray:
    """
    Recursively forecasts the levels of USDIDR.
    At each step t:
        if VIX_lag1 > vix_gate_threshold:
            use threshold_model (with intervention_pull feature calculated using threshold_val)
        else:
            use trend_model (with intervention_pull = 0)
    """
    history = train[TARGET_COL].astype(float).tolist()
    
    # We need the VIX sequence. Let's align VIX to the history and test sequence.
    # Combined VIX: train VIX and test VIX.
    combined_vix = pd.concat([train["VIX"], test["VIX"]], ignore_index=True).astype(float).tolist()
    
    trend_cols = trend_model.feature_names_in_
    threshold_cols = threshold_model.feature_names_in_
    
    preds = []
    for i in range(len(test)):
        # VIX_lag1 is the VIX from the previous day relative to the prediction step
        t_idx = len(train) + i
        vix_lag1 = combined_vix[t_idx - 1]
        
        feats = ade.build_row_features(pd.Series(dtype=float), history, [history[j] - history[j - 1] for j in range(1, len(history))], selected_lags, [], "trend")
        
        if vix_lag1 > vix_gate_threshold:
            # Choose threshold model
            feats["intervention_pull"] = build_intervention_feat(history, threshold_val)
            X_row = pd.DataFrame([feats]).reindex(columns=threshold_cols, fill_value=np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
            ret = float(threshold_model.predict(X_row)[0])
        else:
            # Choose trend model
            feats["intervention_pull"] = 0.0
            X_row = pd.DataFrame([feats]).reindex(columns=trend_cols, fill_value=np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
            ret = float(trend_model.predict(X_row)[0])
            
        next_level = float(history[-1] * math.exp(ret))
        preds.append(next_level)
        history.append(next_level)
        
    return np.asarray(preds, dtype=float)


def main() -> None:
    train, test, actual = load_data()
    y_true = actual[TARGET_COL].astype(float).to_numpy(dtype=float)
    selected_lags = list(range(1, 48))
    threshold_val = 1.50

    # 1. Validation split for selecting the VIX gate threshold
    split_start = pd.Timestamp("2022-01-01")
    split_end = pd.Timestamp("2023-05-31")
    train_fold = train[train[DATE_COL] < split_start].reset_index(drop=True)
    valid_fold = train[(train[DATE_COL] >= split_start) & (train[DATE_COL] <= split_end)].reset_index(drop=True)

    # Fit validation models
    trend_X, trend_y = build_train_table(train_fold, selected_lags, threshold=None)
    trend_model_val = fit_model(trend_X, trend_y)
    
    threshold_X, threshold_y = build_train_table(train_fold, selected_lags, threshold=threshold_val)
    threshold_model_val = fit_model(threshold_X, threshold_y)

    # Grid search VIX threshold on validation fold
    vix_thresholds = np.arange(15.0, 35.1, 0.5)
    best_vix_gate = None
    best_val_rmse = float("inf")
    
    # Check baseline (always trend)
    baseline_val_preds = gated_recursive_forecast(
        train_fold, valid_fold, trend_model_val, threshold_model_val, threshold_val,
        vix_gate_threshold=9999.0, selected_lags=selected_lags
    )
    baseline_val_rmse = rmse(valid_fold[TARGET_COL], baseline_val_preds)
    print(f"Validation baseline (always trend) RMSE: {baseline_val_rmse:.4f}")

    # Check always threshold
    always_val_preds = gated_recursive_forecast(
        train_fold, valid_fold, trend_model_val, threshold_model_val, threshold_val,
        vix_gate_threshold=-9999.0, selected_lags=selected_lags
    )
    always_val_rmse = rmse(valid_fold[TARGET_COL], always_val_preds)
    print(f"Validation always threshold model RMSE: {always_val_rmse:.4f}")

    results_grid = []
    for gate in vix_thresholds:
        preds = gated_recursive_forecast(
            train_fold, valid_fold, trend_model_val, threshold_model_val, threshold_val,
            vix_gate_threshold=float(gate), selected_lags=selected_lags
        )
        score = rmse(valid_fold[TARGET_COL], preds)
        results_grid.append({"vix_gate": float(gate), "rmse": score})
        if score < best_val_rmse:
            best_val_rmse = score
            best_vix_gate = float(gate)
            
    print(f"Best validation VIX gate threshold: {best_vix_gate:.1f} with RMSE: {best_val_rmse:.4f}")

    # 2. Fit final models on full train
    X_trend_full, y_trend_full = build_train_table(train, selected_lags, threshold=None)
    trend_model_full = fit_model(X_trend_full, y_trend_full)
    
    X_thr_full, y_thr_full = build_train_table(train, selected_lags, threshold=threshold_val)
    threshold_model_full = fit_model(X_thr_full, y_thr_full)

    # 3. Evaluate once on OOS test using best selected gate
    test_preds_gated = gated_recursive_forecast(
        train, test, trend_model_full, threshold_model_full, threshold_val,
        vix_gate_threshold=best_vix_gate, selected_lags=selected_lags
    )
    test_rmse_gated = rmse(y_true, test_preds_gated)
    
    # Also evaluate always trend on OOS
    test_preds_trend = gated_recursive_forecast(
        train, test, trend_model_full, threshold_model_full, threshold_val,
        vix_gate_threshold=9999.0, selected_lags=selected_lags
    )
    test_rmse_trend = rmse(y_true, test_preds_trend)

    # Also evaluate always threshold on OOS
    test_preds_thr = gated_recursive_forecast(
        train, test, trend_model_full, threshold_model_full, threshold_val,
        vix_gate_threshold=-9999.0, selected_lags=selected_lags
    )
    test_rmse_thr = rmse(y_true, test_preds_thr)

    print("\nFinal OOS Results on Test Set:")
    print(f"Gated model (VIX gate = {best_vix_gate:.1f}) RMSE: {test_rmse_gated:.4f}")
    print(f"Pure trend model RMSE: {test_rmse_trend:.4f}")
    print(f"Pure threshold model RMSE: {test_rmse_thr:.4f}")

    # Plot predictions
    plt.figure(figsize=(15, 6))
    plt.plot(actual[DATE_COL], y_true, color="black", linewidth=1.5, label="actual")
    plt.plot(actual[DATE_COL], test_preds_trend, linewidth=1.0, label="Pure Trend Model")
    plt.plot(actual[DATE_COL], test_preds_thr, linewidth=1.0, label=f"Pure Threshold Model (thr={threshold_val})")
    plt.plot(actual[DATE_COL], test_preds_gated, linewidth=1.2, color="red", label=f"VIX Gated Model (gate={best_vix_gate:.1f})")
    plt.title("USDIDR Out-Of-Sample Gated Forecast Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("vix_gated_threshold_plot.png", dpi=150)
    plt.close()

    # Save predictions
    pred_df = pd.DataFrame({
        "Date": actual[DATE_COL],
        "actual": y_true,
        "pure_trend": test_preds_trend,
        "pure_threshold": test_preds_thr,
        "gated_predictions": test_preds_gated
    })
    pred_df.to_csv("vix_gated_predictions.csv", index=False)

    # Save results
    results_summary = pd.DataFrame([
        {"model": "Pure Trend Model", "rmse": test_rmse_trend},
        {"model": f"Pure Threshold Model (thr={threshold_val})", "rmse": test_rmse_thr},
        {"model": f"VIX Gated Model (gate={best_vix_gate:.1f})", "rmse": test_rmse_gated}
    ])
    results_summary.to_csv("vix_gated_results.csv", index=False)

    # Write report
    report = [
        "# VIX Gated Regime Switching Experiment",
        "",
        "## Validation Phase",
        f"- Validation Baseline (always trend) RMSE: `{baseline_val_rmse:.4f}`",
        f"- Validation Best Gated RMSE: `{best_val_rmse:.4f}`",
        f"- Selected VIX Gate Threshold: `{best_vix_gate:.1f}`",
        "",
        "## Out-of-Sample Evaluation",
        results_summary.to_markdown(index=False),
        "",
        "## Key Findings",
        f"- Testing the hypothesis that high-VIX environments (VIX_lag1 > {best_vix_gate}) require threshold mean-reversion modeling, while low-VIX environments are trend-dominated.",
        "- Gating selection was done entirely on the training validation fold to prevent leakage.",
    ]
    Path("vix_gated_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
