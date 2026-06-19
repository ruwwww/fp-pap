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


def recursive_forecast(train: pd.DataFrame, test: pd.DataFrame, model, selected_lags: list[int], threshold: float | None) -> np.ndarray:
    history = train[TARGET_COL].astype(float).tolist()
    cols = model.feature_names_in_
    preds = []
    for _ in range(len(test)):
        feats = ade.build_row_features(pd.Series(dtype=float), history, [history[j] - history[j - 1] for j in range(1, len(history))], selected_lags, [], "trend")
        feats["intervention_pull"] = build_intervention_feat(history, threshold)
        X_row = pd.DataFrame([feats]).reindex(columns=cols, fill_value=np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
        ret = float(model.predict(X_row)[0])
        next_level = float(history[-1] * math.exp(ret))
        preds.append(next_level)
        history.append(next_level)
    return np.asarray(preds, dtype=float)


def yearly_rmse(dates: pd.Series, actual: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "actual": actual, "pred": pred})
    df["year"] = df["Date"].dt.year
    rows = []
    for year in sorted(df["year"].unique()):
        g = df[df["year"] == year]
        rows.append({"year": int(year), "rmse": rmse(g["actual"], g["pred"]), "n": int(len(g))})
    return pd.DataFrame(rows)


def main() -> None:
    train, test, actual = load_data()
    y_true = actual[TARGET_COL].astype(float).to_numpy(dtype=float)
    selected_lags = list(range(1, 48))

    # Validation split for threshold selection.
    split_start = pd.Timestamp("2022-01-01")
    split_end = pd.Timestamp("2023-05-31")
    train_fold = train[train[DATE_COL] < split_start].reset_index(drop=True)
    valid_fold = train[(train[DATE_COL] >= split_start) & (train[DATE_COL] <= split_end)].reset_index(drop=True)

    base_X, base_y = build_train_table(train_fold, selected_lags, threshold=None)
    base_model = fit_model(base_X, base_y)
    base_valid = recursive_forecast(train_fold, valid_fold, base_model, selected_lags, threshold=None)
    base_valid_rmse = rmse(valid_fold[TARGET_COL], base_valid)

    thresholds = np.arange(1.5, 3.01, 0.25)
    best_thr = None
    best_val = float("inf")
    for thr in thresholds:
        X, y = build_train_table(train_fold, selected_lags, threshold=float(thr))
        if len(X) < 100:
            continue
        model = fit_model(X, y)
        pred = recursive_forecast(train_fold, valid_fold, model, selected_lags, threshold=float(thr))
        score = rmse(valid_fold[TARGET_COL], pred)
        if score < best_val:
            best_val = score
            best_thr = float(thr)

    # Final fit on full train.
    X_base_full, y_base_full = build_train_table(train, selected_lags, threshold=None)
    base_full_model = fit_model(X_base_full, y_base_full)
    base_test = recursive_forecast(train, test, base_full_model, selected_lags, threshold=None)

    X_thr_full, y_thr_full = build_train_table(train, selected_lags, threshold=best_thr)
    thr_full_model = fit_model(X_thr_full, y_thr_full)
    thr_test = recursive_forecast(train, test, thr_full_model, selected_lags, threshold=best_thr)

    results = pd.DataFrame([
        {"model": "baseline_trend_ridge", "rmse": rmse(y_true, base_test)},
        {"model": f"threshold_mean_reversion_{best_thr:.2f}", "rmse": rmse(y_true, thr_test)},
    ]).sort_values("rmse").reset_index(drop=True)

    pred_df = pd.DataFrame({"Date": actual[DATE_COL], "actual": y_true, "baseline_trend_ridge": base_test, f"threshold_mean_reversion_{best_thr:.2f}": thr_test})
    pred_df.to_csv("threshold_intervention_predictions.csv", index=False)
    results.to_csv("threshold_intervention_results.csv", index=False)
    yearly = []
    for name in ["baseline_trend_ridge", f"threshold_mean_reversion_{best_thr:.2f}"]:
        yearly.append(yearly_rmse(actual[DATE_COL], y_true, pred_df[name].to_numpy(dtype=float)).assign(model=name))
    pd.concat(yearly, ignore_index=True).to_csv("threshold_intervention_yearly_rmse.csv", index=False)

    plt.figure(figsize=(15, 6))
    plt.plot(actual[DATE_COL], y_true, color="black", linewidth=1.5, label="actual")
    plt.plot(actual[DATE_COL], base_test, linewidth=1.0, label="baseline_trend_ridge")
    plt.plot(actual[DATE_COL], thr_test, linewidth=1.0, label=f"threshold_mean_reversion_{best_thr:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("threshold_intervention_plot.png", dpi=150)
    plt.close()

    report = [
        "# Threshold Intervention Experiment",
        "",
        f"- validation baseline RMSE: `{base_valid_rmse:.4f}`",
        f"- selected threshold: `{best_thr:.2f}`",
        f"- validation best RMSE: `{best_val:.4f}`",
        "",
        results.to_markdown(index=False),
        "",
        "## Interpretation",
        "- intervention_pull is only active in extreme positive deviations.",
        "- threshold is chosen on an internal validation split, not on the test labels.",
        "- if this wins, mean reversion is asymmetric rather than smooth.",
    ]
    Path("threshold_intervention_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(results.to_string(index=False))
    print(f"best_thr={best_thr:.2f}")
    print(f"validation_baseline={base_valid_rmse:.4f} validation_best={best_val:.4f}")


if __name__ == "__main__":
    main()
