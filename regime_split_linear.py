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
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test = pd.read_csv(TEST_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    actual = pd.read_csv(ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return train, test, actual


def build_vol(series_levels: list[float]) -> float:
    if len(series_levels) < 22:
        return float("nan")
    rets = np.diff(np.log(np.asarray(series_levels, dtype=float)))
    return float(np.std(rets[-21:], ddof=0))


def build_regime_training_table(train: pd.DataFrame, selected_lags: list[int]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    levels = train[TARGET_COL].astype(float).tolist()
    rows = []
    ys = []
    vols = []
    start = max(max(selected_lags, default=1), 252)
    for t in range(start, len(train)):
        hist = levels[:t]
        row = pd.Series(dtype=float)
        row = ade.build_row_features(row, hist, [hist[i] - hist[i - 1] for i in range(1, len(hist))], selected_lags, [], "trend")
        rows.append(row)
        ys.append(float(math.log(levels[t] / levels[t - 1])))
        vols.append(build_vol(hist))
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    y = pd.Series(ys, dtype=float)
    v = pd.Series(vols, dtype=float)
    valid = X.notna().all(axis=1) & y.notna() & v.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), v.loc[valid].reset_index(drop=True)


def fit_elasticnet(X: pd.DataFrame, y: pd.Series):
    return ade.fit_model(X, y, kind="ridge")


def recursive_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    low_model,
    high_model,
    threshold: float,
    selected_lags: list[int],
) -> np.ndarray:
    history = train[TARGET_COL].astype(float).tolist()
    preds = []
    low_cols = low_model.feature_names_in_
    high_cols = high_model.feature_names_in_
    for i in range(len(test)):
        vol = build_vol(history)
        feats = ade.build_row_features(pd.Series(dtype=float), history, [history[j] - history[j - 1] for j in range(1, len(history))], selected_lags, [], "trend")
        if np.isnan(vol) or vol <= threshold:
            X_row = pd.DataFrame([feats]).reindex(columns=low_cols, fill_value=np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
            ret = float(low_model.predict(X_row)[0])
        else:
            X_row = pd.DataFrame([feats]).reindex(columns=high_cols, fill_value=np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
            ret = float(high_model.predict(X_row)[0])
        next_level = float(history[-1] * math.exp(ret))
        preds.append(next_level)
        history.append(next_level)
    return np.asarray(preds, dtype=float)


def single_model_forecast(train: pd.DataFrame, test: pd.DataFrame, model, selected_lags: list[int]) -> np.ndarray:
    history = train[TARGET_COL].astype(float).tolist()
    preds = []
    cols = model.feature_names_in_
    for i in range(len(test)):
        feats = ade.build_row_features(pd.Series(dtype=float), history, [history[j] - history[j - 1] for j in range(1, len(history))], selected_lags, [], "trend")
        X_row = pd.DataFrame([feats]).reindex(columns=cols, fill_value=np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
        ret = float(model.predict(X_row)[0])
        next_level = float(history[-1] * math.exp(ret))
        preds.append(next_level)
        history.append(next_level)
    return np.asarray(preds, dtype=float)


def main() -> None:
    train, test, actual = load_data()
    y_true = actual[TARGET_COL].astype(float).to_numpy(dtype=float)
    selected_lags = list(range(1, 48))

    X, y, vols = build_regime_training_table(train, selected_lags)
    threshold = float(np.nanmedian(vols))
    low_mask = vols <= threshold
    high_mask = vols > threshold

    low_model = fit_elasticnet(X.loc[low_mask].reset_index(drop=True), y.loc[low_mask].reset_index(drop=True))
    high_model = fit_elasticnet(X.loc[high_mask].reset_index(drop=True), y.loc[high_mask].reset_index(drop=True))

    split_model = fit_elasticnet(X, y)
    split_preds = recursive_forecast(train, test, low_model, high_model, threshold, selected_lags)
    single_preds = single_model_forecast(train, test, split_model, selected_lags)

    results = pd.DataFrame([
        {"model": "single_trend_ridge", "rmse": rmse(y_true, single_preds)},
        {"model": "regime_split_ridge", "rmse": rmse(y_true, split_preds)},
    ]).sort_values("rmse").reset_index(drop=True)

    results.to_csv("regime_split_linear_results.csv", index=False)
    pd.DataFrame({"Date": actual[DATE_COL], "actual": y_true, "single_trend_ridge": single_preds, "regime_split_ridge": split_preds}).to_csv("regime_split_linear_predictions.csv", index=False)
    yearly = []
    for name, pred in {"single_trend_ridge": single_preds, "regime_split_ridge": split_preds}.items():
        df = pd.DataFrame({"Date": actual[DATE_COL], "actual": y_true, "pred": pred})
        df["year"] = pd.to_datetime(df["Date"]).dt.year
        for year in sorted(df["year"].unique()):
            g = df[df["year"] == year]
            yearly.append({"model": name, "year": int(year), "rmse": rmse(g["actual"], g["pred"]), "n": int(len(g))})
    pd.DataFrame(yearly).to_csv("regime_split_linear_yearly_rmse.csv", index=False)

    plt.figure(figsize=(15, 6))
    plt.plot(actual[DATE_COL], y_true, color="black", linewidth=1.5, label="actual")
    plt.plot(actual[DATE_COL], single_preds, linewidth=1.0, label="single_trend_ridge")
    plt.plot(actual[DATE_COL], split_preds, linewidth=1.0, label="regime_split_ridge")
    plt.legend()
    plt.tight_layout()
    plt.savefig("regime_split_linear_plot.png", dpi=150)
    plt.close()

    report = [
        "# Regime Split Linear",
        "",
        f"- volatility threshold (median realized vol 21): `{threshold:.6f}`",
        "",
        results.to_markdown(index=False),
        "",
        "## Interpretation",
        "- Single model is the trend baseline.",
        "- Regime split uses the same features but different coefficients under low/high volatility.",
        "- If split beats single, volatility clustering is not just a feature effect; it changes parameterization.",
    ]
    Path("regime_split_linear_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(results.to_string(index=False))
    print(f"threshold={threshold:.6f}")


if __name__ == "__main__":
    main()
