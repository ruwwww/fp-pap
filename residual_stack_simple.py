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
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

import assumption_driven_experiment as ade
import phase3_lightgbm_experiments as p3

warnings.filterwarnings("ignore")

ROOT = Path(".")
TRAIN_CSV = ROOT / "data_train.csv"
TEST_EXOG_CSV = ROOT / "data_test.csv"
TEST_ACTUAL_CSV = ROOT / "data_test_actual.csv"
DATE_COL = "Date"
TARGET_COL = "USDIDR"

FULL_EXOG = [
    "SP500_diff1",
    "VIX_diff1",
    "GOLD_diff1",
    "IHSG_diff1",
    "OIL_diff1",
    "bi_rate_change",
    "cpi_change",
    "days_since_bi_change",
    "days_since_cpi_release",
    "is_Q2",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def yearly_rmse(dates: pd.Series, actual: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "actual": actual, "pred": pred})
    df["year"] = df["Date"].dt.year
    rows = []
    for year in sorted(df["year"].unique()):
        g = df[df["year"] == year]
        rows.append({"year": int(year), "rmse": rmse(g["actual"], g["pred"]), "n": int(len(g))})
    return pd.DataFrame(rows)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_exog = pd.read_csv(TEST_EXOG_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_actual = pd.read_csv(TEST_ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return train, test_exog, test_actual


def prepare_combined(train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    train_cols = [DATE_COL, TARGET_COL] + [c for c in future.columns if c not in {DATE_COL, TARGET_COL}]
    future_cols = [DATE_COL] + [c for c in future.columns if c not in {DATE_COL, TARGET_COL}]
    return ade.make_causal_exog(pd.concat([train[train_cols], future[future_cols]], ignore_index=True))


def fit_recursive_model(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    feature_mode: str,
    selected_lags: list[int],
    selected_exog: list[str],
    kind: str,
):
    combined = prepare_combined(train_df, future_df)
    levels = train_df[TARGET_COL].astype(float).tolist()
    rows = []
    ys = []
    start = max(max(selected_lags, default=1), 64)
    if feature_mode in {"trend", "full"}:
        start = max(start, 90, 252)
    for t in range(start, len(train_df)):
        row = ade.build_row_features(combined.iloc[t], levels[:t], [levels[i] - levels[i - 1] for i in range(1, t)], selected_lags, selected_exog, feature_mode)
        rows.append(row)
        ys.append(float(math.log(levels[t] / levels[t - 1])))
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    y = pd.Series(ys, dtype=float)
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    model = ade.fit_model(X, y, kind=kind)
    train_pred = model.predict(X)
    history = train_df[TARGET_COL].astype(float).tolist()
    future_preds = []
    for i in range(len(future_df)):
        idx = len(train_df) + i
        row = combined.iloc[idx]
        feats = ade.build_row_features(row, history, [history[j] - history[j - 1] for j in range(1, len(history))], selected_lags, selected_exog, feature_mode)
        X_row = pd.DataFrame([feats]).reindex(columns=X.columns, fill_value=np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
        ret_pred = float(model.predict(X_row)[0])
        next_level = float(history[-1] * math.exp(ret_pred))
        future_preds.append(next_level)
        history.append(next_level)
    return model, X, y, train_pred, np.asarray(future_preds, dtype=float), combined


def build_residual_table(train_df: pd.DataFrame, combined: pd.DataFrame, base_model, base_X: pd.DataFrame, base_y: pd.Series, selected_lags: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    base_ret_pred = base_model.predict(base_X)
    start = len(train_df) - len(base_y)
    levels = train_df[TARGET_COL].astype(float).tolist()
    rows = []
    ys = []
    safe_train = combined.iloc[: len(train_df)].copy()
    for i, t in enumerate(range(start, len(train_df))):
        row = safe_train.iloc[t]
        feats = p3.build_history_features(levels[:t], row, selected_lags, mode="safe")
        rows.append(feats)
        ys.append(float(base_y.iloc[i] - base_ret_pred[i]))
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    y = pd.Series(ys, dtype=float)
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def train_residual_model(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
    split = max(int(len(X) * 0.85), 1)
    model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.01,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X.iloc[:split], y.iloc[:split], eval_set=[(X.iloc[split:], y.iloc[split:])], eval_metric="rmse", callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def recursive_corrected_forecast(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    base_model,
    residual_model,
    selected_lags: list[int],
) -> np.ndarray:
    combined = prepare_combined(train_df, future_df)
    history = train_df[TARGET_COL].astype(float).tolist()
    base_cols = base_model.feature_names_in_
    resid_cols = residual_model.booster_.feature_name()
    out = []
    for i in range(len(future_df)):
        row = combined.iloc[len(train_df) + i]
        base_feats = ade.build_row_features(row, history, [history[j] - history[j - 1] for j in range(1, len(history))], selected_lags, [], "full")
        X_base = pd.DataFrame([base_feats]).reindex(columns=base_cols, fill_value=np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
        base_ret = float(base_model.predict(X_base)[0])
        safe_feats = p3.build_history_features(history, row, selected_lags, mode="safe")
        X_resid = pd.DataFrame([safe_feats]).reindex(columns=resid_cols, fill_value=np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
        resid_ret = float(residual_model.predict(X_resid)[0])
        next_level = float(history[-1] * math.exp(base_ret + resid_ret))
        out.append(next_level)
        history.append(next_level)
    return np.asarray(out, dtype=float)


def fit_blend_weights(preds: dict[str, np.ndarray], actual: np.ndarray) -> tuple[dict[str, float], float]:
    names = list(preds.keys())
    grid = np.linspace(0.0, 1.0, 21)
    best = None
    best_rmse = float("inf")
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1.0:
                continue
            w3 = 1.0 - w1 - w2
            combo = w1 * preds[names[0]] + w2 * preds[names[1]] + w3 * preds[names[2]]
            score = rmse(actual, combo)
            if score < best_rmse:
                best_rmse = score
                best = {names[0]: float(w1), names[1]: float(w2), names[2]: float(w3)}
    return best or {}, best_rmse


def main() -> None:
    train, test_exog, test_actual = load_data()
    y_true = test_actual[TARGET_COL].astype(float).to_numpy(dtype=float)
    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]

    # Validation split for weight tuning.
    split_start = pd.Timestamp("2022-01-01")
    split_end = pd.Timestamp("2023-05-31")
    train_mask = train[DATE_COL] < split_start
    valid_mask = (train[DATE_COL] >= split_start) & (train[DATE_COL] <= split_end)
    train_fold = train.loc[train_mask].reset_index(drop=True)
    valid_fold = train.loc[valid_mask].reset_index(drop=True)

    base_tr_model, base_tr_X, base_tr_y, base_tr_pred, base_valid_pred, _ = fit_recursive_model(train_fold, valid_fold, "full", selected_lags, FULL_EXOG, "elasticnet")
    resid_X, resid_y = build_residual_table(train_fold, prepare_combined(train_fold, valid_fold), base_tr_model, base_tr_X, base_tr_y, selected_lags)
    resid_model = train_residual_model(resid_X, resid_y)
    corrected_valid_pred = recursive_corrected_forecast(train_fold, valid_fold, base_tr_model, resid_model, selected_lags)
    trend_tr_model, trend_tr_X, trend_tr_y, _, trend_valid_pred, _ = fit_recursive_model(train_fold, valid_fold, "trend", selected_lags, [], "ridge")

    weights, oof_rmse = fit_blend_weights(
        {"base": base_valid_pred, "corrected": corrected_valid_pred, "trend": trend_valid_pred},
        valid_fold[TARGET_COL].astype(float).to_numpy(dtype=float),
    )

    # Fit final models on full training set.
    base_model, base_X, base_y, base_train_pred, base_test_pred, _ = fit_recursive_model(train, test_exog, "full", selected_lags, FULL_EXOG, "elasticnet")
    resid_X_full, resid_y_full = build_residual_table(train, prepare_combined(train, test_exog), base_model, base_X, base_y, selected_lags)
    resid_model_full = train_residual_model(resid_X_full, resid_y_full)
    corrected_test_pred = recursive_corrected_forecast(train, test_exog, base_model, resid_model_full, selected_lags)
    trend_model, trend_X, trend_y, trend_train_pred, trend_test_pred, _ = fit_recursive_model(train, test_exog, "trend", selected_lags, [], "ridge")

    ensemble_test_pred = weights["base"] * base_test_pred + weights["corrected"] * corrected_test_pred + weights["trend"] * trend_test_pred

    results = pd.DataFrame([
        {"model": "elasticnet_full", "rmse": rmse(y_true, base_test_pred)},
        {"model": "elasticnet_residual_safe", "rmse": rmse(y_true, corrected_test_pred)},
        {"model": "ar_plus_trend", "rmse": rmse(y_true, trend_test_pred)},
        {"model": "ensemble_final", "rmse": rmse(y_true, ensemble_test_pred)},
    ]).sort_values("rmse").reset_index(drop=True)

    pred_df = pd.DataFrame({
        "Date": test_actual[DATE_COL],
        "actual": y_true,
        "elasticnet_full": base_test_pred,
        "elasticnet_residual_safe": corrected_test_pred,
        "ar_plus_trend": trend_test_pred,
        "ensemble_final": ensemble_test_pred,
    })
    pred_df.to_csv("residual_stack_predictions.csv", index=False)
    results.to_csv("residual_stack_results.csv", index=False)
    pd.DataFrame([weights]).to_csv("residual_stack_weights.csv", index=False)
    yearly = []
    for name in ["elasticnet_full", "elasticnet_residual_safe", "ar_plus_trend", "ensemble_final"]:
        yearly.append(yearly_rmse(test_actual[DATE_COL], y_true, pred_df[name].to_numpy(dtype=float)).assign(model=name))
    pd.concat(yearly, ignore_index=True).to_csv("residual_stack_yearly_rmse.csv", index=False)

    plt.figure(figsize=(15, 6))
    plt.plot(test_actual[DATE_COL], y_true, color="black", linewidth=1.5, label="actual")
    plt.plot(test_actual[DATE_COL], base_test_pred, linewidth=1.0, label="elasticnet_full")
    plt.plot(test_actual[DATE_COL], corrected_test_pred, linewidth=1.0, label="elasticnet_residual_safe")
    plt.plot(test_actual[DATE_COL], ensemble_test_pred, linewidth=1.2, label="ensemble_final")
    plt.legend()
    plt.tight_layout()
    plt.savefig("residual_stack_plot.png", dpi=150)
    plt.close()

    report = [
        "# Residual Stack Ensemble",
        "",
        f"- Validation blend weights: `{weights}`",
        f"- Validation blend RMSE: `{oof_rmse:.4f}`",
        "",
        results.to_markdown(index=False),
        "",
        "## Interpretation",
        "- Base ElasticNet stays the anchor.",
        "- SAFE residual model only corrects leftover structure.",
        "- Final ensemble is convex and tuned on an internal validation window.",
    ]
    Path("residual_stack_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(results.to_string(index=False))
    print(weights)
    print(f"Validation RMSE: {oof_rmse:.4f}")


if __name__ == "__main__":
    main()
