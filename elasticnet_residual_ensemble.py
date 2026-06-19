#!/usr/bin/env python3
from __future__ import annotations

import itertools
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import wilcoxon
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


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_exog = pd.read_csv(TEST_EXOG_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_actual = pd.read_csv(TEST_ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return train, test_exog, test_actual


def yearly_rmse(dates: pd.Series, actual: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "actual": actual, "pred": pred})
    df["year"] = df["Date"].dt.year
    rows = []
    for year in sorted(df["year"].unique()):
        g = df[df["year"] == year]
        rows.append({"year": int(year), "rmse": rmse(g["actual"], g["pred"]), "n": int(len(g))})
    return pd.DataFrame(rows)


def build_residual_train_table(
    train: pd.DataFrame,
    base_model,
    base_X: pd.DataFrame,
    base_y: pd.Series,
    selected_lags: list[int],
) -> tuple[pd.DataFrame, pd.Series]:
    levels = train[TARGET_COL].astype(float).tolist()
    base_ret_pred = base_model.predict(base_X)
    start = len(train) - len(base_y)
    rows = []
    ys = []
    safe_train = p3.add_causal_state_features(train.copy())
    for idx, t in enumerate(range(start, len(train))):
        current_row = safe_train.iloc[t]
        feats = p3.build_history_features(levels[:t], current_row, selected_lags, mode="safe")
        rows.append(feats)
        ys.append(float(base_y.iloc[idx] - base_ret_pred[idx]))
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    y = pd.Series(ys, dtype=float)
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def train_residual_model(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
    split = max(int(len(X) * 0.85), 1)
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X.iloc[:split],
        y.iloc[:split],
        eval_set=[(X.iloc[split:], y.iloc[split:])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def recursive_base_forecast(
    model,
    train: pd.DataFrame,
    test_exog: pd.DataFrame,
    selected_lags: list[int],
    feature_mode: str,
) -> np.ndarray:
    levels = train[TARGET_COL].astype(float).tolist()
    combined = pd.concat([train[[DATE_COL] + [c for c in test_exog.columns if c != DATE_COL]], test_exog], ignore_index=True)
    combined = ade.make_causal_exog(combined)
    out = []
    for i in range(len(test_exog)):
        row = combined.iloc[len(train) + i]
        feats = ade.build_row_features(row, levels, [levels[j] - levels[j - 1] for j in range(1, len(levels))], selected_lags, [], feature_mode)
        X_row = pd.DataFrame([feats]).reindex(columns=model.feature_names_in_, fill_value=np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
        pred_ret = float(model.predict(X_row)[0])
        next_level = float(levels[-1] * math.exp(pred_ret))
        out.append(next_level)
        levels.append(next_level)
    return np.asarray(out, dtype=float)


def recursive_corrected_forecast(
    base_model,
    residual_model,
    train: pd.DataFrame,
    test_exog: pd.DataFrame,
    selected_lags: list[int],
) -> np.ndarray:
    levels = train[TARGET_COL].astype(float).tolist()
    combined = pd.concat([train[[DATE_COL] + [c for c in test_exog.columns if c != DATE_COL]], test_exog], ignore_index=True)
    combined = p3.add_causal_state_features(combined)
    base_cols = base_model.feature_names_in_
    resid_cols = residual_model.booster_.feature_name()
    out = []
    for i in range(len(test_exog)):
        row = combined.iloc[len(train) + i]
        diffs = [levels[j] - levels[j - 1] for j in range(1, len(levels))]
        base_feats = ade.build_row_features(row, levels, diffs, selected_lags, [], "full")
        X_base = pd.DataFrame([base_feats]).reindex(columns=base_cols, fill_value=np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
        base_ret = float(base_model.predict(X_base)[0])
        safe_feats = p3.build_history_features(levels, row, selected_lags, mode="safe")
        X_resid = pd.DataFrame([safe_feats]).reindex(columns=resid_cols, fill_value=np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
        resid_ret = float(residual_model.predict(X_resid)[0])
        next_level = float(levels[-1] * math.exp(base_ret + resid_ret))
        out.append(next_level)
        levels.append(next_level)
    return np.asarray(out, dtype=float)


def fold_masks(df: pd.DataFrame, valid_start: str, valid_end: str) -> tuple[pd.Series, pd.Series]:
    valid_start = pd.Timestamp(valid_start)
    valid_end = pd.Timestamp(valid_end)
    valid = (df[DATE_COL] >= valid_start) & (df[DATE_COL] <= valid_end)
    train = df[DATE_COL] < valid_start
    return train, valid


def fit_blend_weights(preds: dict[str, np.ndarray], actual: np.ndarray) -> tuple[dict[str, float], float]:
    names = list(preds.keys())
    grid = np.linspace(0.0, 1.0, 21)
    best = None
    best_rmse = float("inf")
    if len(names) == 2:
        for w in grid:
            combo = w * preds[names[0]] + (1.0 - w) * preds[names[1]]
            score = rmse(actual, combo)
            if score < best_rmse:
                best_rmse = score
                best = {names[0]: float(w), names[1]: float(1.0 - w)}
    elif len(names) == 3:
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
    else:
        raise ValueError("expected 2 or 3 model preds")
    return best or {}, best_rmse


def main() -> None:
    train, test_exog, test_actual = load_data()
    y_true = test_actual[TARGET_COL].astype(float).to_numpy(dtype=float)

    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]

    # Full ElasticNet baseline.
    base_train_X, base_train_y, _ = ade.build_train_table(train.copy(), test_exog.copy(), target_mode="return")
    base_model = ade.fit_model(base_train_X, base_train_y, kind="elasticnet")
    base_test_pred = ade.recursive_forecast(base_model, base_train_X, base_train_y, target_mode="return")

    # Safe residual model on top of ElasticNet.
    resid_X, resid_y = build_residual_train_table(train, base_model, base_train_X, base_train_y, selected_lags)
    residual_model = train_residual_model(resid_X, resid_y)
    corrected_test_pred = recursive_corrected_forecast(base_model, residual_model, train, test_exog, selected_lags)

    # AR + trend auxiliary model.
    trend_train_X, trend_train_y, _ = ade.build_train_table(train.copy(), test_exog.copy(), target_mode="return")
    trend_model = ade.fit_model(trend_train_X, trend_train_y, kind="ridge")
    trend_test_pred = ade.recursive_forecast(trend_model, trend_train_X, trend_train_y, target_mode="return")

    # OOF blend weights from yearly folds.
    fold_specs = [
        ("2019-01-01", "2019-12-31"),
        ("2020-01-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-05-31"),
    ]
    oof_rows = []
    for valid_start, valid_end in fold_specs:
        train_mask, valid_mask = fold_masks(train, valid_start, valid_end)
        fold_train = train.loc[train_mask].reset_index(drop=True)
        fold_valid = train.loc[valid_mask].reset_index(drop=True)
        if len(fold_train) < 500 or len(fold_valid) < 30:
            continue
        fold_base_X, fold_base_y, _ = ade.build_train_table(fold_train.copy(), fold_valid.copy(), target_mode="return")
        fold_base_model = ade.fit_model(fold_base_X, fold_base_y, kind="elasticnet")
        fold_base_pred = ade.recursive_forecast(fold_base_model, fold_base_X, fold_base_y, target_mode="return")
        fold_resid_X, fold_resid_y = build_residual_train_table(fold_train, fold_base_model, fold_base_X, fold_base_y, selected_lags)
        fold_residual_model = train_residual_model(fold_resid_X, fold_resid_y)
        fold_corrected_pred = recursive_corrected_forecast(fold_base_model, fold_residual_model, fold_train, fold_valid, selected_lags)
        fold_trend_X, fold_trend_y, _ = ade.build_train_table(fold_train.copy(), fold_valid.copy(), target_mode="return")
        fold_trend_model = ade.fit_model(fold_trend_X, fold_trend_y, kind="ridge")
        fold_trend_pred = ade.recursive_forecast(fold_trend_model, fold_trend_X, fold_trend_y, target_mode="return")
        actual = fold_valid[TARGET_COL].astype(float).to_numpy(dtype=float)
        oof_rows.append(pd.DataFrame({"actual": actual, "base": fold_base_pred, "corrected": fold_corrected_pred, "trend": fold_trend_pred}))

    oof = pd.concat(oof_rows, ignore_index=True)
    weights, oof_rmse = fit_blend_weights({"base": oof["base"].to_numpy(), "corrected": oof["corrected"].to_numpy(), "trend": oof["trend"].to_numpy()}, oof["actual"].to_numpy())
    final_pred = weights["base"] * base_test_pred + weights["corrected"] * corrected_test_pred + weights["trend"] * trend_test_pred

    # Comparison table.
    candidates = pd.DataFrame([
        {"model": "elasticnet_full", "rmse": rmse(y_true, base_test_pred)},
        {"model": "elasticnet_residual_safe", "rmse": rmse(y_true, corrected_test_pred)},
        {"model": "ar_plus_trend", "rmse": rmse(y_true, trend_test_pred)},
        {"model": "ensemble_final", "rmse": rmse(y_true, final_pred)},
    ]).sort_values("rmse").reset_index(drop=True)

    y_year = yearly_rmse(test_actual[DATE_COL], y_true, final_pred)
    y_year.to_csv("elasticnet_residual_ensemble_yearly_rmse.csv", index=False)
    candidates.to_csv("elasticnet_residual_ensemble_results.csv", index=False)

    pred_df = pd.DataFrame({
        "Date": test_actual[DATE_COL],
        "actual": y_true,
        "elasticnet_full": base_test_pred,
        "elasticnet_residual_safe": corrected_test_pred,
        "ar_plus_trend": trend_test_pred,
        "ensemble_final": final_pred,
    })
    pred_df.to_csv("elasticnet_residual_ensemble_predictions.csv", index=False)

    plt.figure(figsize=(15, 6))
    plt.plot(test_actual[DATE_COL], y_true, color="black", linewidth=1.5, label="actual")
    plt.plot(test_actual[DATE_COL], base_test_pred, linewidth=1.0, label="elasticnet_full")
    plt.plot(test_actual[DATE_COL], corrected_test_pred, linewidth=1.0, label="elasticnet_residual_safe")
    plt.plot(test_actual[DATE_COL], final_pred, linewidth=1.2, label="ensemble_final")
    plt.legend()
    plt.tight_layout()
    plt.savefig("elasticnet_residual_ensemble_plot.png", dpi=150)
    plt.close()

    base_rmse = float(rmse(y_true, base_test_pred))
    ensemble_rmse = float(rmse(y_true, final_pred))
    improvement = (base_rmse - ensemble_rmse) / base_rmse * 100.0
    w_stat, w_p = wilcoxon(base_test_pred - y_true, final_pred - y_true, zero_method="wilcox")

    report = [
        "# ElasticNet Residual Ensemble",
        "",
        f"- Blend weights: `{weights}`",
        f"- OOF blend RMSE: `{oof_rmse:.4f}`",
        "",
        candidates.to_markdown(index=False),
        "",
        "## Per-Year",
        y_year.to_markdown(index=False),
        "",
        f"- Base ElasticNet RMSE: `{base_rmse:.4f}`",
        f"- Ensemble RMSE: `{ensemble_rmse:.4f}`",
        f"- Improvement over ElasticNet: `{improvement:.2f}%`",
        f"- Wilcoxon p-value vs ElasticNet residuals: `{w_p:.4g}`",
        "",
        "## Interpretation",
        "- Residual SAFE model is intended to capture what ElasticNet misses.",
        "- Ensemble weights are estimated from out-of-fold validation, not from the test labels.",
        "- If this still fails to beat ElasticNet materially, the ceiling is likely linear.",
    ]
    Path("elasticnet_residual_ensemble_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(candidates.to_string(index=False))
    print(weights)
    print(f"OOF RMSE: {oof_rmse:.4f}")


if __name__ == "__main__":
    main()
