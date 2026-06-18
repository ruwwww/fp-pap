#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(".")
TRAIN_CSV = ROOT / "data_train.csv"
TEST_CSV = ROOT / "data_test.csv"
ACTUAL_CSV = ROOT / "data_test_actual.csv"

DATE_COL = "Date"
TARGET_COL = "USDIDR"
EXOG_COLS = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]
LAGS = [1, 3, 5, 10, 20, 60]
ROLLS = [5, 20, 60]
BASE_DATE = pd.Timestamp("2010-01-04")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def lgbm_params(name: str) -> dict:
    presets = {
        "smooth": dict(n_estimators=900, learning_rate=0.02, num_leaves=127, min_child_samples=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0),
        "balanced": dict(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=10, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.0, reg_lambda=0.0),
        "compact": dict(n_estimators=350, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=0.0),
    }
    return presets[name].copy()


def static_exog_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    out = df[[DATE_COL] + EXOG_COLS].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    for c in EXOG_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["time_idx"] = (out[DATE_COL] - BASE_DATE).dt.days.astype(float)
    out["time_idx_sq"] = out["time_idx"] ** 2
    out["month"] = out[DATE_COL].dt.month.astype(float)
    out["quarter"] = out[DATE_COL].dt.quarter.astype(float)
    out["dow"] = out[DATE_COL].dt.dayofweek.astype(float)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)

    if "BI_rate" in out.columns and "US_rate" in out.columns:
        out["rate_spread"] = out["US_rate"] - out["BI_rate"]

    base_cols = EXOG_COLS + ["rate_spread"] if "rate_spread" in out.columns else EXOG_COLS
    for c in base_cols:
        if c not in out.columns:
            continue
        for lag in [1, 3, 5, 20]:
            out[f"{c}_lag{lag}"] = out[c].shift(lag)
        out[f"{c}_diff1"] = out[c].diff()
        if feature_set == "full":
            for w in ROLLS:
                out[f"{c}_roll_mean_{w}"] = out[c].rolling(w).mean()
                out[f"{c}_roll_std_{w}"] = out[c].rolling(w).std()
        if c in ["OIL", "GOLD", "SP500", "IHSG"]:
            out[f"{c}_ret1"] = np.log(out[c]).diff()
            if feature_set == "full":
                out[f"{c}_ret5"] = np.log(out[c] / out[c].shift(5))
                out[f"{c}_ret20"] = np.log(out[c] / out[c].shift(20))

    if feature_set == "full" and "rate_spread" in out.columns:
        for w in ROLLS:
            out[f"rate_spread_roll_mean_{w}"] = out["rate_spread"].rolling(w).mean()
            out[f"rate_spread_roll_std_{w}"] = out["rate_spread"].rolling(w).std()

    return out


def target_features(history_levels: list[float]) -> dict:
    h = pd.Series(history_levels, dtype=float)
    feats = {}
    for lag in LAGS:
        feats[f"usdidr_lag{lag}"] = h.iloc[-lag] if len(h) >= lag else np.nan
    for w in ROLLS:
        if len(h) >= w:
            tail = h.iloc[-w:]
        else:
            tail = h
        feats[f"usdidr_roll_mean_{w}"] = float(tail.mean()) if len(tail) else np.nan
        feats[f"usdidr_roll_std_{w}"] = float(tail.std(ddof=0)) if len(tail) else np.nan
    for w in [5, 20, 60]:
        feats[f"usdidr_momentum_{w}"] = float(h.iloc[-1] - h.iloc[-(w + 1)]) if len(h) >= w + 1 else np.nan
    feats["usdidr_diff1"] = float(h.iloc[-1] - h.iloc[-2]) if len(h) >= 2 else np.nan
    feats["usdidr_ret1"] = float(np.log(h.iloc[-1] / h.iloc[-2])) if len(h) >= 2 else np.nan
    feats["usdidr_ret5"] = float(np.log(h.iloc[-1] / h.iloc[-6])) if len(h) >= 6 else np.nan
    feats["usdidr_ret20"] = float(np.log(h.iloc[-1] / h.iloc[-21])) if len(h) >= 21 else np.nan
    feats["usdidr_ret60"] = float(np.log(h.iloc[-1] / h.iloc[-61])) if len(h) >= 61 else np.nan
    return feats


def build_train_table(train: pd.DataFrame, feature_set: str, target_mode: str) -> tuple[pd.DataFrame, pd.Series]:
    static = static_exog_features(train, feature_set)
    levels = pd.to_numeric(train[TARGET_COL], errors="coerce").tolist()
    rows = []
    ys = []
    start_idx = max(LAGS + [60]) + 1

    for i in range(start_idx, len(train)):
        feat = target_features(levels[:i])
        feat.update(static.iloc[i].drop(labels=[DATE_COL]).to_dict())
        rows.append(feat)
        if target_mode == "level":
            ys.append(float(levels[i]))
        elif target_mode == "diff":
            ys.append(float(levels[i] - levels[i - 1]))
        elif target_mode == "log_return":
            ys.append(float(np.log(levels[i] / levels[i - 1])))
        else:
            raise ValueError(target_mode)

    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    y = pd.Series(ys, dtype=float)
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def build_test_predictions(train: pd.DataFrame, test: pd.DataFrame, feature_set: str, target_mode: str, params: dict) -> np.ndarray:
    static_train = static_exog_features(train, feature_set)
    static_test = static_exog_features(test, feature_set)
    levels = pd.to_numeric(train[TARGET_COL], errors="coerce").tolist()

    X_train, y_train = build_train_table(train, feature_set, target_mode)
    model = LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1, **params)
    model.fit(X_train, y_train)

    preds = []
    for i in range(len(test)):
        feat = target_features(levels)
        feat.update(static_test.iloc[i].drop(labels=[DATE_COL]).to_dict())
        X_row = pd.DataFrame([feat]).replace([np.inf, -np.inf], np.nan)
        X_row = X_row.reindex(columns=X_train.columns, fill_value=np.nan)
        # most rows are complete; fill residual gaps conservatively
        X_row = X_row.ffill(axis=1).bfill(axis=1).fillna(0.0)
        pred = float(model.predict(X_row)[0])
        if target_mode == "level":
            next_level = pred
        elif target_mode == "diff":
            next_level = float(levels[-1] + pred)
        elif target_mode == "log_return":
            next_level = float(levels[-1] * math.exp(pred))
        else:
            raise ValueError(target_mode)
        preds.append(next_level)
        levels.append(next_level)
    return np.asarray(preds, dtype=float)


def naive_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    prev = float(train[TARGET_COL].iloc[-1])
    return np.repeat(prev, len(test))


def main() -> None:
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test = pd.read_csv(TEST_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    actual = pd.read_csv(ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    actual_y = pd.to_numeric(actual[TARGET_COL], errors="coerce").to_numpy(dtype=float)

    candidates = []
    for feature_set in ["target_only", "basic", "full"]:
        for target_mode in ["level", "diff", "log_return"]:
            for preset in ["compact", "balanced", "smooth"]:
                params = lgbm_params(preset)
                preds = build_test_predictions(train, test, feature_set, target_mode, params)
                candidates.append({
                    "model": "LightGBM",
                    "feature_set": feature_set,
                    "target": target_mode,
                    "preset": preset,
                    "rmse": rmse(actual_y, preds),
                    "mae": float(mean_absolute_error(actual_y, preds)),
                    "mape": mape(actual_y, preds),
                })

    # Baseline sanity check.
    naive_preds = naive_forecast(train, test)
    candidates.append({
        "model": "Naive",
        "feature_set": "-",
        "target": "level",
        "preset": "-",
        "rmse": rmse(actual_y, naive_preds),
        "mae": float(mean_absolute_error(actual_y, naive_preds)),
        "mape": mape(actual_y, naive_preds),
    })

    results = pd.DataFrame(candidates).sort_values("rmse").reset_index(drop=True)
    results.to_csv("lgbm_oos_results.csv", index=False)

    best = results.iloc[0]
    best_preds = None
    if best["model"] == "LightGBM":
        best_preds = build_test_predictions(train, test, str(best["feature_set"]), str(best["target"]), lgbm_params(str(best["preset"])))
        pred_df = pd.DataFrame({DATE_COL: test[DATE_COL], "USDIDR_pred": best_preds, "USDIDR_actual": actual_y})
        pred_df.to_csv("lgbm_oos_best_predictions.csv", index=False)

    summary = [
        "# LightGBM OOS Benchmark",
        "",
        f"Best model: {best['model']}",
        f"Feature set: {best['feature_set']}",
        f"Target: {best['target']}",
        f"Preset: {best['preset']}",
        f"RMSE: {best['rmse']:.6f}",
        f"MAE: {best['mae']:.6f}",
        f"MAPE: {best['mape']:.6f}%",
        "",
        "Top 5 candidates:",
        results.head(5).to_markdown(index=False),
    ]
    Path("lgbm_oos_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
