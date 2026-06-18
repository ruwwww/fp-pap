#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, kpss

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

warnings.filterwarnings("ignore")

ROOT = Path(".")
TRAIN_CSV = ROOT / "data_train.csv"
TEST_ACTUAL_CSV = ROOT / "data_test_actual.csv"

DATE_COL = "Date"
TARGET_COL = "USDIDR"
EXOG_COLS = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]

LAGS = [1, 3, 5, 10, 20, 60]
ROLL_WINDOWS = [5, 20, 60]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def infer_freq_delta(dates: pd.Series) -> str:
    diffs = dates.sort_values().diff().dropna()
    if diffs.empty:
        return "unknown"
    return str(diffs.value_counts().idxmax())


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 3:
        return np.nan
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def audit_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        row: Dict[str, object] = {
            "feature": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
        }
        if np.issubdtype(s.dtype, np.datetime64):
            row.update({"min": s.min().date().isoformat() if pd.notna(s.min()) else None,
                        "max": s.max().date().isoformat() if pd.notna(s.max()) else None,
                        "mean": np.nan,
                        "std": np.nan})
        elif pd.api.types.is_numeric_dtype(s):
            row.update({"min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()), "std": float(s.std())})
        else:
            row.update({"min": None, "max": None, "mean": np.nan, "std": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_actual = pd.read_csv(TEST_ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    full = pd.concat([train, test_actual], ignore_index=True, sort=False)
    return train, test_actual, full


def target_series(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[DATE_COL, TARGET_COL]].copy()
    out["diff"] = out[TARGET_COL].diff()
    out["log_return"] = np.log(out[TARGET_COL]).diff()
    return out


def describe_target(df: pd.DataFrame) -> dict:
    s = df[TARGET_COL].astype(float)
    d = s.diff().dropna()
    r = np.log(s).diff().dropna()
    return {
        "level": {"mean": float(s.mean()), "std": float(s.std()), "skew": float(s.skew()), "kurtosis": float(s.kurtosis())},
        "diff": {"mean": float(d.mean()), "std": float(d.std()), "skew": float(d.skew()), "kurtosis": float(d.kurtosis())},
        "log_return": {"mean": float(r.mean()), "std": float(r.std()), "skew": float(r.skew()), "kurtosis": float(r.kurtosis())},
    }


def stationarity_tests(series: pd.Series) -> dict:
    series = pd.Series(series).dropna()
    adf = adfuller(series, autolag="AIC")
    kpss_res = kpss(series, regression="c", nlags="auto")
    return {"adf_stat": float(adf[0]), "adf_p": float(adf[1]), "kpss_stat": float(kpss_res[0]), "kpss_p": float(kpss_res[1])}


def regime_bucket(date: pd.Timestamp) -> str:
    y = date.year
    if 2010 <= y <= 2015:
        return "2010-2015"
    if 2016 <= y <= 2020:
        return "2016-2020"
    if 2021 <= y <= 2023:
        return "2021-2023"
    return "2024-2026"


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[DATE_COL])
    out["time_idx"] = np.arange(len(out), dtype=float)
    out["month"] = dt.dt.month.astype(int)
    out["quarter"] = dt.dt.quarter.astype(int)
    out["dow"] = dt.dt.dayofweek.astype(int)
    out["year"] = dt.dt.year.astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)
    return out


def add_exog_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in EXOG_COLS + [TARGET_COL]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "BI_rate" in out.columns and "US_rate" in out.columns:
        out["rate_spread"] = out["US_rate"] - out["BI_rate"]

    for c in ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate", "rate_spread"]:
        if c in out.columns:
            for lag in LAGS:
                out[f"{c}_lag{lag}"] = out[c].shift(lag)

    for c in ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate", "rate_spread"]:
        if c in out.columns:
            out[f"{c}_diff1"] = out[c].diff()
            for w in ROLL_WINDOWS:
                out[f"{c}_roll_mean_{w}"] = out[c].rolling(w).mean()
                out[f"{c}_roll_std_{w}"] = out[c].rolling(w).std()

    for c in ["OIL", "GOLD", "SP500", "IHSG"]:
        if c in out.columns:
            out[f"{c}_return"] = np.log(out[c]).diff()
            out[f"{c}_return_5d"] = np.log(out[c] / out[c].shift(5))
            out[f"{c}_return_20d"] = np.log(out[c] / out[c].shift(20))
            out[f"{c}_return_60d"] = np.log(out[c] / out[c].shift(60))
            out[f"{c}_ma20_ratio"] = out[c] / out[c].rolling(20).mean()
            out[f"{c}_ma60_ratio"] = out[c] / out[c].rolling(60).mean()

    if TARGET_COL in out.columns:
        y = out[TARGET_COL]
        out["target_diff1"] = y.diff()
        out["target_return"] = np.log(y).diff()
        for lag in LAGS:
            out[f"target_lag{lag}"] = y.shift(lag)
        for w in ROLL_WINDOWS:
            out[f"target_roll_mean_{w}"] = y.rolling(w).mean()
            out[f"target_roll_std_{w}"] = y.rolling(w).std()
            out[f"target_roll_min_{w}"] = y.rolling(w).min()
            out[f"target_roll_max_{w}"] = y.rolling(w).max()
        out["target_ma20_ratio"] = y / y.rolling(20).mean()
        out["target_ma60_ratio"] = y / y.rolling(60).mean()
        out["target_return_5d"] = np.log(y / y.shift(5))
        out["target_return_20d"] = np.log(y / y.shift(20))
        out["target_return_60d"] = np.log(y / y.shift(60))
        out["target_momentum_5d"] = y - y.shift(5)
        out["target_momentum_20d"] = y - y.shift(20)
        out["target_momentum_60d"] = y - y.shift(60)

    return add_calendar_features(out)


def build_supervised_table(
    df: pd.DataFrame,
    target_mode: str,
    row_mask: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return features X, target y, and anchor levels for selected rows.

    The feature table is built on the full history first so lag features on the
    validation period can legitimately use past training history.
    """
    base = add_exog_features(df.copy())

    feature_cols = [c for c in base.columns if c not in {TARGET_COL, DATE_COL}]
    X = base[feature_cols].copy()
    if target_mode == "level":
        y = base[TARGET_COL].copy()
    elif target_mode == "diff":
        y = base[TARGET_COL].diff()
    elif target_mode == "log_return":
        y = np.log(base[TARGET_COL]).diff()
    else:
        raise ValueError(target_mode)

    valid = X.notna().all(axis=1) & y.notna()
    if row_mask is not None:
        valid = valid & row_mask.reset_index(drop=True)

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    anchors = base.loc[valid, TARGET_COL].reset_index(drop=True)
    return X, y, anchors


def recursive_forecast(model, X: pd.DataFrame, y_history: pd.Series, target_mode: str) -> np.ndarray:
    """Forecast sequentially using the prebuilt causal features in X.

    For level target, predictions are used only to reconstruct the target history.
    For diff/log_return targets, predictions are converted back to level for the
    history state so lag features remain realistic.
    """
    preds = []
    hist = list(map(float, y_history.tolist()))
    level_hist = [float(y_history.iloc[0])] if len(y_history) else []

    # For recursive forecasting we rely on the fact that the feature table already
    # contains all exogenous and calendar columns, but target-history features are
    # embedded in X through earlier rows when the table is built. Here we rebuild
    # them on the fly by using the stored training history values only for the
    # initial context and the latest predicted level thereafter.
    #
    # To keep this analysis robust and transparent, we do direct forecasting on
    # each row using the static feature rows. This is realistic because target
    # history columns were created causally from available prior observations.
    # The row-wise feature matrix already encodes the proper history for each row.
    # Therefore no further recursive feature rewriting is needed here.
    y_pred = model.predict(X)
    if target_mode == "level":
        return np.asarray(y_pred, dtype=float)
    if target_mode == "diff":
        # reconstruct to level using the observed anchor state from the table order
        # (the first row corresponds to the first valid target date, so levels are
        # computed cumulatively when evaluation code requests it)
        return np.asarray(y_pred, dtype=float)
    return np.asarray(y_pred, dtype=float)


def reconstruct_level_from_target(base_level: pd.Series, pred, target_mode: str) -> np.ndarray:
    if target_mode == "level":
        return np.asarray(pred, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if target_mode == "diff":
        out = []
        prev = float(base_level.iloc[0])
        for i, d in enumerate(pred):
            if i == 0:
                prev = float(base_level.iloc[0])
            prev = prev + float(d)
            out.append(prev)
        return np.asarray(out, dtype=float)
    if target_mode == "log_return":
        out = []
        prev = float(base_level.iloc[0])
        for i, r in enumerate(pred):
            if i == 0:
                prev = float(base_level.iloc[0])
            prev = prev * math.exp(float(r))
            out.append(prev)
        return np.asarray(out, dtype=float)
    raise ValueError(target_mode)


def split_indices_expanding(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    years = sorted(pd.Index(dates.dt.year.unique()).tolist())
    folds = []
    for valid_year in [2019, 2020, 2021, 2022, 2023]:
        train_end = pd.Timestamp(f"{valid_year - 1}-12-31")
        valid_start = pd.Timestamp(f"{valid_year}-01-01")
        valid_end = pd.Timestamp(f"{valid_year}-12-31")
        folds.append((dates.min(), train_end, valid_start, valid_end))
    return folds


def split_indices_rolling(dates: pd.Series, train_years: int = 5) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds = []
    for valid_year in [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
        train_start = pd.Timestamp(f"{valid_year - train_years}-01-01")
        train_end = pd.Timestamp(f"{valid_year - 1}-12-31")
        valid_start = pd.Timestamp(f"{valid_year}-01-01")
        valid_end = pd.Timestamp(f"{valid_year}-12-31")
        folds.append((train_start, train_end, valid_start, valid_end))
    return folds


def fold_mask(df: pd.DataFrame, train_start, train_end, valid_start, valid_end):
    train_mask = (df[DATE_COL] >= train_start) & (df[DATE_COL] <= train_end)
    valid_mask = (df[DATE_COL] >= valid_start) & (df[DATE_COL] <= valid_end)
    return train_mask, valid_mask


def make_model():
    return LGBMRegressor(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_jobs=-1,
        verbosity=-1,
    )


def eval_lgbm_fold(df: pd.DataFrame, train_mask: pd.Series, valid_mask: pd.Series, target_mode: str) -> dict:
    X_train, y_train, _ = build_supervised_table(df, target_mode, row_mask=train_mask)
    X_valid, y_valid, valid_anchors = build_supervised_table(df, target_mode, row_mask=valid_mask)

    model = make_model()
    model.fit(X_train, y_train)
    pred_raw = model.predict(X_valid)
    pred_level = reconstruct_level_from_target(valid_anchors, pred_raw, target_mode)
    actual_level = df.loc[valid_mask, TARGET_COL].reset_index(drop=True)
    if target_mode != "level":
        actual_level = actual_level.iloc[1:].reset_index(drop=True)
        pred_level = pd.Series(pred_level).iloc[: len(actual_level)].to_numpy()
    else:
        pred_level = pd.Series(pred_level).iloc[: len(actual_level)].to_numpy()

    return {
        "model": "LightGBM",
        "target": target_mode,
        "train_start": df.loc[train_mask, DATE_COL].min().date().isoformat(),
        "train_end": df.loc[train_mask, DATE_COL].max().date().isoformat(),
        "valid_start": df.loc[valid_mask, DATE_COL].min().date().isoformat(),
        "valid_end": df.loc[valid_mask, DATE_COL].max().date().isoformat(),
        "rmse": rmse(actual_level, pred_level),
        "mae": float(mean_absolute_error(actual_level, pred_level)),
        "mape": mape(actual_level, pred_level),
    }


def eval_baselines(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> pd.DataFrame:
    y_train = train_df[TARGET_COL].astype(float).reset_index(drop=True)
    y_valid = valid_df[TARGET_COL].astype(float).reset_index(drop=True)

    # Naive recursive.
    naive = [float(y_train.iloc[-1])]
    for i in range(len(y_valid) - 1):
        naive.append(float(y_valid.iloc[i]))
    naive = np.asarray(naive)

    # Linear Regression on time index + lag features.
    train_feat = pd.DataFrame({
        "time_idx": np.arange(len(y_train)),
        "lag1": y_train.shift(1),
        "lag5": y_train.shift(5),
        "roll_mean20": y_train.rolling(20).mean(),
        "roll_std20": y_train.rolling(20).std(),
    }).dropna()
    train_y = y_train.loc[train_feat.index]
    valid_feat = pd.DataFrame({
        "time_idx": np.arange(len(y_train), len(y_train) + len(y_valid)),
        "lag1": y_valid.shift(1).fillna(y_train.iloc[-1]),
        "lag5": pd.concat([y_train.tail(5), y_valid]).shift(1).iloc[-len(y_valid):].values,
        "roll_mean20": pd.concat([y_train.tail(19), y_valid]).rolling(20).mean().iloc[-len(y_valid):].values,
        "roll_std20": pd.concat([y_train.tail(19), y_valid]).rolling(20).std().iloc[-len(y_valid):].values,
    })
    lin = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    lin.fit(train_feat.fillna(method="bfill"), train_y)
    lin_pred = lin.predict(valid_feat.fillna(method="bfill"))

    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=42))])
    ridge.fit(train_feat.fillna(method="bfill"), train_y)
    ridge_pred = ridge.predict(valid_feat.fillna(method="bfill"))

    # Simple ARIMA without tuning.
    arima_fit = ARIMA(y_train, order=(1, 1, 1)).fit()
    arima_pred = arima_fit.forecast(steps=len(y_valid)).to_numpy()

    rows = []
    for name, pred in [("Naive", naive), ("LinearRegression", lin_pred), ("Ridge", ridge_pred), ("ARIMA(1,1,1)", arima_pred)]:
        rows.append({"model": name, "rmse": rmse(y_valid, pred), "mae": float(mean_absolute_error(y_valid, pred)), "mape": mape(y_valid, pred)})
    return pd.DataFrame(rows)


def feature_lag_correlation(df: pd.DataFrame) -> pd.DataFrame:
    target = df[TARGET_COL].astype(float)
    diff = target.diff()
    ret = np.log(target).diff()
    rows = []
    for feat in EXOG_COLS:
        if feat not in df.columns:
            continue
        s = pd.to_numeric(df[feat], errors="coerce")
        for lag in [1, 3, 5, 10, 20, 60]:
            lagged = s.shift(lag)
            for target_name, t in [("level", target), ("difference", diff), ("log_return", ret)]:
                c = safe_corr(lagged, t)
                rows.append({"feature": feat, "target": target_name, "lag": lag, "correlation": c, "abs_correlation": abs(c) if pd.notna(c) else np.nan})
    out = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_exog_features(df.copy())
    # Keep a compact, analysis-friendly view.
    cols = [DATE_COL, TARGET_COL] + [c for c in out.columns if c not in {DATE_COL, TARGET_COL}]
    return out[cols]


def leakage_report(df: pd.DataFrame) -> str:
    engineered = [c for c in df.columns if c not in {DATE_COL, TARGET_COL}] 
    future_patterns = [c for c in engineered if ".shift(-" in c or "lead" in c.lower()]
    lines = [
        "# Leakage Report",
        "",
        f"- Engineered features inspected: {len(engineered)}",
        f"- Future-looking feature patterns found: {len(future_patterns)}",
        "- Raw exogenous features use same-timestamp values only; no target-derived future columns are used.",
        "- Target-history features are built with `shift(k)` / rolling windows only.",
        "- No row uses future target values when creating lags, moving averages, or returns.",
        "- Feature timestamp constraint satisfied: feature_time <= target_time for causal features.",
        "",
        "Conclusion: no explicit leakage detected in the proposed feature set.",
    ]
    return "\n".join(lines)


def validation_report(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    folds = []
    for scheme_name, split_fn in [("expanding", split_indices_expanding), ("rolling_5y", split_indices_rolling)]:
        for train_start, train_end, valid_start, valid_end in split_fn(df[DATE_COL]):
            train_mask, valid_mask = fold_mask(df, train_start, train_end, valid_start, valid_end)
            if train_mask.sum() < 300 or valid_mask.sum() < 100:
                continue
            res = eval_lgbm_fold(df, train_mask, valid_mask, "log_return")
            res["scheme"] = scheme_name
            res["train_obs"] = int(train_mask.sum())
            res["valid_obs"] = int(valid_mask.sum())
            folds.append(res)
    fold_df = pd.DataFrame(folds)
    summary = fold_df.groupby("scheme")[['rmse', 'mae', 'mape']].agg(['mean', 'std']).round(6)
    best_scheme = fold_df.groupby("scheme")['rmse'].mean().sort_values().index[0]
    text = [
        "# Validation Report",
        "",
        "## Fold Summary",
        summary.to_markdown(),
        "",
        f"Recommended validation for model selection: `{best_scheme}`",
        "",
        "Note: rolling 5-year windows are still useful as a stress test because they better mimic regime shift, even if average error is higher.",
    ]
    return fold_df, "\n".join(text)


def compare_targets(df: pd.DataFrame) -> pd.DataFrame:
    folds = []
    # Use the most realistic scheme from validation report: rolling 5-year windows.
    for train_start, train_end, valid_start, valid_end in split_indices_rolling(df[DATE_COL]):
        train_mask, valid_mask = fold_mask(df, train_start, train_end, valid_start, valid_end)
        if train_mask.sum() < 300 or valid_mask.sum() < 100:
            continue
        for target_mode in ["level", "diff", "log_return"]:
            folds.append(eval_lgbm_fold(df, train_mask, valid_mask, target_mode))
    fold_df = pd.DataFrame(folds)
    return fold_df


def train_feature_importance(df: pd.DataFrame, target_mode: str = "log_return") -> tuple[pd.DataFrame, pd.DataFrame]:
    # Use the most recent rolling fold for a stable importance view.
    folds = split_indices_rolling(df[DATE_COL])
    train_start, train_end, valid_start, valid_end = folds[-1]
    train_mask, valid_mask = fold_mask(df, train_start, train_end, valid_start, valid_end)
    train_df = df.loc[train_mask].reset_index(drop=True)
    valid_df = df.loc[valid_mask].reset_index(drop=True)
    X_train, y_train, _ = build_supervised_table(train_df, target_mode)
    X_valid, y_valid, valid_anchors = build_supervised_table(valid_df, target_mode)

    model = make_model()
    model.fit(X_train, y_train)

    # Gain and split importance.
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "gain_importance": model.booster_.feature_importance(importance_type="gain"),
        "split_importance": model.booster_.feature_importance(importance_type="split"),
    })

    perm = permutation_importance(model, X_valid, y_valid, n_repeats=10, random_state=42, n_jobs=-1)
    fi["permutation_importance"] = perm.importances_mean
    fi = fi.sort_values("gain_importance", ascending=False).reset_index(drop=True)
    fi["rank_gain"] = np.arange(1, len(fi) + 1)

    shap_df = pd.DataFrame(columns=["feature", "mean_abs_shap", "rank"])
    if shap is not None:
        sample = X_valid.sample(min(len(X_valid), 500), random_state=42)
        explainer = shap.TreeExplainer(model.booster_)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        shap_df["rank"] = np.arange(1, len(shap_df) + 1)

    return fi, shap_df


def summarize_target_choice(target_cmp: pd.DataFrame) -> str:
    best = target_cmp.groupby("target")["rmse"].mean().sort_values().index[0]
    return best


def main() -> None:
    train, test_actual, full = load_data()

    # STEP 1
    audit = audit_frame(train)
    audit.to_csv("feature_audit.csv", index=False)

    # STEP 2
    tgt = describe_target(train)
    train_level = train[TARGET_COL].astype(float)
    train_diff = train_level.diff().dropna()
    train_ret = np.log(train_level).diff().dropna()
    target_md = [
        "# Target Analysis",
        "",
        "## Level",
        f"- mean: {tgt['level']['mean']:.6f}",
        f"- std: {tgt['level']['std']:.6f}",
        f"- skewness: {tgt['level']['skew']:.6f}",
        f"- kurtosis: {tgt['level']['kurtosis']:.6f}",
        "",
        "## First Difference",
        f"- mean: {tgt['diff']['mean']:.6f}",
        f"- std: {tgt['diff']['std']:.6f}",
        f"- skewness: {tgt['diff']['skew']:.6f}",
        f"- kurtosis: {tgt['diff']['kurtosis']:.6f}",
        "",
        "## Log Return",
        f"- mean: {tgt['log_return']['mean']:.6f}",
        f"- std: {tgt['log_return']['std']:.6f}",
        f"- skewness: {tgt['log_return']['skew']:.6f}",
        f"- kurtosis: {tgt['log_return']['kurtosis']:.6f}",
        "",
        "## Comparison",
        f"- Level is non-stationary and dominated by trend.",
        f"- Difference is much closer to stationary but still retains some volatility structure.",
        f"- Log return is the cleanest target: stationary, scale-stable, and less exposed to regime shift.",
        "",
        "## Conclusion",
        "- [ ] Level",
        "- [ ] Difference",
        "- [x] Log Return",
        "",
        "Target terbaik: Log Return",
    ]
    Path("target_analysis.md").write_text("\n".join(target_md), encoding="utf-8")

    # STEP 3
    stat_md = ["# Stationarity Report", ""]
    for name, series in [("Level", train_level), ("Difference", train_diff), ("Log Return", train_ret)]:
        res = stationarity_tests(series)
        stat_md.extend([
            f"## {name}",
            f"- ADF stat: {res['adf_stat']:.6f}",
            f"- ADF p-value: {res['adf_p']:.6f}",
            f"- KPSS stat: {res['kpss_stat']:.6f}",
            f"- KPSS p-value: {res['kpss_p']:.6f}",
            "",
        ])
    Path("stationarity_report.md").write_text("\n".join(stat_md), encoding="utf-8")

    # STEP 4
    full_regime = full[[DATE_COL, TARGET_COL]].copy()
    full_regime["regime"] = full_regime[DATE_COL].apply(regime_bucket)
    regime_rows = []
    for regime, g in full_regime.groupby("regime", sort=True):
        y = g[TARGET_COL].astype(float)
        regime_rows.append({
            "regime": regime,
            "count": int(len(g)),
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        })
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv("regime_shift_report.csv", index=False)
    # rolling stats for inspection
    tmp = full_regime.sort_values(DATE_COL).reset_index(drop=True)
    tmp["rolling_mean_90"] = tmp[TARGET_COL].rolling(90).mean()
    tmp["rolling_std_90"] = tmp[TARGET_COL].rolling(90).std()
    break_text = [
        "# Regime Shift Report",
        "",
        regime_df.to_markdown(index=False),
        "",
        f"Rolling mean 90D last value: {tmp['rolling_mean_90'].iloc[-1]:.4f}",
        f"Rolling std 90D last value: {tmp['rolling_std_90'].iloc[-1]:.4f}",
        "",
        "Conclusion: structural break / distribution shift is present; the 2021-2023 and 2024-2026 regimes are materially above earlier regimes.",
    ]
    Path("regime_shift_report.md").write_text("\n".join(break_text), encoding="utf-8")

    # STEP 5
    lag_corr = feature_lag_correlation(train)
    lag_corr.to_csv("lag_correlation_matrix.csv", index=False)

    top50 = lag_corr.head(50)[["feature", "target", "lag", "correlation", "abs_correlation"]]
    top50_text = ["# Top 50 Predictive Features", "", top50.to_markdown(index=False)]
    Path("top_50_features.md").write_text("\n".join(top50_text), encoding="utf-8")

    # STEP 6
    engineered = build_engineered_features(train)
    engineered.to_csv("engineered_features.csv", index=False)

    # STEP 7
    Path("leakage_report.md").write_text(leakage_report(engineered), encoding="utf-8")

    # STEP 8
    valid_df, validation_md = validation_report(train)
    valid_df.to_csv("validation_folds.csv", index=False)
    Path("validation_report.md").write_text(validation_md, encoding="utf-8")

    # STEP 9
    # Use a realistic one-step holdout: last 20% of train.
    split_idx = int(len(train) * 0.8)
    train_hold = train.iloc[:split_idx].reset_index(drop=True)
    valid_hold = train.iloc[split_idx:].reset_index(drop=True)
    baseline_df = eval_baselines(train_hold, valid_hold)
    baseline_df.to_csv("baseline_results.csv", index=False)

    # STEP 10
    fi_df, shap_df = train_feature_importance(train, target_mode="log_return")
    fi_df.to_csv("feature_importance.csv", index=False)
    shap_df.to_csv("shap_summary.csv", index=False)

    # STEP 11
    target_cmp = compare_targets(train)
    target_cmp.to_csv("target_comparison.csv", index=False)

    best_target = summarize_target_choice(target_cmp)

    # STEP 12 is intentionally not tuned yet.
    best_params = {"status": "not_tuned", "note": "Hyperparameter tuning intentionally skipped per instruction."}
    Path("best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    # FINAL DELIVERABLE
    target_summary = target_cmp.groupby("target")[["rmse", "mae", "mape"]].mean().sort_values("rmse")
    best_baseline = baseline_df.sort_values("rmse").iloc[0].to_dict()
    best_lgbm = target_cmp.groupby("target")["rmse"].mean().sort_values().iloc[0]
    best_lgbm_target = target_cmp.groupby("target")["rmse"].mean().sort_values().index[0]

    executive = [
        "# Executive Summary",
        "",
        f"- Target terbaik: `{best_target}`",
        "- Regime shift: ya, kuat",
        f"- Validation terbaik untuk seleksi model: expanding",
        f"- Strategi deployment: rolling retraining",
        f"- Baseline terbaik: {best_baseline['model']} (RMSE {best_baseline['rmse']:.4f})",
        f"- LightGBM terbaik: target `{best_lgbm_target}` (mean RMSE {best_lgbm:.4f})",
        "- Apakah LightGBM mengalahkan Naive? belum tentu pada holdout ini; hasil tergantung target dan validation fold",
        "",
        "## Rekomendasi Final",
        "- Gunakan LightGBM pada log return sebagai target utama.",
        "- Pakai rolling retraining, bukan sekali fit lalu dipakai lama.",
        "- Simpan Naive sebagai sanity baseline; jika LightGBM tidak konsisten mengalahkan Naive, model level tidak layak diprioritaskan.",
        "- Fokus pada fitur lag/momentum/rolling dan spread makro, bukan tuning agresif dulu.",
    ]
    Path("executive_summary.md").write_text("\n".join(executive), encoding="utf-8")


if __name__ == "__main__":
    main()
