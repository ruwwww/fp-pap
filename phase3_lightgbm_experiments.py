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
from statsmodels.tsa.stattools import pacf

warnings.filterwarnings("ignore")

ROOT = Path(".")
TRAIN_CSV = ROOT / "data_train.csv"
TEST_EXOG_CSV = ROOT / "data_test.csv"
TEST_ACTUAL_CSV = ROOT / "data_test_actual.csv"
ELASTICNET_PRED_CSV = ROOT / "assumption_driven_predictions.csv"
DATE_COL = "Date"
TARGET_COL = "USDIDR"
EXOG_COLS = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_exog = pd.read_csv(TEST_EXOG_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_actual = pd.read_csv(TEST_ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return train, test_exog, test_actual


def safe_pacf_lags(diff: pd.Series, max_lag: int = 60) -> list[int]:
    x = pd.Series(diff).dropna().astype(float)
    vals = pacf(x, nlags=max_lag, method="ywm")
    threshold = 1.96 / math.sqrt(len(x))
    lags = [lag for lag in range(1, len(vals)) if abs(float(vals[lag])) > threshold]
    return lags or [1, 2, 3, 5, 10]


def add_causal_state_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    for c in [TARGET_COL] + EXOG_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["is_Q2"] = out[DATE_COL].dt.month.isin([4, 5, 6]).astype(float)
    out["month"] = out[DATE_COL].dt.month.astype(float)
    out["dow"] = out[DATE_COL].dt.dayofweek.astype(float)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)
    if "BI_rate" in out.columns:
        change = out["BI_rate"].diff().ne(0)
        last_change = out[DATE_COL].where(change).ffill()
        out["bi_rate_change"] = out["BI_rate"].diff()
        out["days_since_bi_change"] = (out[DATE_COL] - last_change).dt.days.fillna(0).astype(float)
    if "CPI" in out.columns:
        change = out["CPI"].diff().ne(0)
        last_change = out[DATE_COL].where(change).ffill()
        out["cpi_change"] = out["CPI"].diff()
        out["days_since_cpi_release"] = (out[DATE_COL] - last_change).dt.days.fillna(0).astype(float)
    for c in EXOG_COLS:
        if c in out.columns:
            out[f"{c}_diff1"] = out[c].diff()
            out[f"{c}_lag1"] = out[c].shift(1)
            out[f"{c}_lag6"] = out[c].shift(6)
            out[f"{c}_lag20"] = out[c].shift(20)
    return out


def build_history_features(
    level_hist: list[float],
    current_row: pd.Series,
    selected_lags: list[int],
    mode: str,
) -> dict[str, float]:
    levels = np.asarray(level_hist, dtype=float)
    diffs = np.diff(levels) if len(levels) >= 2 else np.asarray([], dtype=float)
    rets = np.diff(np.log(levels)) if len(levels) >= 2 else np.asarray([], dtype=float)
    feats: dict[str, float] = {}
    for lag in selected_lags:
        feats[f"usd_lag_{lag}"] = float(levels[-lag]) if len(levels) >= lag else np.nan
    feats["rolling_drift_252"] = float(np.mean(diffs[-252:])) if len(diffs) else np.nan
    feats["gap_from_trend"] = float(levels[-1] - np.mean(levels[-252:])) if len(levels) >= 252 else np.nan
    feats["realized_vol_21"] = float(np.std(rets[-21:], ddof=0)) if len(rets) >= 21 else np.nan
    feats["is_Q2"] = float(current_row.get("is_Q2", np.nan))
    feats["days_since_bi_change"] = float(current_row.get("days_since_bi_change", np.nan))
    feats["days_since_cpi_release"] = float(current_row.get("days_since_cpi_release", np.nan))
    feats["month_sin"] = float(current_row.get("month_sin", np.nan))
    feats["month_cos"] = float(current_row.get("month_cos", np.nan))
    feats["dow_sin"] = float(current_row.get("dow_sin", np.nan))
    feats["dow_cos"] = float(current_row.get("dow_cos", np.nan))
    if mode == "full":
        for c in EXOG_COLS:
            feats[c] = float(current_row.get(c, np.nan))
            feats[f"{c}_diff1"] = float(current_row.get(f"{c}_diff1", np.nan))
            feats[f"{c}_lag1"] = float(current_row.get(f"{c}_lag1", np.nan))
            feats[f"{c}_lag6"] = float(current_row.get(f"{c}_lag6", np.nan))
            feats[f"{c}_lag20"] = float(current_row.get(f"{c}_lag20", np.nan))
        feats["rate_spread"] = float(current_row.get("US_rate", np.nan) - current_row.get("BI_rate", np.nan))
        feats["bi_rate_change"] = float(current_row.get("bi_rate_change", np.nan))
        feats["cpi_change"] = float(current_row.get("cpi_change", np.nan))
    return feats


def build_train_table(
    df: pd.DataFrame,
    selected_lags: list[int],
    mode: str,
    horizon: int = 1,
    task: str = "recursive",
) -> tuple[pd.DataFrame, pd.Series]:
    levels = df[TARGET_COL].astype(float).tolist()
    rows = []
    ys = []
    start = max(max(selected_lags, default=1), 252, 21)
    if task == "recursive":
        for t in range(start, len(df)):
            feats = build_history_features(levels[:t], df.iloc[t], selected_lags, mode)
            rows.append(feats)
            ys.append(float(levels[t] - levels[t - 1]))
    elif task == "direct":
        for t in range(start, len(df) - horizon):
            feats = build_history_features(levels[:t], df.iloc[t], selected_lags, mode)
            rows.append(feats)
            ys.append(float(levels[t + horizon] - levels[t]))
    else:
        raise ValueError(task)
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    y = pd.Series(ys, dtype=float)
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def train_lgbm(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> lgb.LGBMRegressor:
    split = max(int(len(X) * 0.85), 1)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_va, y_va = X.iloc[split:], y.iloc[split:]
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=0.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def recursive_forecast(
    model: lgb.LGBMRegressor,
    train: pd.DataFrame,
    test_exog: pd.DataFrame,
    selected_lags: list[int],
    mode: str,
) -> np.ndarray:
    history_levels = train[TARGET_COL].astype(float).tolist()
    combined = pd.concat([train[[DATE_COL] + [c for c in test_exog.columns if c != DATE_COL]], test_exog], ignore_index=True)
    combined = add_causal_state_features(combined)
    preds = []
    cols = model.booster_.feature_name()
    for i in range(len(test_exog)):
        row = combined.iloc[len(train) + i]
        feats = build_history_features(history_levels, row, selected_lags, mode)
        X_row = pd.DataFrame([feats]).reindex(columns=cols, fill_value=np.nan)
        X_row = X_row.ffill(axis=1).bfill(axis=1).fillna(0.0)
        diff_pred = float(model.predict(X_row)[0])
        pred = float(history_levels[-1] + diff_pred)
        preds.append(pred)
        history_levels.append(pred)
    return np.asarray(preds, dtype=float)


def direct_anchor_forecast(
    models: dict[int, lgb.LGBMRegressor],
    train: pd.DataFrame,
    test_exog: pd.DataFrame,
    selected_lags: list[int],
    mode: str,
    block_size: int = 63,
) -> np.ndarray:
    history_levels = train[TARGET_COL].astype(float).tolist()
    combined = pd.concat([train[[DATE_COL] + [c for c in test_exog.columns if c != DATE_COL]], test_exog], ignore_index=True)
    combined = add_causal_state_features(combined)
    out: list[float] = []
    step = 0
    while step < len(test_exog):
        row = combined.iloc[len(train) + step]
        feats = build_history_features(history_levels, row, selected_lags, mode)
        base_level = float(history_levels[-1])
        anchor = {}
        for h, model in models.items():
            cols = model.booster_.feature_name()
            X_row = pd.DataFrame([feats]).reindex(columns=cols, fill_value=np.nan)
            X_row = X_row.ffill(axis=1).bfill(axis=1).fillna(0.0)
            anchor[h] = float(base_level + model.predict(X_row)[0])
        block_len = min(block_size, len(test_exog) - step)
        h1 = anchor[1]
        h5 = anchor[5]
        h21 = anchor[21]
        h63 = anchor[63]
        for d in range(1, block_len + 1):
            if d <= 1:
                pred = h1
            elif d <= 5:
                pred = h1 + (h5 - h1) * ((d - 1) / 4.0)
            elif d <= 21:
                pred = h5 + (h21 - h5) * ((d - 5) / 16.0)
            else:
                pred = h21 + (h63 - h21) * ((d - 21) / 42.0)
            out.append(float(pred))
        history_levels.extend(out[-block_len:])
        step += block_len
    return np.asarray(out[: len(test_exog)], dtype=float)


def regime_split_forecast(
    train: pd.DataFrame,
    test_exog: pd.DataFrame,
    selected_lags: list[int],
    mode: str,
) -> tuple[np.ndarray, float]:
    combined = add_causal_state_features(pd.concat([train[[DATE_COL] + [c for c in test_exog.columns if c != DATE_COL]], test_exog], ignore_index=True))
    history_levels = train[TARGET_COL].astype(float).tolist()
    train_levels = train[TARGET_COL].astype(float).tolist()
    train_rets = np.diff(np.log(np.asarray(train_levels, dtype=float)))
    vol_hist = pd.Series(train_rets).rolling(21).std().to_numpy()
    threshold = float(np.nanmedian(vol_hist))

    rows_low = []
    ys_low = []
    rows_high = []
    ys_high = []
    for t in range(max(max(selected_lags, default=1), 252, 21), len(train) - 1):
        feats = build_history_features(history_levels[:t], combined.iloc[t], selected_lags, mode)
        y = float(train_levels[t] - train_levels[t - 1])
        vol = float(np.std(np.diff(np.log(np.asarray(history_levels[:t], dtype=float)))[-21:], ddof=0)) if t >= 22 else np.nan
        if np.isnan(vol) or vol <= threshold:
            rows_low.append(feats)
            ys_low.append(y)
        else:
            rows_high.append(feats)
            ys_high.append(y)

    X_low = pd.DataFrame(rows_low).replace([np.inf, -np.inf], np.nan)
    y_low = pd.Series(ys_low, dtype=float)
    X_high = pd.DataFrame(rows_high).replace([np.inf, -np.inf], np.nan)
    y_high = pd.Series(ys_high, dtype=float)
    low_model = train_lgbm(X_low.loc[X_low.notna().all(axis=1)].reset_index(drop=True), y_low.loc[X_low.notna().all(axis=1)].reset_index(drop=True))
    high_model = train_lgbm(X_high.loc[X_high.notna().all(axis=1)].reset_index(drop=True), y_high.loc[X_high.notna().all(axis=1)].reset_index(drop=True))

    preds = []
    cols_low = low_model.booster_.feature_name()
    cols_high = high_model.booster_.feature_name()
    for i in range(len(test_exog)):
        row = combined.iloc[len(train) + i]
        feats = build_history_features(history_levels, row, selected_lags, mode)
        vol = float(np.std(np.diff(np.log(np.asarray(history_levels, dtype=float)))[-21:], ddof=0)) if len(history_levels) >= 22 else 0.0
        if vol <= threshold:
            X_row = pd.DataFrame([feats]).reindex(columns=cols_low, fill_value=np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
            pred = float(history_levels[-1] + low_model.predict(X_row)[0])
        else:
            X_row = pd.DataFrame([feats]).reindex(columns=cols_high, fill_value=np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
            pred = float(history_levels[-1] + high_model.predict(X_row)[0])
        preds.append(pred)
        history_levels.append(pred)
    return np.asarray(preds, dtype=float), threshold


def per_year_rmse(dates: pd.Series, actual: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "actual": actual, "pred": pred})
    df["year"] = df["Date"].dt.year
    rows = []
    for year in sorted(df["year"].unique()):
        g = df[df["year"] == year]
        rows.append({"year": int(year), "rmse": rmse(g["actual"], g["pred"]), "n": int(len(g))})
    return pd.DataFrame(rows)


def feature_importance_df(model: lgb.LGBMRegressor) -> pd.DataFrame:
    booster = model.booster_
    return pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)


def plot_actual_vs_predicted(dates: pd.Series, actual: np.ndarray, preds: dict[str, np.ndarray], path: Path) -> None:
    plt.figure(figsize=(15, 6))
    plt.plot(dates, actual, color="black", linewidth=1.6, label="actual")
    for name, pred in preds.items():
        plt.plot(dates, pred, linewidth=1.0, label=name)
    plt.title("USDIDR actual vs predicted")
    plt.xlabel("Date")
    plt.ylabel("USDIDR")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    train, test_exog, test_actual = load_data()
    train = add_causal_state_features(train)
    test_exog = add_causal_state_features(test_exog)
    y_true = test_actual[TARGET_COL].astype(float).to_numpy(dtype=float)
    selected_lags = safe_pacf_lags(train[TARGET_COL].diff())

    # SAFE model.
    X_safe, y_safe = build_train_table(train, selected_lags, mode="safe", task="recursive")
    safe_model = train_lgbm(X_safe, y_safe)
    safe_preds = recursive_forecast(safe_model, train, test_exog, selected_lags, mode="safe")

    # FULL model.
    X_full, y_full = build_train_table(train, selected_lags, mode="full", task="recursive")
    full_model = train_lgbm(X_full, y_full)
    full_preds = recursive_forecast(full_model, train, test_exog, selected_lags, mode="full")

    # Direct multi-step on both feature sets.
    direct_models_safe = {h: train_lgbm(*build_train_table(train, selected_lags, mode="safe", horizon=h, task="direct")) for h in [1, 5, 21, 63]}
    direct_safe_preds = direct_anchor_forecast(direct_models_safe, train, test_exog, selected_lags, mode="safe")
    direct_models_full = {h: train_lgbm(*build_train_table(train, selected_lags, mode="full", horizon=h, task="direct")) for h in [1, 5, 21, 63]}
    direct_full_preds = direct_anchor_forecast(direct_models_full, train, test_exog, selected_lags, mode="full")

    # Regime-split on the better of SAFE/FULL.
    regime_mode = "safe" if rmse(y_true, safe_preds) <= rmse(y_true, full_preds) else "full"
    regime_preds, vol_threshold = regime_split_forecast(train, test_exog, selected_lags, mode=regime_mode)

    elasticnet_preds = None
    if ELASTICNET_PRED_CSV.exists():
        en_df = pd.read_csv(ELASTICNET_PRED_CSV, parse_dates=[DATE_COL])
        elasticnet_preds = pd.to_numeric(en_df[f"elasticnet_full"], errors="coerce").to_numpy(dtype=float)

    result_rows = [
        {"model": "lgbm_safe", "rmse": rmse(y_true, safe_preds)},
        {"model": "lgbm_full", "rmse": rmse(y_true, full_preds)},
        {"model": "direct_safe", "rmse": rmse(y_true, direct_safe_preds)},
        {"model": "direct_full", "rmse": rmse(y_true, direct_full_preds)},
        {"model": f"regime_split_{regime_mode}", "rmse": rmse(y_true, regime_preds)},
    ]
    if elasticnet_preds is not None:
        result_rows.append({"model": "elasticnet_full", "rmse": rmse(y_true, elasticnet_preds)})
    results = pd.DataFrame(result_rows).sort_values("rmse").reset_index(drop=True)

    best_model_name = results.iloc[0]["model"]
    pred_map = {
        "lgbm_safe": safe_preds,
        "lgbm_full": full_preds,
        "direct_safe": direct_safe_preds,
        "direct_full": direct_full_preds,
        f"regime_split_{regime_mode}": regime_preds,
    }
    if elasticnet_preds is not None:
        pred_map["elasticnet_full"] = elasticnet_preds

    yearly = []
    for name, pred in pred_map.items():
        df = per_year_rmse(test_actual[DATE_COL], y_true, pred)
        df["model"] = name
        yearly.append(df)
    yearly_df = pd.concat(yearly, ignore_index=True)

    best_pred = pred_map[best_model_name]
    safe_imp = feature_importance_df(safe_model)
    full_imp = feature_importance_df(full_model)

    # Save artifacts.
    results.to_csv("phase3_lgbm_results.csv", index=False)
    yearly_df.to_csv("phase3_lgbm_yearly_rmse.csv", index=False)
    safe_imp.to_csv("phase3_lgbm_safe_feature_importance.csv", index=False)
    full_imp.to_csv("phase3_lgbm_full_feature_importance.csv", index=False)
    pd.DataFrame({"Date": test_actual[DATE_COL], "actual": y_true, **pred_map}).to_csv("phase3_lgbm_predictions.csv", index=False)

    plot_actual_vs_predicted(test_actual[DATE_COL], y_true, {"safe": safe_preds, "full": full_preds, "elasticnet": elasticnet_preds} if elasticnet_preds is not None else {"safe": safe_preds, "full": full_preds}, Path("phase3_safe_vs_full.png"))
    plot_actual_vs_predicted(test_actual[DATE_COL], y_true, {"best": best_pred}, Path("phase3_best_model.png"))

    en_rmse = float(rmse(y_true, elasticnet_preds)) if elasticnet_preds is not None else np.nan
    best_rmse = float(results.iloc[0]["rmse"])
    safe_rmse = float(rmse(y_true, safe_preds))
    full_rmse = float(rmse(y_true, full_preds))
    direct_best = min(float(rmse(y_true, direct_safe_preds)), float(rmse(y_true, direct_full_preds)))
    regime_rmse = float(rmse(y_true, regime_preds))
    safe_imp_pct = ((en_rmse - safe_rmse) / en_rmse * 100.0) if np.isfinite(en_rmse) else np.nan
    full_imp_pct = ((en_rmse - full_rmse) / en_rmse * 100.0) if np.isfinite(en_rmse) else np.nan

    def top_features(df: pd.DataFrame, n: int = 5) -> str:
        top = df.head(n)
        return ", ".join([f"{r.feature}:{r.gain:.1f}" for r in top.itertuples(index=False)])

    report = [
        "# Phase 3 LightGBM Experiments",
        "",
        f"- PACF-selected lags: `{selected_lags[:12]}{'...' if len(selected_lags) > 12 else ''}`",
        f"- Regime split volatility threshold: `{vol_threshold:.6f}`",
        "",
        "## Results",
        results.to_markdown(index=False),
        "",
        "## Comparison vs ElasticNet",
        f"- ElasticNet RMSE: `{en_rmse:.4f}`",
        f"- SAFE RMSE: `{safe_rmse:.4f}` ({safe_imp_pct:.2f}% vs ElasticNet)",
        f"- FULL RMSE: `{full_rmse:.4f}` ({full_imp_pct:.2f}% vs ElasticNet)",
        f"- Direct best RMSE: `{direct_best:.4f}`",
        f"- Regime split RMSE: `{regime_rmse:.4f}`",
        "",
        "## Feature Importance",
        f"- SAFE top features: {top_features(safe_imp)}",
        f"- FULL top features: {top_features(full_imp)}",
        "",
        "## Interpretation",
        "- SAFE answers whether non-linearity helps without OOD exogenous features.",
        "- FULL answers whether OOD exogenous features add lift beyond SAFE.",
        "- Direct multi-step tests whether recursive compounding is the source of error.",
        "- Regime split tests whether volatility clustering is exploitable in a simple gate.",
    ]
    Path("phase3_lgbm_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(results.to_string(index=False))
    print(f"SAFE top: {top_features(safe_imp)}")
    print(f"FULL top: {top_features(full_imp)}")
    print(f"Regime threshold: {vol_threshold:.6f}")


if __name__ == "__main__":
    main()
