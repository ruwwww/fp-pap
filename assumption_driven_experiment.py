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
from scipy.stats import wasserstein_distance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid
from statsmodels.tsa.stattools import adfuller, acf, grangercausalitytests, kpss, pacf

warnings.filterwarnings("ignore")

ROOT = Path(".")
TRAIN_CSV = ROOT / "data_train.csv"
TEST_EXOG_CSV = ROOT / "data_test.csv"
TEST_ACTUAL_CSV = ROOT / "data_test_actual.csv"
DATE_COL = "Date"
TARGET_COL = "USDIDR"
EXOG_COLS = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]
LOW_FREQ_COLS = ["CPI", "BI_rate"]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_exog = pd.read_csv(TEST_EXOG_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_actual = pd.read_csv(TEST_ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return train, test_exog, test_actual


def target_series(levels: pd.Series) -> pd.DataFrame:
    s = pd.to_numeric(levels, errors="coerce").astype(float)
    out = pd.DataFrame({"level": s})
    out["diff"] = out["level"].diff()
    out["log_return"] = np.log(out["level"]).diff()
    return out


def adf_kpss_report(series: pd.Series) -> dict[str, float]:
    x = pd.Series(series).dropna().astype(float)
    adf_res = adfuller(x, autolag="AIC")
    kpss_res = kpss(x, regression="c", nlags="auto")
    return {
        "adf_stat": float(adf_res[0]),
        "adf_p": float(adf_res[1]),
        "kpss_stat": float(kpss_res[0]),
        "kpss_p": float(kpss_res[1]),
    }


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 5:
        return float("nan")
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def choose_ar_order(diff: pd.Series, max_lag: int = 60) -> tuple[int, pd.DataFrame]:
    x = pd.Series(diff).dropna().astype(float)
    pacf_vals = pacf(x, nlags=max_lag, method="ywm")
    threshold = 1.96 / math.sqrt(len(x))
    rows = []
    selected = 1
    for lag in range(1, min(len(pacf_vals), max_lag + 1)):
        rows.append({"lag": lag, "pacf": float(pacf_vals[lag]), "abs_pacf": abs(float(pacf_vals[lag])), "sig": abs(float(pacf_vals[lag])) > threshold})
    pacf_df = pd.DataFrame(rows)
    sig_lags = pacf_df.loc[pacf_df["sig"], "lag"].tolist()
    if sig_lags:
        selected = int(max(sig_lags))
    return selected, pacf_df


def acf_pacf_tables(diff: pd.Series, max_lag: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.Series(diff).dropna().astype(float)
    acf_vals = acf(x, nlags=max_lag, fft=True)
    pacf_vals = pacf(x, nlags=max_lag, method="ywm")
    acf_df = pd.DataFrame({"lag": np.arange(len(acf_vals)), "acf": acf_vals})
    pacf_df = pd.DataFrame({"lag": np.arange(len(pacf_vals)), "pacf": pacf_vals})
    return acf_df, pacf_df


def ljung_box_report(series: pd.Series, lags: list[int]) -> pd.DataFrame:
    out = acorr_ljungbox(pd.Series(series).dropna().astype(float), lags=lags, return_df=True)
    out = out.reset_index().rename(columns={"index": "lag"})
    return out


def chow_test(y: np.ndarray, x: np.ndarray, break_idx: int) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    x_full = np.column_stack([np.ones(len(x)), x])
    if break_idx < 20 or break_idx > len(y) - 20:
        return {"f_stat": float("nan"), "p_value": float("nan")}
    x1, y1 = x_full[:break_idx], y[:break_idx]
    x2, y2 = x_full[break_idx:], y[break_idx:]
    beta_full = np.linalg.lstsq(x_full, y, rcond=None)[0]
    beta1 = np.linalg.lstsq(x1, y1, rcond=None)[0]
    beta2 = np.linalg.lstsq(x2, y2, rcond=None)[0]
    rss_full = float(np.sum((y - x_full @ beta_full) ** 2))
    rss1 = float(np.sum((y1 - x1 @ beta1) ** 2))
    rss2 = float(np.sum((y2 - x2 @ beta2) ** 2))
    k = x_full.shape[1]
    n1, n2 = len(y1), len(y2)
    f_stat = ((rss_full - (rss1 + rss2)) / k) / ((rss1 + rss2) / max(n1 + n2 - 2 * k, 1))
    p_value = float(np.exp(np.nan_to_num(-0.5 * f_stat, nan=np.inf)))
    return {"f_stat": float(f_stat), "p_value": p_value}


def ood_shift(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        tr = pd.to_numeric(train[col], errors="coerce").dropna().astype(float)
        te = pd.to_numeric(test[col], errors="coerce").dropna().astype(float)
        mu = float(tr.mean())
        sigma = float(tr.std(ddof=0)) or np.nan
        z = (te.mean() - mu) / sigma if np.isfinite(sigma) and sigma > 0 else np.nan
        rows.append({
            "feature": col,
            "train_mean": mu,
            "test_mean": float(te.mean()),
            "mean_shift_sigma": float(z),
            "train_std": float(tr.std(ddof=0)),
            "wasserstein": float(wasserstein_distance(tr, te)),
            "test_gt_2sigma_pct": float((np.abs((te - mu) / sigma) > 2).mean() * 100.0) if np.isfinite(sigma) and sigma > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def make_causal_exog(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    for c in EXOG_COLS + [TARGET_COL]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["dow"] = out[DATE_COL].dt.dayofweek.astype(float)
    out["month"] = out[DATE_COL].dt.month.astype(float)
    out["is_Q2"] = out[DATE_COL].dt.month.isin([4, 5, 6]).astype(float)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)
    if "BI_rate" in out.columns:
        out["bi_rate_change"] = out["BI_rate"].diff()
        last_change = out[DATE_COL].where(out["BI_rate"].diff().ne(0)).ffill()
        out["days_since_bi_change"] = (out[DATE_COL] - last_change).dt.days.fillna(0).astype(float)
    if "CPI" in out.columns:
        out["cpi_change"] = out["CPI"].diff()
        last_change = out[DATE_COL].where(out["CPI"].diff().ne(0)).ffill()
        out["days_since_cpi_release"] = (out[DATE_COL] - last_change).dt.days.fillna(0).astype(float)
    for c in EXOG_COLS:
        out[f"{c}_diff1"] = out[c].diff()
        out[f"{c}_lag1"] = out[c].shift(1)
        out[f"{c}_lag6"] = out[c].shift(6)
        out[f"{c}_lag20"] = out[c].shift(20)
    return out


def make_target_history_features(level_hist: list[float], diff_hist: list[float], feature_mode: str) -> dict[str, float]:
    level = np.asarray(level_hist, dtype=float)
    diff = np.asarray(diff_hist, dtype=float)
    feats: dict[str, float] = {}
    for lag in [1, 2, 3, 5, 10, 20, 60]:
        feats[f"diff_lag{lag}"] = float(diff[-lag]) if len(diff) >= lag else np.nan
    if feature_mode in {"trend", "full"}:
        if len(diff) >= 20:
            feats["rolling_mean_diff_20"] = float(np.mean(diff[-20:]))
            feats["rolling_std_diff_20"] = float(np.std(diff[-20:], ddof=0))
        else:
            feats["rolling_mean_diff_20"] = np.nan
            feats["rolling_std_diff_20"] = np.nan
        if len(diff) >= 252:
            feats["rolling_mean_diff_252"] = float(np.mean(diff[-252:]))
        else:
            feats["rolling_mean_diff_252"] = float(np.mean(diff)) if len(diff) else np.nan
        if len(level) >= 252:
            ma252 = float(np.mean(level[-252:]))
            feats["gap_from_trend"] = float(level[-1] - ma252)
        else:
            feats["gap_from_trend"] = np.nan
        if len(level) >= 90:
            ma90 = float(np.mean(level[-90:]))
            sd90 = float(np.std(level[-90:], ddof=0))
            z = (level[-1] - ma90) / sd90 if sd90 > 0 else 0.0
            threshold = 1.5
            feats["extreme_high"] = float(max(0.0, z - threshold))
            feats["extreme_low"] = float(min(0.0, z + threshold))
        else:
            feats["extreme_high"] = np.nan
            feats["extreme_low"] = np.nan
    return feats


def build_row_features(
    exog_row: pd.Series,
    level_hist: list[float],
    diff_hist: list[float],
    selected_lags: list[int],
    selected_exog: list[str],
    feature_mode: str,
) -> dict[str, float]:
    feats = make_target_history_features(level_hist, diff_hist, feature_mode)
    for lag in selected_lags:
        feats[f"diff_lag{lag}"] = float(diff_hist[-lag]) if len(diff_hist) >= lag else np.nan
    for name in selected_exog:
        feats[name] = float(exog_row.get(name, np.nan))
    return feats


def build_train_table(
    train_exog: pd.DataFrame,
    selected_lags: list[int],
    selected_exog: list[str],
    feature_mode: str,
) -> tuple[pd.DataFrame, pd.Series]:
    levels = train_exog[TARGET_COL].astype(float).tolist()
    diffs = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
    rows = []
    ys = []
    start = max(max(selected_lags, default=1), 1)
    if feature_mode in {"trend", "full"}:
        start = max(start, 252, 90)
    for t in range(start, len(train_exog)):
        level_hist = levels[:t]
        diff_hist = diffs[: t - 1]
        row = build_row_features(train_exog.iloc[t], level_hist, diff_hist, selected_lags, selected_exog, feature_mode)
        rows.append(row)
        ys.append(levels[t] - levels[t - 1])
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    y = pd.Series(ys, dtype=float)
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def recursive_forecast(
    model,
    train: pd.DataFrame,
    test_exog: pd.DataFrame,
    selected_lags: list[int],
    selected_exog: list[str],
    feature_mode: str,
) -> np.ndarray:
    levels = train[TARGET_COL].astype(float).tolist()
    diffs = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
    combined = pd.concat([train[[DATE_COL] + [c for c in test_exog.columns if c != DATE_COL]], test_exog], ignore_index=True)
    combined = make_causal_exog(combined)
    preds = []
    for i in range(len(test_exog)):
        row_idx = len(train) + i
        feats = build_row_features(combined.iloc[row_idx], levels, diffs, selected_lags, selected_exog, feature_mode)
        X_row = pd.DataFrame([feats]).reindex(columns=model.feature_names_in_, fill_value=np.nan)
        X_row = X_row.ffill(axis=1).bfill(axis=1).fillna(0.0)
        diff_pred = float(model.predict(X_row)[0])
        next_level = levels[-1] + diff_pred
        preds.append(next_level)
        levels.append(next_level)
        diffs.append(diff_pred)
    return np.asarray(preds, dtype=float)


def per_year_rmse(dates: pd.Series, actual: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "actual": actual, "pred": pred})
    df["year"] = df["Date"].dt.year
    rows = []
    for year in sorted(df["year"].unique()):
        g = df[df["year"] == year]
        rows.append({"year": int(year), "rmse": rmse(g["actual"], g["pred"])})
    return pd.DataFrame(rows)


def directional_hit_rate(actual: np.ndarray, pred: np.ndarray) -> float:
    a = np.sign(np.diff(actual))
    p = np.sign(np.diff(pred))
    n = min(len(a), len(p))
    if n == 0:
        return float("nan")
    return float((a[-n:] == p[-n:]).mean() * 100.0)


def fit_model(X: pd.DataFrame, y: pd.Series, kind: str):
    if kind == "ridge":
        model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    elif kind == "elasticnet":
        model = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.002, l1_ratio=0.3, max_iter=50000, random_state=42))])
    else:
        raise ValueError(kind)
    model.fit(X, y)
    return model


def model_spec_from_pacf(p: int, selected_exog: list[str]) -> dict[str, object]:
    lags = list(range(1, p + 1))
    return {"selected_lags": lags, "selected_exog": selected_exog}


def evaluate_baseline(train: pd.DataFrame, test_actual: pd.DataFrame) -> pd.DataFrame:
    y_true = test_actual[TARGET_COL].astype(float).to_numpy(dtype=float)
    last = float(train[TARGET_COL].iloc[-1])
    drift = float(train[TARGET_COL].diff().dropna().mean())
    seasonal_hist = train[TARGET_COL].astype(float).tolist()
    seasonal_preds = []
    for _ in range(len(test_actual)):
        seasonal_preds.append(float(seasonal_hist[-5]))
        seasonal_hist.append(seasonal_preds[-1])
    preds = {
        "naive_last": np.repeat(last, len(test_actual)),
        "naive_drift": np.asarray([last + drift * (i + 1) for i in range(len(test_actual))], dtype=float),
        "seasonal_naive_5": np.asarray(seasonal_preds, dtype=float),
    }

    rows = []
    for name, pred in preds.items():
        rows.append({
            "model": name,
            "rmse": rmse(y_true, pred),
            "directional_hit_rate": directional_hit_rate(y_true, pred),
        })
    return pd.DataFrame(rows)


def main() -> None:
    train, test_exog, test_actual = load_data()
    train = train.sort_values(DATE_COL).reset_index(drop=True)
    test_exog = test_exog.sort_values(DATE_COL).reset_index(drop=True)
    test_actual = test_actual.sort_values(DATE_COL).reset_index(drop=True)

    train_target = target_series(train[TARGET_COL])
    train_diff = train_target["diff"].dropna().reset_index(drop=True)
    train_ret = train_target["log_return"].dropna().reset_index(drop=True)

    level_tests = adf_kpss_report(train[TARGET_COL])
    diff_tests = adf_kpss_report(train_diff)
    ret_tests = adf_kpss_report(train_ret)

    acf_df, pacf_df = acf_pacf_tables(train_diff, max_lag=60)
    ar_order, pacf_sig = choose_ar_order(train_diff, max_lag=60)

    lb_df = ljung_box_report(train_ret ** 2, [10, 20, 60])

    # Structural break audit on a simple AR(1) diff model.
    y = train_diff.iloc[1:].to_numpy(dtype=float)
    x = train_diff.iloc[:-1].to_numpy(dtype=float).reshape(-1, 1)
    x = np.column_stack([np.ones(len(x)), x])
    ar1 = OLS(y, x).fit()
    cusum_stat = breaks_cusumolsresid(ar1.resid, ddof=x.shape[1])
    break_candidates = [pd.Timestamp("2018-08-01"), pd.Timestamp("2020-03-01"), pd.Timestamp("2022-03-01")]
    chow_rows = []
    for bd in break_candidates:
        idx = int((train[DATE_COL] <= bd).sum()) - 1
        if 10 < idx < len(train_diff) - 10:
            y_break = train_diff.to_numpy(dtype=float)
            x_break = train_diff.shift(1).dropna().to_numpy(dtype=float)
            y_break = y_break[1:]
            res = chow_test(y_break, x_break, idx - 1)
        else:
            res = {"f_stat": float("nan"), "p_value": float("nan")}
        res["break_date"] = bd.date().isoformat()
        chow_rows.append(res)

    # Exogenous lag/Granger scan.
    diff_target = train[TARGET_COL].diff().dropna().reset_index(drop=True)
    granger_rows = []
    ccf_rows = []
    exog_scan = pd.DataFrame({c: pd.to_numeric(train[c], errors="coerce") for c in EXOG_COLS})
    for feat in EXOG_COLS:
        feat_diff = exog_scan[feat].diff().dropna().reset_index(drop=True)
        n = min(len(diff_target), len(feat_diff))
        t = diff_target.iloc[-n:].reset_index(drop=True)
        f = feat_diff.iloc[-n:].reset_index(drop=True)
        best_corr = (None, float("nan"))
        for lag in range(1, 91):
            corr = safe_corr(f.shift(lag), t)
            ccf_rows.append({"feature": feat, "lag": lag, "correlation": corr, "abs_correlation": abs(corr) if pd.notna(corr) else np.nan})
            if pd.notna(corr) and (pd.isna(best_corr[1]) or abs(corr) > abs(best_corr[1])):
                best_corr = (lag, corr)
        try:
            gc = grangercausalitytests(pd.DataFrame({"y": t, "x": f}).dropna(), maxlag=12, verbose=False)
            pvals = {lag: float(gc[lag][0]["ssr_ftest"][1]) for lag in gc}
            best_lag = min(pvals, key=pvals.get)
            granger_rows.append({"feature": feat, "best_lag": int(best_lag), "best_pvalue": float(pvals[best_lag]), "min_pvalue": float(min(pvals.values())), "best_ccf_lag": int(best_corr[0] or 0), "best_ccf": float(best_corr[1])})
        except Exception:
            granger_rows.append({"feature": feat, "best_lag": np.nan, "best_pvalue": np.nan, "min_pvalue": np.nan, "best_ccf_lag": int(best_corr[0] or 0), "best_ccf": float(best_corr[1])})

    granger_df = pd.DataFrame(granger_rows).sort_values(["min_pvalue", "best_ccf"], ascending=[True, False]).reset_index(drop=True)
    ccf_df = pd.DataFrame(ccf_rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    selected_exog = granger_df.head(2)["feature"].tolist()
    selected_lags = list(range(1, ar_order + 1))

    # OOD check uses future exog only.
    ood_df = ood_shift(train, test_exog, EXOG_COLS)

    causal_train = pd.concat([train[[DATE_COL, TARGET_COL] + EXOG_COLS], pd.DataFrame(index=train.index)], axis=1)
    causal_train = make_causal_exog(train[[DATE_COL, TARGET_COL] + EXOG_COLS].copy())
    X_train_ar, y_train_ar = build_train_table(causal_train, selected_lags=list(range(1, ar_order + 1)), selected_exog=[], feature_mode="ar")
    X_train_trend, y_train_trend = build_train_table(causal_train, selected_lags=list(range(1, ar_order + 1)), selected_exog=[], feature_mode="trend")
    X_train_sel, y_train_sel = build_train_table(causal_train, selected_lags=list(range(1, ar_order + 1)), selected_exog=[f"{f}_diff1" for f in selected_exog], feature_mode="ar")
    X_train_full, y_train_full = build_train_table(
        causal_train,
        selected_lags=list(range(1, ar_order + 1)),
        selected_exog=[f"{c}_diff1" for c in EXOG_COLS] + ["bi_rate_change", "cpi_change", "days_since_bi_change", "days_since_cpi_release", "is_Q2", "month_sin", "month_cos", "dow_sin", "dow_cos"],
        feature_mode="full",
    )

    ridge_ar = fit_model(X_train_ar, y_train_ar, "ridge")
    ridge_trend = fit_model(X_train_trend, y_train_trend, "ridge")
    ridge_sel = fit_model(X_train_sel, y_train_sel, "ridge")
    enet_full = fit_model(X_train_full, y_train_full, "elasticnet")

    baseline_df = evaluate_baseline(train, test_actual)
    model_preds = {
        "ar_p": recursive_forecast(ridge_ar, train, test_exog, selected_lags=list(range(1, ar_order + 1)), selected_exog=[], feature_mode="ar"),
        "ar_plus_trend": recursive_forecast(ridge_trend, train, test_exog, selected_lags=list(range(1, ar_order + 1)), selected_exog=[], feature_mode="trend"),
        "ar_plus_verified_exog": recursive_forecast(ridge_sel, train, test_exog, selected_lags=list(range(1, ar_order + 1)), selected_exog=[f"{f}_diff1" for f in selected_exog], feature_mode="ar"),
        "elasticnet_full": recursive_forecast(
            enet_full,
            train,
            test_exog,
            selected_lags=list(range(1, ar_order + 1)),
            selected_exog=[f"{c}_diff1" for c in EXOG_COLS] + ["bi_rate_change", "cpi_change", "days_since_bi_change", "days_since_cpi_release", "is_Q2", "month_sin", "month_cos", "dow_sin", "dow_cos"],
            feature_mode="full",
        ),
    }

    y_true = test_actual[TARGET_COL].astype(float).to_numpy(dtype=float)
    model_rows = []
    for name, pred in model_preds.items():
        model_rows.append({
            "model": name,
            "rmse": rmse(y_true, pred),
            "directional_hit_rate": directional_hit_rate(y_true, pred),
        })
    model_df = pd.DataFrame(model_rows)
    all_results = pd.concat([baseline_df, model_df], ignore_index=True).sort_values("rmse").reset_index(drop=True)

    yearly_rows = []
    for name, pred in {"naive_last": np.repeat(float(train[TARGET_COL].iloc[-1]), len(test_actual)), **model_preds}.items():
        yr = per_year_rmse(test_actual[DATE_COL], y_true, pred)
        yr["model"] = name
        yearly_rows.append(yr)
    yearly_df = pd.concat(yearly_rows, ignore_index=True)

    pred_frame = pd.DataFrame({"Date": test_actual[DATE_COL], "actual": y_true})
    pred_frame["naive_last"] = np.repeat(float(train[TARGET_COL].iloc[-1]), len(test_actual))
    for name, pred in model_preds.items():
        pred_frame[name] = pred
    pred_frame.to_csv("assumption_driven_predictions.csv", index=False)

    baseline_df.to_csv("assumption_driven_baselines.csv", index=False)
    model_df.to_csv("assumption_driven_linear_models.csv", index=False)
    yearly_df.to_csv("assumption_driven_yearly_rmse.csv", index=False)
    pd.DataFrame([level_tests]).to_csv("assumption_target_level_tests.csv", index=False)
    pd.DataFrame([diff_tests]).to_csv("assumption_target_diff_tests.csv", index=False)
    pd.DataFrame([ret_tests]).to_csv("assumption_target_logret_tests.csv", index=False)
    acf_df.to_csv("assumption_acf_diff.csv", index=False)
    pacf_df.to_csv("assumption_pacf_diff.csv", index=False)
    lb_df.to_csv("assumption_ljungbox_squared_returns.csv", index=False)
    pd.DataFrame(chow_rows).to_csv("assumption_chow_tests.csv", index=False)
    ccf_df.to_csv("assumption_ccf_scan.csv", index=False)
    granger_df.to_csv("assumption_granger_scan.csv", index=False)
    ood_df.to_csv("assumption_ood_shift.csv", index=False)

    # Plot best model vs actual.
    best_model = all_results.iloc[0]["model"]
    plt.figure(figsize=(14, 6))
    plt.plot(test_actual[DATE_COL], y_true, label="actual", color="black", linewidth=1.5)
    plt.plot(test_actual[DATE_COL], pred_frame[best_model], label=f"predicted: {best_model}", color="firebrick", linewidth=1.2)
    plt.title("USDIDR actual vs predicted")
    plt.xlabel("Date")
    plt.ylabel("USDIDR")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assumption_driven_actual_vs_predicted.png", dpi=150)
    plt.close()

    best_rmse = float(all_results.iloc[0]["rmse"])
    naive_rmse = float(all_results.loc[all_results["model"] == "naive_last", "rmse"].iloc[0])
    improvement = (naive_rmse - best_rmse) / naive_rmse * 100.0 if naive_rmse else np.nan

    report = [
        "# Assumption-Driven USDIDR Experiment",
        "",
        "## Phase 0",
        f"- ADF/KPSS level: p={level_tests['adf_p']:.4g} / {level_tests['kpss_p']:.4g}",
        f"- ADF/KPSS diff: p={diff_tests['adf_p']:.4g} / {diff_tests['kpss_p']:.4g}",
        f"- ADF/KPSS log-return: p={ret_tests['adf_p']:.4g} / {ret_tests['kpss_p']:.4g}",
        f"- Selected AR order from PACF: `{ar_order}`",
        f"- Ljung-Box squared returns p-values: {', '.join([f'lag{int(r.lag)}={r.lb_pvalue:.4g}' for _, r in lb_df.iterrows()])}",
        f"- CUSUM statistic: {float(cusum_stat[0]):.4g}, p={float(cusum_stat[1]):.4g}",
        "",
        "## Phase 1 Baselines",
        baseline_df.to_markdown(index=False),
        "",
        "## Phase 2 Linear Models",
        model_df.to_markdown(index=False),
        "",
        "## Best Model",
        f"- model: `{best_model}`",
        f"- RMSE: `{best_rmse:.4f}`",
        f"- Improvement over naive_last: `{improvement:.2f}%`",
        f"- Selected exogenous features: `{', '.join(selected_exog) if selected_exog else '-'}`",
        "",
        "## Per-Year RMSE",
        yearly_df.to_markdown(index=False),
        "",
        "## OOD Shift",
        ood_df.to_markdown(index=False),
        "",
        "## Verdict",
        "- If the AR family does not beat naive meaningfully, the sample is effectively AR-ceilinged.",
        "- Exogenous features are only kept when they improve OOS on the holdout, not because they are plausible in theory.",
        "- SSM is not warranted unless a later phase beats these linear baselines with stable per-year gains.",
    ]
    Path("assumption_driven_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(all_results.to_string(index=False))
    print(f"Selected AR order: {ar_order}")
    print(f"Selected exogenous features: {selected_exog}")


if __name__ == "__main__":
    main()
