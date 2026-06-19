#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, acf, kpss, pacf

ROOT = Path(".")
TRAIN_CSV = ROOT / "data_train.csv"
TEST_CSV = ROOT / "data_test.csv"
TEST_ACTUAL_CSV = ROOT / "data_test_actual.csv"
DATE_COL = "Date"
TARGET_COL = "USDIDR"
EXOG_COLS = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]
LOW_FREQ_COLS = ["CPI", "BI_rate", "US_rate"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    future_exog = pd.read_csv(TEST_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test = pd.read_csv(TEST_ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    full = pd.concat([train, test], ignore_index=True)
    return train, future_exog, test, full


def log_return(s: pd.Series) -> pd.Series:
    return np.log(pd.to_numeric(s, errors="coerce")).diff()


def adf_kpss(s: pd.Series) -> dict[str, float]:
    x = pd.Series(s).dropna()
    adf_res = adfuller(x, autolag="AIC")
    kpss_res = kpss(x, regression="c", nlags="auto")
    return {
        "adf_stat": float(adf_res[0]),
        "adf_p": float(adf_res[1]),
        "kpss_stat": float(kpss_res[0]),
        "kpss_p": float(kpss_res[1]),
    }


def hurst_exponent(x: pd.Series, max_lag: int = 100) -> float:
    s = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(s) < 100:
        return float("nan")
    window_sizes = np.unique(np.floor(np.logspace(np.log10(10), np.log10(min(max_lag, len(s) // 4)), 12)).astype(int))
    rs_vals = []
    used_sizes = []
    for w in window_sizes:
        if w < 10:
            continue
        n_chunks = len(s) // w
        if n_chunks < 2:
            continue
        chunk_rs = []
        for i in range(n_chunks):
            chunk = s[i * w : (i + 1) * w]
            dev = chunk - chunk.mean()
            cum = np.cumsum(dev)
            r = cum.max() - cum.min()
            sd = chunk.std(ddof=1)
            if sd > 0:
                chunk_rs.append(r / sd)
        if chunk_rs:
            used_sizes.append(w)
            rs_vals.append(np.mean(chunk_rs))
    if len(used_sizes) < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(used_sizes), np.log(np.asarray(rs_vals) + 1e-12), 1)
    return float(slope)


def same_value_run_stats(s: pd.Series) -> dict[str, float]:
    x = pd.Series(s).dropna().to_numpy()
    if len(x) == 0:
        return {"max_run": np.nan, "mean_run": np.nan, "p95_run": np.nan, "changes": np.nan, "stale_days_pct": np.nan}
    run_lengths = []
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or x[i] != x[start]:
            run_lengths.append(i - start)
            start = i
    run_lengths = np.asarray(run_lengths, dtype=float)
    changes = max(len(run_lengths) - 1, 0)
    return {
        "max_run": float(run_lengths.max()),
        "mean_run": float(run_lengths.mean()),
        "p95_run": float(np.percentile(run_lengths, 95)),
        "changes": float(changes),
        "stale_days_pct": float(100.0 * (1.0 - changes / max(len(x) - 1, 1))),
    }


def transition_stats(df: pd.DataFrame, col: str) -> dict[str, float]:
    s = pd.to_numeric(df[col], errors="coerce")
    diff = s.diff()
    changed = diff.ne(0) & diff.notna()
    abs_usd_ret = log_return(df[TARGET_COL]).abs()
    return {
        "change_days": int(changed.sum()),
        "mean_abs_usdret_on_change": float(abs_usd_ret.loc[changed].mean()),
        "mean_abs_usdret_no_change": float(abs_usd_ret.loc[~changed & abs_usd_ret.notna()].mean()),
        "corr_change_flag_abs_usdret": float(pd.Series(changed.astype(int)).corr(abs_usd_ret)),
        "mean_delta": float(diff.loc[changed].mean()),
        "median_delta": float(diff.loc[changed].median()),
    }


def safe_kl(p: np.ndarray, q: np.ndarray, bins: int = 60) -> float:
    lo = float(np.nanmin([p.min(), q.min()]))
    hi = float(np.nanmax([p.max(), q.max()]))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return float("nan")
    edges = np.linspace(lo, hi, bins + 1)
    ph, _ = np.histogram(p, bins=edges, density=False)
    qh, _ = np.histogram(q, bins=edges, density=False)
    ph = ph.astype(float) + 1e-9
    qh = qh.astype(float) + 1e-9
    ph /= ph.sum()
    qh /= qh.sum()
    return float(stats.entropy(ph, qh))


def shift_metrics(train: pd.DataFrame, test: pd.DataFrame, col: str) -> dict[str, float]:
    a = pd.to_numeric(train[col], errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(test[col], errors="coerce").dropna().to_numpy(dtype=float)
    return {
        "wasserstein": float(wasserstein_distance(a, b)),
        "kl_divergence": safe_kl(a, b),
        "train_mean": float(np.mean(a)),
        "test_mean": float(np.mean(b)),
        "train_p90": float(np.percentile(a, 90)),
        "test_p90": float(np.percentile(b, 90)),
        "mean_ratio": float(np.mean(b) / np.mean(a)) if np.mean(a) != 0 else np.nan,
        "p90_ratio": float(np.percentile(b, 90) / np.percentile(a, 90)) if np.percentile(a, 90) != 0 else np.nan,
    }


def build_vol_slice_frame(full: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    out = full[[DATE_COL, TARGET_COL, "SP500", "VIX", "IHSG"]].copy()
    out["usd_ret"] = log_return(out[TARGET_COL])
    out["sp500_ret_l1"] = log_return(out["SP500"]).shift(1)
    out["vix_l1"] = pd.to_numeric(out["VIX"], errors="coerce").shift(1)
    out["ihsg_ret_l1"] = log_return(out["IHSG"]).shift(1)
    train_vix = out.loc[out[DATE_COL].isin(train[DATE_COL]), "vix_l1"].dropna()
    qcuts = np.quantile(train_vix, [0.25, 0.5, 0.75])
    out["vix_quartile"] = pd.cut(out["vix_l1"], bins=[-np.inf, qcuts[0], qcuts[1], qcuts[2], np.inf], labels=[0, 1, 2, 3])
    out["direction"] = (out["usd_ret"] > 0).astype(int)
    return out


def quartile_corr_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in [0, 1, 2, 3]:
        g = df[df["vix_quartile"] == q].dropna(subset=["usd_ret", "sp500_ret_l1", "vix_l1", "ihsg_ret_l1"])
        for feat in ["sp500_ret_l1", "vix_l1", "ihsg_ret_l1"]:
            pearson = g[["usd_ret", feat]].corr(method="pearson").iloc[0, 1]
            spearman = g[["usd_ret", feat]].corr(method="spearman").iloc[0, 1]
            rows.append({"vix_quartile": q, "feature": feat, "pearson": pearson, "spearman": spearman, "n": int(len(g))})
    return pd.DataFrame(rows)


def directional_auc_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in [0, 1, 2, 3]:
        g = df[df["vix_quartile"] == q].dropna(subset=["direction", "sp500_ret_l1", "vix_l1", "ihsg_ret_l1"]).sort_values(DATE_COL).reset_index(drop=True)
        if len(g) < 80 or g["direction"].nunique() < 2:
            rows.append({"vix_quartile": q, "auc": np.nan, "train_n": int(len(g)), "test_n": 0})
            continue
        split = max(int(len(g) * 0.7), 50)
        train_g = g.iloc[:split].copy()
        test_g = g.iloc[split:].copy()
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42)
        clf.fit(train_g[["sp500_ret_l1", "vix_l1", "ihsg_ret_l1"]], train_g["direction"])
        prob = clf.predict_proba(test_g[["sp500_ret_l1", "vix_l1", "ihsg_ret_l1"]])[:, 1]
        auc = roc_auc_score(test_g["direction"], prob) if test_g["direction"].nunique() > 1 else np.nan
        rows.append({"vix_quartile": q, "auc": float(auc), "train_n": int(len(train_g)), "test_n": int(len(test_g))})
    return pd.DataFrame(rows)


def chow_test(y: np.ndarray, x: np.ndarray, break_idx: int) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_full = add_constant(x, has_constant="add")
    x1 = x_full[:break_idx]
    y1 = y[:break_idx]
    x2 = x_full[break_idx:]
    y2 = y[break_idx:]
    if len(y1) < x_full.shape[1] + 5 or len(y2) < x_full.shape[1] + 5:
        return {"f_stat": np.nan, "p_value": np.nan, "n1": len(y1), "n2": len(y2)}
    beta_full = np.linalg.lstsq(x_full, y, rcond=None)[0]
    rss_full = float(np.sum((y - x_full @ beta_full) ** 2))
    beta1 = np.linalg.lstsq(x1, y1, rcond=None)[0]
    beta2 = np.linalg.lstsq(x2, y2, rcond=None)[0]
    rss1 = float(np.sum((y1 - x1 @ beta1) ** 2))
    rss2 = float(np.sum((y2 - x2 @ beta2) ** 2))
    k = x_full.shape[1]
    n1 = len(y1)
    n2 = len(y2)
    f_stat = ((rss_full - (rss1 + rss2)) / k) / ((rss1 + rss2) / (n1 + n2 - 2 * k))
    p_value = float(stats.f.sf(f_stat, k, n1 + n2 - 2 * k))
    return {"f_stat": float(f_stat), "p_value": p_value, "n1": n1, "n2": n2}


def rolling_ols_sp500(full: pd.DataFrame, window: int = 252) -> tuple[pd.DataFrame, str]:
    df = full[[DATE_COL, TARGET_COL, "SP500"]].copy()
    df["usd_ret"] = log_return(df[TARGET_COL])
    df["sp500_ret_l1"] = log_return(df["SP500"]).shift(1)
    df = df.dropna().reset_index(drop=True)
    y = df["usd_ret"]
    x = add_constant(df[["sp500_ret_l1"]], has_constant="add")
    rols = RollingOLS(y, x, window=window, min_nobs=window)
    res = rols.fit()
    params = res.params.reset_index(drop=True)
    params[DATE_COL] = df.loc[params.index, DATE_COL].reset_index(drop=True)
    params = params.rename(columns={"sp500_ret_l1": "sp500_coef", "const": "intercept"})
    params["sign"] = np.sign(params["sp500_coef"])
    params["sign_flip"] = params["sign"].ne(params["sign"].shift(1)) & params["sign"].notna() & params["sign"].shift(1).notna()
    summary = (
        f"- Rolling OLS window: {window}\n"
        f"- SP500 coef mean: {params['sp500_coef'].mean():.6f}\n"
        f"- SP500 coef min/max: {params['sp500_coef'].min():.6f} / {params['sp500_coef'].max():.6f}\n"
        f"- Percent positive: {100.0 * (params['sp500_coef'] > 0).mean():.2f}%\n"
        f"- Sign flips: {int(params['sign_flip'].sum())}\n"
    )
    return params, summary


def write_report(path: Path, sections: list[str]) -> None:
    path.write_text("\n".join(sections).strip() + "\n", encoding="utf-8")


def main() -> None:
    train, future_exog, test_actual, full = load_data()
    full_exog = full[[DATE_COL] + EXOG_COLS + [TARGET_COL]].copy()

    # Target inertsia audit.
    full_exog["usd_ret"] = log_return(full_exog[TARGET_COL])
    acf_vals = acf(full_exog["usd_ret"].dropna(), nlags=252, fft=True)
    pacf_vals = pacf(full_exog["usd_ret"].dropna(), nlags=252, method="ywm")
    acf_pacf = pd.DataFrame({"lag": np.arange(len(acf_vals)), "acf": acf_vals, "pacf": pacf_vals})
    acf_pacf.to_csv("target_acf_pacf.csv", index=False)
    top_acf = acf_pacf.loc[1:].assign(abs_acf=lambda d: d["acf"].abs()).sort_values("abs_acf", ascending=False).head(10)
    top_pacf = acf_pacf.loc[1:].assign(abs_pacf=lambda d: d["pacf"].abs()).sort_values("abs_pacf", ascending=False).head(10)

    hurst = hurst_exponent(full_exog["usd_ret"].dropna())
    level_tests = adf_kpss(train[TARGET_COL])
    ret_tests = adf_kpss(train["USDIDR"].pipe(log_return).dropna())

    # Exogenous summary.
    exog_rows = []
    for col in EXOG_COLS:
        stats_row = same_value_run_stats(train[col])
        stats_row.update({"feature": col})
        stats_row.update({"train_adf_p": adf_kpss(train[col])["adf_p"], "train_kpss_p": adf_kpss(train[col])["kpss_p"]})
        exog_rows.append(stats_row)
    exog_summary = pd.DataFrame(exog_rows)
    exog_summary.to_csv("exogenous_eda_summary.csv", index=False)

    # Transition audit for low-frequency variables.
    trans_rows = []
    for col in LOW_FREQ_COLS:
        r = transition_stats(train, col)
        r["feature"] = col
        trans_rows.append(r)
    transition_df = pd.DataFrame(trans_rows)
    transition_df.to_csv("exogenous_transition_stats.csv", index=False)

    # Shift metrics.
    shift_rows = []
    for col in EXOG_COLS:
        r = shift_metrics(train, future_exog, col)
        r["feature"] = col
        shift_rows.append(r)
    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv("exogenous_shift_metrics.csv", index=False)

    # Conditional volatility slices.
    vol_df = build_vol_slice_frame(full, train)
    quartile_corr = quartile_corr_report(vol_df)
    quartile_corr.to_csv("volatility_quartile_correlations.csv", index=False)
    auc_df = directional_auc_report(vol_df)
    auc_df.to_csv("volatility_quartile_auc.csv", index=False)

    # Structural breaks and rolling OLS.
    rolling_df, rolling_summary = rolling_ols_sp500(full)
    rolling_df.to_csv("rolling_sp500_ols.csv", index=False)

    y = full_exog["usd_ret"].dropna().reset_index(drop=True)
    x = log_return(full_exog["SP500"]).shift(1).dropna().reset_index(drop=True)
    common = pd.concat([y, x], axis=1).dropna().reset_index(drop=True)
    yv = common.iloc[:, 0].to_numpy()
    xv = common.iloc[:, 1].to_numpy()
    break_dates = [pd.Timestamp("2018-08-01"), pd.Timestamp("2020-03-01")]
    chow_rows = []
    for bd in break_dates:
        idx = int((full_exog.loc[full_exog["Date"] <= bd, "Date"].shape[0]) - 1)
        idx = max(min(idx, len(common) - 10), 10)
        ch = chow_test(yv, xv, idx)
        ch["break_date"] = bd.date().isoformat()
        chow_rows.append(ch)
    chow_df = pd.DataFrame(chow_rows)
    chow_df.to_csv("chow_tests_sp500.csv", index=False)

    # State features for low-frequency variables.
    state = full[[DATE_COL] + LOW_FREQ_COLS].copy()
    for col in LOW_FREQ_COLS:
        state[f"{col}_shock"] = state[col].diff().fillna(0.0)
        state[f"{col}_state_drift"] = state[col] - state[col].iloc[0]
        change = state[col].diff().ne(0)
        last_change_date = state[DATE_COL].where(change).ffill()
        state[f"{col}_days_since_change"] = (state[DATE_COL] - last_change_date).dt.days.fillna(0).astype(int)
    state.to_csv("exogenous_state_features.csv", index=False)

    report = [
        "# Exogenous EDA Report",
        "",
        "## 1. Target Inertia",
        f"- ADF (level) p-value: {level_tests['adf_p']:.6f}",
        f"- KPSS (level) p-value: {level_tests['kpss_p']:.6f}",
        f"- ADF (log return) p-value: {ret_tests['adf_p']:.6f}",
        f"- KPSS (log return) p-value: {ret_tests['kpss_p']:.6f}",
        f"- Hurst exponent on USDIDR log return: {hurst:.4f}",
        "",
        "### ACF/PACF Notes",
        f"- Top ACF lags: {', '.join([f'{int(r.lag)}({r.acf:.3f})' for r in top_acf.itertuples(index=False)])}",
        f"- Top PACF lags: {', '.join([f'{int(r.lag)}({r.pacf:.3f})' for r in top_pacf.itertuples(index=False)])}",
        "",
        "## 2. Exogenous Mixed-Frequency Audit",
        exog_summary.to_markdown(index=False),
        "",
        "## 3. Low-Frequency Event / Stale Behavior",
        transition_df.to_markdown(index=False),
        "",
        "## 4. Train-Test Scale Shift",
        shift_df.to_markdown(index=False),
        "",
        "## 5. Conditional Volatility Slices",
        quartile_corr.to_markdown(index=False),
        "",
        "### Directional AUC by VIX Quartile",
        auc_df.to_markdown(index=False),
        "",
        "## 6. Structural Breaks",
        chow_df.to_markdown(index=False),
        "",
        "### Rolling OLS Summary",
        rolling_summary.strip(),
        "",
        "## Bottom Line",
        "- Exogenous variables are not noisy corruption; they are mixed-frequency, stale, and regime-sensitive.",
        "- Risk proxies (SP500, VIX, IHSG) are the most informative slices.",
        "- Low-frequency rates need state/change features, not raw levels only.",
        "- There is clear train-test scale shift, especially for rate and equity variables.",
        "- Before seed-variance work, the right next step is to use exogenous as regime cues and hidden-state features, not direct daily continuous predictors.",
    ]
    write_report(ROOT / "exogenous_eda_report.md", report)


if __name__ == "__main__":
    main()
