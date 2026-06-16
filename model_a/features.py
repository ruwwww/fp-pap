import pandas as pd
import numpy as np

TARGET_LAGS = [1, 2, 3, 5, 10, 20, 30, 60]
ROLLING_MEAN_WINDOWS = [5, 10, 20, 60]
ROLLING_STD_WINDOWS = [5, 20]
EMA_SPANS = [5, 20, 60]
EXOG_COLS = ["OIL", "GOLD", "SP500", "VIX"]
EXOG_LAGS = [1, 5]
EXOG_ROLLING_WINDOWS = [5, 20]


def build_features(df):
    """Build features for linear model. Returns X (DataFrame), y (Series or None if no target)."""
    df = df.copy()
    has_target = "USDIDR" in df.columns

    if has_target:
        for lag in TARGET_LAGS:
            df[f"target_lag_{lag}"] = df["USDIDR"].shift(1).shift(lag - 1)

        for w in ROLLING_MEAN_WINDOWS:
            df[f"target_rmean_{w}"] = df["USDIDR"].shift(1).rolling(w).mean()

        for w in ROLLING_STD_WINDOWS:
            df[f"target_rstd_{w}"] = df["USDIDR"].shift(1).rolling(w).std()

        for span in EMA_SPANS:
            df[f"target_ema_{span}"] = df["USDIDR"].shift(1).ewm(span=span).mean()

    for col in EXOG_COLS:
        if col not in df.columns:
            continue
        for lag in EXOG_LAGS:
            df[f"{col}_lag_{lag}"] = df[col].shift(1).shift(lag - 1)
        for w in EXOG_ROLLING_WINDOWS:
            df[f"{col}_rmean_{w}"] = df[col].shift(1).rolling(w).mean()

    if "GOLD" in df.columns and "OIL" in df.columns:
        df["gold_oil_ratio"] = df["GOLD"].shift(1) / (df["OIL"].shift(1) + 1e-8)
    if "SP500" in df.columns and "VIX" in df.columns:
        df["sp500_vix_ratio"] = df["SP500"].shift(1) / (df["VIX"].shift(1) + 1e-8)
    if "BI_rate" in df.columns and "US_rate" in df.columns:
        df["rate_spread"] = df["BI_rate"].shift(1) - df["US_rate"].shift(1)

    df = df.dropna().reset_index(drop=True)

    y = df["USDIDR"] if has_target else None

    drop_cols = ["Date"]
    if has_target:
        drop_cols.append("USDIDR")
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return X, y
