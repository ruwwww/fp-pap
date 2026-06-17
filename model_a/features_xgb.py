import pandas as pd
import numpy as np

TARGET_LAGS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 60]
ROLLING_MEAN_WINDOWS = [5, 10, 20, 60]
ROLLING_STD_WINDOWS = [5, 10, 20]
EMA_SPANS = [5, 10, 20, 60]
EXOG_COLS = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]
EXOG_LAGS = [1, 5, 10]
EXOG_ROLLING_WINDOWS = [5, 20]


def build_features_xgb(df):
    """Build richer features for XGBoost. Returns X (DataFrame), y (Series or None)."""
    df = df.copy()
    has_target = "USDIDR" in df.columns

    # --- Target lag / rolling features ---
    if has_target:
        for lag in TARGET_LAGS:
            df[f"lag_{lag}"] = df["USDIDR"].shift(lag)
        for w in ROLLING_MEAN_WINDOWS:
            df[f"rmean_{w}"] = df["USDIDR"].shift(1).rolling(w).mean()
        for w in ROLLING_STD_WINDOWS:
            df[f"rstd_{w}"] = df["USDIDR"].shift(1).rolling(w).std()
        for span in EMA_SPANS:
            df[f"ema_{span}"] = df["USDIDR"].shift(1).ewm(span=span).mean()
        # Momentum
        df["mom_5"]  = df["USDIDR"].shift(1) - df["USDIDR"].shift(6)
        df["mom_20"] = df["USDIDR"].shift(1) - df["USDIDR"].shift(21)
        # Volatility ratio
        df["vol_ratio"] = (df["USDIDR"].shift(1).rolling(5).std() /
                           (df["USDIDR"].shift(1).rolling(20).std() + 1e-8))

    # --- Exogenous features (no lag needed, already known at forecast time) ---
    for col in EXOG_COLS:
        if col not in df.columns:
            continue
        # Current value (known from test data)
        df[f"{col}_cur"] = df[col]
        # Lags
        for lag in EXOG_LAGS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        # Rolling mean
        for w in EXOG_ROLLING_WINDOWS:
            df[f"{col}_rm{w}"] = df[col].shift(1).rolling(w).mean()

    # --- Macro interaction features ---
    if "GOLD" in df.columns and "OIL" in df.columns:
        df["gold_oil_ratio"] = df["GOLD"] / (df["OIL"] + 1e-8)
    if "SP500" in df.columns and "VIX" in df.columns:
        df["sp500_vix_ratio"] = df["SP500"] / (df["VIX"] + 1e-8)
    if "BI_rate" in df.columns and "US_rate" in df.columns:
        df["rate_spread"] = df["BI_rate"] - df["US_rate"]
    if "US_rate" in df.columns and "CPI" in df.columns:
        df["real_rate_us"] = df["US_rate"] - df["CPI"]

    # --- Calendar features ---
    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"])
        df["month"]     = dt.dt.month
        df["dayofweek"] = dt.dt.dayofweek
        df["quarter"]   = dt.dt.quarter

    df = df.dropna().reset_index(drop=True)

    y = df["USDIDR"] if has_target else None

    drop_cols = ["Date"] + ([c for c in EXOG_COLS if c in df.columns])
    if has_target:
        drop_cols.append("USDIDR")
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return X, y
