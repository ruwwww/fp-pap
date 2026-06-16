import pandas as pd

EXOG = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]


def build_features_ridge(df):
    df = df.copy()
    has_target = "USDIDR" in df.columns

    if has_target:
        for lag in [1, 2, 3, 5, 10, 20, 30, 60]:
            df[f"target_lag_{lag}"] = df["USDIDR"].shift(lag)
        for w in [5, 10, 20, 60]:
            df[f"target_rmean_{w}"] = df["USDIDR"].shift(1).rolling(w).mean()
            df[f"target_rstd_{w}"] = df["USDIDR"].shift(1).rolling(w).std()
        for span in [5, 20, 60]:
            df[f"target_ema_{span}"] = df["USDIDR"].shift(1).ewm(span=span).mean()
        for d in [1, 5, 20]:
            df[f"target_diff_{d}"] = df["USDIDR"].diff(d).shift(1)

    for col in EXOG:
        if col not in df.columns:
            continue
        for lag in [1, 5]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        for w in [5, 20]:
            df[f"{col}_rmean_{w}"] = df[col].shift(1).rolling(w).mean()

    if all(c in df.columns for c in ["GOLD", "OIL"]):
        df["gold_oil_ratio"] = df["GOLD"] / (df["OIL"] + 1e-8)
    if all(c in df.columns for c in ["SP500", "VIX"]):
        df["sp500_vix_ratio"] = df["SP500"] / (df["VIX"] + 1e-8)
    if all(c in df.columns for c in ["BI_rate", "US_rate"]):
        df["rate_spread"] = df["BI_rate"] - df["US_rate"]

    df = df.dropna().reset_index(drop=True)
    y = df["USDIDR"] if has_target else None
    drop_cols = ["Date"]
    if has_target:
        drop_cols.append("USDIDR")
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y


def prepare_gru_raw(df):
    df = df.copy()
    has_target = "USDIDR" in df.columns
    cols = []
    for c in ["USDIDR", "OIL", "GOLD", "SP500", "VIX"]:
        if c in df.columns:
            cols.append(c)

    result = df[cols].copy()

    if has_target:
        result["usdidr_ema5"] = df["USDIDR"].shift(1).ewm(span=5).mean()
        result["usdidr_rmean5"] = df["USDIDR"].shift(1).rolling(5).mean()
    else:
        result["usdidr_ema5"] = 0
        result["usdidr_rmean5"] = 0

    result = result.dropna().reset_index(drop=True)
    return result