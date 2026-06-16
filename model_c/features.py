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
        df["target_rmin_20"] = df["USDIDR"].shift(1).rolling(20).min()
        df["target_rmax_20"] = df["USDIDR"].shift(1).rolling(20).max()
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
    if all(c in df.columns for c in ["IHSG", "SP500"]):
        df["ihsg_sp500_ratio"] = df["IHSG"] / (df["SP500"] + 1e-8)
    if all(c in df.columns for c in ["BI_rate", "US_rate"]):
        df["rate_spread"] = df["BI_rate"] - df["US_rate"]

    df = df.dropna().reset_index(drop=True)
    y = df["USDIDR"] if has_target else None
    drop_cols = ["Date"]
    if has_target:
        drop_cols.append("USDIDR")
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y


def build_features_rf(df):
    df = df.copy()
    has_target = "USDIDR" in df.columns

    if has_target:
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f"target_lag_{lag}"] = df["USDIDR"].shift(lag)
        for w in [20]:
            df[f"target_rmean_{w}"] = df["USDIDR"].shift(1).rolling(w).mean()
            df[f"target_rstd_{w}"] = df["USDIDR"].shift(1).rolling(w).std()
        df["target_ema_20"] = df["USDIDR"].shift(1).ewm(span=20).mean()

    for col in ["OIL", "GOLD", "SP500", "VIX"]:
        if col not in df.columns:
            continue
        df[f"{col}_lag_1"] = df[col].shift(1)

    if all(c in df.columns for c in ["GOLD", "OIL"]):
        df["gold_oil_ratio"] = df["GOLD"] / (df["OIL"] + 1e-8)
    if all(c in df.columns for c in ["SP500", "VIX"]):
        df["sp500_vix_ratio"] = df["SP500"] / (df["VIX"] + 1e-8)

    df = df.dropna().reset_index(drop=True)
    y = df["USDIDR"] if has_target else None
    drop_cols = ["Date"]
    if has_target:
        drop_cols.append("USDIDR")
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y