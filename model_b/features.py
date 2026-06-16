import pandas as pd


def prepare_raw_features(df):
    df = df.copy()
    has_target = "USDIDR" in df.columns

    cols = []
    for c in ["USDIDR", "OIL", "GOLD", "SP500", "VIX"]:
        if c in df.columns:
            cols.append(c)

    result = df[cols].copy()

    if has_target:
        result["usdidr_lag1"] = df["USDIDR"].shift(1)
        result["usdidr_lag2"] = df["USDIDR"].shift(2)
        result["usdidr_ema5"] = df["USDIDR"].shift(1).ewm(span=5).mean()
        result["usdidr_rmean5"] = df["USDIDR"].shift(1).rolling(5).mean()
    else:
        result["usdidr_lag1"] = 0
        result["usdidr_lag2"] = 0
        result["usdidr_ema5"] = 0
        result["usdidr_rmean5"] = 0

    result = result.dropna().reset_index(drop=True)
    return result