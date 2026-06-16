import pandas as pd
from pathlib import Path
from typing import Tuple

RAW_DIR = Path("data/raw")
TARGET = "USDIDR"
DATE = "Date"


def load_train() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "data_train.csv")
    df[DATE] = pd.to_datetime(df[DATE])
    df = df.sort_values(DATE).reset_index(drop=True)
    return df


def load_test() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "data_test.csv")
    df[DATE] = pd.to_datetime(df[DATE])
    df = df.sort_values(DATE).reset_index(drop=True)
    return df


def temporal_split(df: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = int(n * train_ratio)
    return df.iloc[:split].copy().reset_index(drop=True), df.iloc[split:].copy().reset_index(drop=True)
