"""
src/features/feature_engineering.py  (v3 — Modular Feature Configs)
───────────────────────────────────────────────────────────────────
Feature engineering for the Rupiah Resilience dataset.

KEY CHANGE (v3): Accepts named feature configs from
config/feature_configs.yaml. Each model can specify which
config to use via the `feature_config` field in models_config.yaml.

Config presets:
  - full:     101 features (default)
  - reduced:  ~40 features (for tree models)
  - minimal:  ~15 features (lag-only + ratios)
  - lag_only: ~10 features (pure autoregressive)
  - dl_only:  Raw time series (NO flat features — for LSTM/GRU)
  - ridge_full: Full numeric features (no calendar)

Usage:
    # Default (full)
    fe = TimeSeriesFeatureEngineer(target_col="USDIDR")

    # Named config
    fe = TimeSeriesFeatureEngineer(target_col="USDIDR", config_name="reduced")

    # Direct dict config
    fe = TimeSeriesFeatureEngineer(target_col="USDIDR", config_dict={...})
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = "config/feature_configs.yaml"

VALID_ROLLING_STATS = {"mean", "std", "min", "max", "range"}


def _load_config(
    config_name: Optional[str] = None,
    config_dict: Optional[dict] = None,
    config_path: str = _DEFAULT_CONFIG_PATH,
) -> dict:
    """Load feature config by name or return a validated config dict."""
    if config_dict is not None:
        cfg = {**_full_defaults(), **config_dict}
        _validate_config(cfg)
        return cfg

    name = config_name or "full"
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Feature config file not found: {path}, using defaults")
        return {**_full_defaults(), "config_name": name}

    with open(path, encoding="utf-8") as f:
        all_configs = yaml.safe_load(f) or {}

    cfgs = all_configs.get("feature_configs", {})
    if name not in cfgs:
        logger.warning(
            f"Feature config '{name}' not found in {config_path}, "
            f"available: {list(cfgs.keys())}. Falling back to 'full'."
        )
        name = "full"

    cfg = {**_full_defaults(), **cfgs[name], "config_name": name}
    _validate_config(cfg)
    return cfg


def _full_defaults() -> dict:
    return {
        "config_name": "full",
        "target_lags": [1, 2, 3, 5, 10, 20, 30, 60],
        "exog_lags": [1, 5],
        "rolling_windows": [5, 10, 20, 60],
        "rolling_stats": ["mean", "std", "min", "max", "range"],
        "ema_spans": [5, 20, 60],
        "diffs": [1, 5, 20],
        "add_calendar": True,
        "add_ratios": True,
        "raw_only": False,
    }


def _validate_config(cfg: dict):
    invalid = [s for s in cfg.get("rolling_stats", []) if s not in VALID_ROLLING_STATS]
    if invalid:
        raise ValueError(
            f"Invalid rolling_stats: {invalid}. Valid: {VALID_ROLLING_STATS}"
        )


class TimeSeriesFeatureEngineer:
    """
    Creates lag, rolling, ratio, and calendar features.

    Parameters
    ----------
    target_col : str
        Name of the target column.
    date_col : str
        Name of the date column.
    config_name : str, optional
        Name of the feature config preset (from feature_configs.yaml).
    config_dict : dict, optional
        Inline config dict (overrides config_name if both provided).
        Keys: target_lags, exog_lags, rolling_windows, rolling_stats,
              ema_spans, diffs, add_calendar, add_ratios, raw_only.
    config_path : str
        Path to feature_configs.yaml.
    """

    def __init__(
        self,
        target_col: str = "USDIDR",
        date_col: str = "Date",
        config_name: Optional[str] = None,
        config_dict: Optional[dict] = None,
        config_path: str = _DEFAULT_CONFIG_PATH,
    ):
        self.target_col = target_col
        self.date_col = date_col

        self._cfg = _load_config(config_name, config_dict, config_path)

        self.target_lags = self._cfg["target_lags"]
        self.exog_lags = self._cfg["exog_lags"]
        self.rolling_windows = self._cfg["rolling_windows"]
        self.rolling_stats = self._cfg["rolling_stats"]
        self.ema_spans = self._cfg["ema_spans"]
        self.diffs = self._cfg["diffs"]
        self.add_calendar = self._cfg["add_calendar"]
        self.add_ratios = self._cfg["add_ratios"]
        self._raw_only = self._cfg["raw_only"]

        self._exog_cols: List[str] = []
        self._feature_cols: List[str] = []

    def get_config_snapshot(self) -> dict:
        """Return a frozen copy of the feature config for artifact logging."""
        return dict(self._cfg)

    # ── Raw mode (dl_only): return exogenous columns as-is ────────────────
    def _raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only exogenous columns as features (no transformations)."""
        cols = self._exog_cols + [self.target_col]
        return df[cols].copy()

    # ── Calendar features ──────────────────────────────────────────────────
    def _add_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.date_col not in df.columns:
            return df
        dt = df[self.date_col]
        df["day_of_week"] = dt.dt.dayofweek
        df["day_of_month"] = dt.dt.day
        df["day_of_year"] = dt.dt.dayofyear
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        df["month"] = dt.dt.month
        df["quarter"] = dt.dt.quarter
        df["year"] = dt.dt.year
        df["is_month_start"] = dt.dt.is_month_start.astype(int)
        df["is_month_end"] = dt.dt.is_month_end.astype(int)
        df["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)
        return df

    # ── Target lag features ────────────────────────────────────────────────
    def _add_target_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        for lag in self.target_lags:
            df[f"usdidr_lag_{lag}"] = df[self.target_col].shift(lag)
        return df

    # ── Rolling features on target (configurable stats) ────────────────────
    def _add_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.rolling_windows:
            return df
        shifted = df[self.target_col].shift(1)
        for w in self.rolling_windows:
            rs = shifted.rolling(w, min_periods=1)
            for stat in self.rolling_stats:
                if stat == "mean":
                    df[f"usdidr_rmean_{w}"] = rs.mean()
                elif stat == "std":
                    df[f"usdidr_rstd_{w}"] = rs.std().fillna(0)
                elif stat == "min":
                    df[f"usdidr_rmin_{w}"] = rs.min()
                elif stat == "max":
                    df[f"usdidr_rmax_{w}"] = rs.max()
                elif stat == "range":
                    df[f"usdidr_rmin_{w}"] = rs.min()
                    df[f"usdidr_rmax_{w}"] = rs.max()
                    df[f"usdidr_rrange_{w}"] = (
                        df[f"usdidr_rmax_{w}"] - df[f"usdidr_rmin_{w}"]
                    )
        # EMA
        for span in self.ema_spans:
            df[f"usdidr_ema_{span}"] = shifted.ewm(span=span, adjust=False).mean()
        # Diffs / momentum
        for d in self.diffs:
            df[f"usdidr_diff_{d}"] = df[self.target_col].shift(1).diff(d)
        return df

    # ── Exogenous lags & rolling ───────────────────────────────────────────
    def _add_exog_features(self, df: pd.DataFrame) -> pd.DataFrame:
        exog_cols = [c for c in self._exog_cols if c in df.columns]
        if not exog_cols:
            return df
        for col in exog_cols:
            for lag in self.exog_lags:
                df[f"{col.lower()}_lag_{lag}"] = df[col].shift(lag)
            shifted = df[col].shift(1)
            df[f"{col.lower()}_rmean_5"] = shifted.rolling(5, min_periods=1).mean()
            df[f"{col.lower()}_rmean_20"] = shifted.rolling(20, min_periods=1).mean()
            df[f"{col.lower()}_diff_1"] = df[col].shift(1).diff(1)
        return df

    # ── Ratio / interaction features ───────────────────────────────────────
    def _add_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        eps = 1e-8
        if "OIL" in df.columns and "GOLD" in df.columns:
            df["gold_oil_ratio"] = df["GOLD"] / (df["OIL"].abs() + eps)
        if "SP500" in df.columns and "VIX" in df.columns:
            df["sp500_vix_ratio"] = df["SP500"] / (df["VIX"] + eps)
        if "IHSG" in df.columns and "SP500" in df.columns:
            df["ihsg_sp500_ratio"] = df["IHSG"] / (df["SP500"] + eps)
        if "BI_rate" in df.columns and "US_rate" in df.columns:
            df["rate_spread"] = df["BI_rate"] - df["US_rate"]
        if "usdidr_lag_1" in df.columns and "usdidr_ema_20" in df.columns:
            df["usdidr_vs_ema20"] = df["usdidr_lag_1"] - df["usdidr_ema_20"]
        return df

    # ── Public API ─────────────────────────────────────────────────────────
    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        train_df = train_df.copy()
        self._exog_cols = [
            c for c in train_df.columns
            if c not in (self.target_col, self.date_col)
        ]

        if self._raw_only:
            result = self._raw_features(train_df)
            self._feature_cols = self.get_feature_columns(result)
            logger.info(
                f"[raw_only] Features: {result.shape[1]} cols "
                f"(exogenous variables as-is)"
            )
            return result

        df = self._add_target_lags(train_df)
        df = self._add_rolling(df)
        df = self._add_exog_features(df)
        if self.add_ratios:
            df = self._add_ratios(df)
        if self.add_calendar:
            df = self._add_calendar(df)

        self._feature_cols = self.get_feature_columns(df)
        n_orig = train_df.shape[1]
        logger.info(
            f"[{self._cfg['config_name']}] Feature engineering: "
            f"{n_orig} → {df.shape[1]} cols "
            f"({df.shape[1] - n_orig} new features)"
        )
        return df

    def transform(
        self,
        test_df: pd.DataFrame,
        full_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if self._raw_only:
            cols = self._exog_cols + ([self.target_col] if self.target_col in test_df.columns else [])
            return test_df[cols].copy()

        df = full_df.copy()
        df = self._add_target_lags(df)
        df = self._add_rolling(df)
        df = self._add_exog_features(df)
        if self.add_ratios:
            df = self._add_ratios(df)
        if self.add_calendar:
            df = self._add_calendar(df)
        return df.loc[test_df.index].copy()

    def transform_kaggle_test(
        self,
        train_df: pd.DataFrame,
        kaggle_test_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if self._raw_only:
            cols = self._exog_cols + ([self.target_col] if self.target_col in kaggle_test_df.columns else [])
            return kaggle_test_df[cols].copy()

        test_with_nan = kaggle_test_df.copy()
        test_with_nan[self.target_col] = np.nan

        cols_order = [
            c for c in train_df.columns if c in test_with_nan.columns
        ] + [c for c in test_with_nan.columns if c not in train_df.columns]
        test_with_nan = test_with_nan[
            [c for c in train_df.columns if c in test_with_nan.columns or c == self.target_col]
        ]

        full = pd.concat([train_df, test_with_nan], ignore_index=True)
        full_fe = self.fit_transform(full)

        test_indices = range(len(train_df), len(full))
        return full_fe.iloc[list(test_indices)].copy()

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        exclude = {self.target_col, self.date_col}
        return [c for c in df.columns if c not in exclude]