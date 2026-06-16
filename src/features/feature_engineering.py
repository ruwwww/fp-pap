"""
src/features/feature_engineering.py  (v2 — USD/IDR dataset)
─────────────────────────────────────────────────────────────
Feature engineering for the Rupiah Resilience dataset.

Exogenous variables available: OIL, GOLD, SP500, IHSG, VIX, CPI, BI_rate, US_rate
Target: USDIDR

KEY INSIGHT: This dataset already has multivariate features.
Feature engineering adds TEMPORAL/LAG features on top of the existing columns.

⚠️  ANTI-LEAKAGE RULES (CRITICAL):
  - Rolling stats: ALWAYS shift(1) before rolling to exclude current row
  - Lag features: shift ≥ 1
  - All transformations fit on TRAIN only
  - Test features computed with train context prepended
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Optional

logger = logging.getLogger(__name__)


class TimeSeriesFeatureEngineer:
    """
    Creates lag, rolling, ratio, and calendar features.
    Works on BOTH the target (USDIDR) and exogenous variables (OIL, GOLD, etc.)
    """

    def __init__(
        self,
        target_col: str = "USDIDR",
        date_col: str = "Date",
        # Lag config
        target_lags: List[int] = [1, 2, 3, 5, 10, 20, 30, 60],
        exog_lags: List[int] = [1, 5],              # lags for exogenous vars
        # Rolling config
        rolling_windows: List[int] = [5, 10, 20, 60],
        # Calendar
        add_calendar: bool = True,
        # Ratio / interaction features
        add_ratios: bool = True,
    ):
        self.target_col     = target_col
        self.date_col       = date_col
        self.target_lags    = target_lags
        self.exog_lags      = exog_lags
        self.rolling_windows = rolling_windows
        self.add_calendar   = add_calendar
        self.add_ratios     = add_ratios

        # Exogenous columns (populated during fit)
        self._exog_cols: List[str] = []
        self._feature_cols: List[str] = []

    # ── Calendar features ──────────────────────────────────────────────────
    def _add_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.date_col not in df.columns:
            return df
        dt = df[self.date_col]
        df["day_of_week"]   = dt.dt.dayofweek
        df["day_of_month"]  = dt.dt.day
        df["day_of_year"]   = dt.dt.dayofyear
        df["week_of_year"]  = dt.dt.isocalendar().week.astype(int)
        df["month"]         = dt.dt.month
        df["quarter"]       = dt.dt.quarter
        df["year"]          = dt.dt.year
        df["is_month_start"] = dt.dt.is_month_start.astype(int)
        df["is_month_end"]   = dt.dt.is_month_end.astype(int)
        df["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)
        # Cyclical encoding for month and day_of_week
        df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"]      = np.sin(2 * np.pi * df["day_of_week"] / 5)
        df["dow_cos"]      = np.cos(2 * np.pi * df["day_of_week"] / 5)
        return df

    # ── Target lag features ────────────────────────────────────────────────
    def _add_target_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lag the USDIDR target. shift(n) ensures no leakage."""
        for lag in self.target_lags:
            df[f"usdidr_lag_{lag}"] = df[self.target_col].shift(lag)
        return df

    # ── Rolling features on target ─────────────────────────────────────────
    def _add_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling stats on USDIDR.
        shift(1) excludes current row → no leakage.
        """
        shifted = df[self.target_col].shift(1)
        for w in self.rolling_windows:
            df[f"usdidr_rmean_{w}"] = shifted.rolling(w, min_periods=1).mean()
            df[f"usdidr_rstd_{w}"]  = shifted.rolling(w, min_periods=1).std().fillna(0)
            df[f"usdidr_rmin_{w}"]  = shifted.rolling(w, min_periods=1).min()
            df[f"usdidr_rmax_{w}"]  = shifted.rolling(w, min_periods=1).max()
            # Range
            df[f"usdidr_rrange_{w}"] = (
                df[f"usdidr_rmax_{w}"] - df[f"usdidr_rmin_{w}"]
            )
        # EMA (also shifted)
        for span in [5, 20, 60]:
            df[f"usdidr_ema_{span}"] = shifted.ewm(span=span, adjust=False).mean()
        # Momentum / diff
        df["usdidr_diff_1"]  = df[self.target_col].shift(1).diff(1)
        df["usdidr_diff_5"]  = df[self.target_col].shift(1).diff(5)
        df["usdidr_diff_20"] = df[self.target_col].shift(1).diff(20)
        return df

    # ── Exogenous lags & rolling ───────────────────────────────────────────
    def _add_exog_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lags and rolling for exogenous variables.
        All use shift ≥ 1 → no leakage.
        """
        exog_cols = [c for c in self._exog_cols if c in df.columns]
        for col in exog_cols:
            # Lags
            for lag in self.exog_lags:
                df[f"{col.lower()}_lag_{lag}"] = df[col].shift(lag)
            # Short rolling (shifted)
            shifted = df[col].shift(1)
            df[f"{col.lower()}_rmean_5"]  = shifted.rolling(5, min_periods=1).mean()
            df[f"{col.lower()}_rmean_20"] = shifted.rolling(20, min_periods=1).mean()
            df[f"{col.lower()}_diff_1"]   = df[col].shift(1).diff(1)
        return df

    # ── Ratio / interaction features ───────────────────────────────────────
    def _add_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Domain-specific feature interactions for USD/IDR:
          - GOLD/OIL ratio (commodity spread)
          - SP500/VIX (risk appetite)
          - IHSG/SP500 (relative equity performance)
          - USDIDR lag vs EMA (mean-reversion signal)
        """
        eps = 1e-8
        if "OIL" in df.columns and "GOLD" in df.columns:
            df["gold_oil_ratio"] = df["GOLD"] / (df["OIL"].abs() + eps)
        if "SP500" in df.columns and "VIX" in df.columns:
            df["sp500_vix_ratio"] = df["SP500"] / (df["VIX"] + eps)
        if "IHSG" in df.columns and "SP500" in df.columns:
            df["ihsg_sp500_ratio"] = df["IHSG"] / (df["SP500"] + eps)
        if "BI_rate" in df.columns and "US_rate" in df.columns:
            df["rate_spread"] = df["BI_rate"] - df["US_rate"]
        # Mean-reversion: lag_1 relative to EMA_20 (if exists)
        if "usdidr_lag_1" in df.columns and "usdidr_ema_20" in df.columns:
            df["usdidr_vs_ema20"] = df["usdidr_lag_1"] - df["usdidr_ema_20"]
        return df

    # ── Public API ─────────────────────────────────────────────────────────
    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on training data, return feature-engineered training DataFrame.
        Saves internal state for transform().
        """
        # Determine exogenous columns (everything except target and date)
        self._exog_cols = [
            c for c in train_df.columns
            if c not in (self.target_col, self.date_col)
        ]

        df = train_df.copy()
        df = self._add_target_lags(df)
        df = self._add_rolling(df)
        df = self._add_exog_features(df)
        if self.add_ratios:
            df = self._add_ratios(df)
        if self.add_calendar:
            df = self._add_calendar(df)

        self._feature_cols = self.get_feature_columns(df)
        n_orig = train_df.shape[1]
        logger.info(
            f"Feature engineering: {n_orig} → {df.shape[1]} cols "
            f"({df.shape[1]-n_orig} new features)"
        )
        return df

    def transform(
        self,
        test_df: pd.DataFrame,
        full_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform test set using full context (train + test).

        Args:
            test_df: Test-only DataFrame (for index alignment)
            full_df: pd.concat([train_df, test_df]) — needed for correct lags

        ⚠️  ANTI-LEAKAGE: full_df rows before test start are used as context.
        """
        df = full_df.copy()
        df = self._add_target_lags(df)
        df = self._add_rolling(df)
        df = self._add_exog_features(df)
        if self.add_ratios:
            df = self._add_ratios(df)
        if self.add_calendar:
            df = self._add_calendar(df)

        # Return only test portion
        return df.loc[test_df.index].copy()

    def transform_kaggle_test(
        self,
        train_df: pd.DataFrame,
        kaggle_test_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Special transform for Kaggle test set (no USDIDR column).
        Appends a placeholder USDIDR=NaN row for each test row,
        then computes features using train context.

        Used for generating Kaggle submission predictions.
        """
        # Add NaN target to test so feature functions can run
        test_with_nan = kaggle_test_df.copy()
        test_with_nan[self.target_col] = np.nan

        # Reorder columns to match train
        cols_order = [c for c in train_df.columns if c in test_with_nan.columns] + \
                     [c for c in test_with_nan.columns if c not in train_df.columns]
        test_with_nan = test_with_nan[[c for c in train_df.columns
                                       if c in test_with_nan.columns or c == self.target_col]]

        full = pd.concat([train_df, test_with_nan], ignore_index=True)
        full_fe = self.fit_transform(full)

        test_indices = range(len(train_df), len(full))
        return full_fe.iloc[list(test_indices)].copy()

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return feature columns (exclude target and date)."""
        exclude = {self.target_col, self.date_col}
        return [c for c in df.columns if c not in exclude]
