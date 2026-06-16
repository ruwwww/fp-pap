"""
src/data/preprocessor.py
────────────────────────
Handles missing values, outlier treatment, and scaling.
All operations are fit-on-train-only to prevent data leakage.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Optional, Literal

logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """
    Stateful preprocessor that fits ONLY on training data.
    Call fit_transform(train) → transform(test) pattern.

    ⚠️  ANTI-LEAKAGE: Never fit on test data.
    """

    def __init__(
        self,
        scaler_type: Literal["standard", "minmax", "robust", "none"] = "standard",
        impute_strategy: Literal["mean", "median", "ffill", "bfill"] = "ffill",
        remove_outliers: bool = False,
        outlier_threshold: float = 3.0,
    ):
        self.scaler_type = scaler_type
        self.impute_strategy = impute_strategy
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold

        self.scaler = None
        self.feature_cols_: list = []
        self._fitted = False

    def _get_scaler(self):
        return {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "none": None,
        }[self.scaler_type]

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using the configured strategy."""
        if self.impute_strategy in ("ffill", "bfill"):
            return df.fillna(method=self.impute_strategy).bfill().ffill()
        else:
            imputer = SimpleImputer(strategy=self.impute_strategy)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            return df

    def _handle_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Cap outliers using Z-score method (IQR-based alternative available)."""
        z = np.abs((df[col] - df[col].mean()) / df[col].std())
        upper = df[col].mean() + self.outlier_threshold * df[col].std()
        lower = df[col].mean() - self.outlier_threshold * df[col].std()
        df[col] = df[col].clip(lower=lower, upper=upper)
        n_outliers = (z > self.outlier_threshold).sum()
        if n_outliers:
            logger.debug(f"  Capped {n_outliers} outliers in '{col}'")
        return df

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        feature_cols: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Fit on training data and return transformed training data.

        Args:
            train_df: Training DataFrame
            feature_cols: Columns to scale. If None, uses all numeric columns.

        Returns:
            Transformed training DataFrame
        """
        df = train_df.copy()
        df = self._impute(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols_ = feature_cols if feature_cols else numeric_cols

        if self.remove_outliers:
            for col in self.feature_cols_:
                df = self._handle_outliers(df, col)

        self.scaler = self._get_scaler()
        if self.scaler is not None:
            df[self.feature_cols_] = self.scaler.fit_transform(df[self.feature_cols_])
            logger.info(f"Fitted {self.scaler_type} scaler on {len(self.feature_cols_)} features.")

        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data (test/future) using the fitted scaler.

        ⚠️  ANTI-LEAKAGE: Never re-fit here.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")

        df = df.copy()
        df = df.ffill().bfill()  # lightweight impute for test

        if self.scaler is not None:
            present_cols = [c for c in self.feature_cols_ if c in df.columns]
            df[present_cols] = self.scaler.transform(df[present_cols])

        return df

    def inverse_transform(self, values: np.ndarray, col_index: int = 0) -> np.ndarray:
        """Invert scaling for predictions (useful for DL models)."""
        if self.scaler is None:
            return values
        dummy = np.zeros((len(values), len(self.feature_cols_)))
        dummy[:, col_index] = values.flatten()
        inversed = self.scaler.inverse_transform(dummy)
        return inversed[:, col_index]
