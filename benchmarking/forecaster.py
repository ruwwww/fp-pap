"""
benchmarking/forecaster.py
───────────────────────────
Future forecasting module.
Generates predictions from 1 Jun 2023 → 29 Mei 2026 (sesuai instruksi poin 6).

Supports both ML (tabular) and DL (sequence) models.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


class FutureForecaster:
    """
    Generate multi-step ahead forecasts using a fitted model.

    Strategy:
      - For ML models: iterative one-step-ahead prediction
        (feed previous prediction as lag feature for next step)
      - For DL models: already handles sequences internally
    """

    def __init__(
        self,
        model,
        feature_engineer,
        target_col: str,
        date_col: str,
        freq: str = "D",
    ):
        self.model = model
        self.feature_engineer = feature_engineer
        self.target_col = target_col
        self.date_col = date_col
        self.freq = freq

    def forecast(
        self,
        history_df: pd.DataFrame,
        start_date: str = "2023-06-01",
        end_date: str = "2026-05-29",
    ) -> pd.DataFrame:
        """
        Iteratively forecast from start_date to end_date.

        Args:
            history_df: Full historical DataFrame (used as context)
            start_date: First forecast date
            end_date: Last forecast date

        Returns:
            DataFrame with columns [date_col, 'forecast']
        """
        forecast_dates = pd.date_range(start=start_date, end=end_date, freq=self.freq)
        logger.info(f"Forecasting {len(forecast_dates)} steps: {start_date} → {end_date}")

        extended_df = history_df.copy()
        forecast_values = []

        for step, date in enumerate(forecast_dates):
            # Build feature row for this date
            new_row = {self.date_col: date, self.target_col: np.nan}
            temp_df = pd.concat(
                [extended_df, pd.DataFrame([new_row])],
                ignore_index=True
            )

            # Apply feature engineering to the last row
            temp_fe = self.feature_engineer.fit_transform(temp_df)
            feat_cols = self.feature_engineer.get_feature_columns(temp_fe)
            X_step = temp_fe[feat_cols].iloc[[-1]]  # last row = current step

            # Predict
            pred = float(self.model.predict(X_step)[0])
            forecast_values.append(pred)

            # Append prediction to context (so next step can use it as lag)
            new_row[self.target_col] = pred
            extended_df = pd.concat(
                [extended_df, pd.DataFrame([new_row])],
                ignore_index=True
            )

            if step % 100 == 0:
                logger.debug(f"  Step {step}/{len(forecast_dates)}: {date.date()} → {pred:.4f}")

        result = pd.DataFrame({
            self.date_col: forecast_dates,
            "forecast": forecast_values,
        })
        logger.info(f"Forecast complete. Shape: {result.shape}")
        return result

    def save_forecast(
        self,
        forecast_df: pd.DataFrame,
        path: str = "data/submissions/forecast.csv",
    ) -> str:
        """Save forecast to CSV for Kaggle submission."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(path, index=False)
        logger.info(f"Forecast saved → {path}")
        return path
