"""
src/data/loader.py  (v2 — Rupiah Resilience aware)
────────────────────────────────────────────────────
Handles:
  - Train/test loading for USD/IDR dataset
  - Temporal train-test split from training data (for internal evaluation)
  - Loading Kaggle test set (for submission)

Dataset structure:
  data_train.csv : Date, OIL, GOLD, USDIDR, SP500, IHSG, VIX, CPI, BI_rate, US_rate
  data_test.csv  : Date, OIL, GOLD,         SP500, IHSG, VIX, CPI, BI_rate, US_rate
  submission.csv : Date, USDIDR (all zeros → fill with predictions)
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load project configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DataLoader:
    """
    Unified data loader for the Rupiah Resilience competition dataset.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.target_col = self.config["project"]["target_column"]   # "USDIDR"
        self.date_col   = self.config["project"]["date_column"]      # "Date"
        self.feat_cols  = self.config["project"]["feature_columns"]  # [OIL, GOLD, ...]
        self.raw_dir    = Path(self.config["paths"]["raw_data"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Generic CSV ────────────────────────────────────────────────────────
    def load_csv(
        self,
        filepath: str,
        parse_dates: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Load any CSV, parse date column, sort by date."""
        df = pd.read_csv(filepath, **kwargs)
        if parse_dates and self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df = df.sort_values(self.date_col).reset_index(drop=True)
            logger.info(
                f"Loaded {Path(filepath).name}: {df.shape} | "
                f"Range: {df[self.date_col].min().date()} → {df[self.date_col].max().date()}"
            )
        return df

    # ── Dataset-specific loaders ───────────────────────────────────────────
    def load_train(self) -> pd.DataFrame:
        """Load data_train.csv (has USDIDR target)."""
        path = self.config["data_files"]["train"]
        df = self.load_csv(path)
        logger.info(f"Train columns: {list(df.columns)}")
        return df

    def load_kaggle_test(self) -> pd.DataFrame:
        """
        Load data_test.csv (Kaggle competition test — NO USDIDR column).
        This is the data we need to predict for submission.
        Period: 2023-06-01 → 2026-05-29
        """
        path = self.config["data_files"]["test"]
        df = self.load_csv(path)
        logger.info(f"Kaggle test columns: {list(df.columns)}")
        assert self.target_col not in df.columns, \
            "Test set should NOT have USDIDR — check data file"
        return df

    def load_submission_template(self) -> pd.DataFrame:
        """Load submission.csv (Date + USDIDR=0.0 placeholder)."""
        path = self.config["data_files"]["submission_template"]
        df = pd.read_csv(path)
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        return df

    # ── Internal train-test split (for model evaluation) ──────────────────
    def get_internal_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.80,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Temporal split of training data for internal model evaluation.
        NO SHUFFLE — preserves time order.

        Args:
            df: Full training DataFrame (from load_train())
            train_ratio: Fraction to use for training

        Returns:
            (train_df, test_df)
        """
        n = len(df)
        split_idx = int(n * train_ratio)
        train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df  = df.iloc[split_idx:].copy().reset_index(drop=True)
        logger.info(
            f"Internal split {train_ratio:.0%}/{1-train_ratio:.0%}: "
            f"Train={len(train_df)} ({train_df[self.date_col].min().date()} → "
            f"{train_df[self.date_col].max().date()}) | "
            f"Test={len(test_df)} ({test_df[self.date_col].min().date()} → "
            f"{test_df[self.date_col].max().date()})"
        )
        return train_df, test_df

    def get_XY(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract X (features) and y (target) from a DataFrame."""
        feat_cols = feature_cols or self.feat_cols
        # Only use columns that exist in this df
        available = [c for c in feat_cols if c in df.columns]
        return df[available], df[self.target_col]

    # ── Submission builder ─────────────────────────────────────────────────
    def build_submission(
        self,
        predictions: pd.Series,
        dates: Optional[pd.Series] = None,
        output_path: str = "data/submissions/submission.csv",
    ) -> pd.DataFrame:
        """
        Create submission CSV matching Kaggle format (Date, USDIDR).

        Args:
            predictions: Model predictions (length = 778)
            dates: Date series. If None, loads from submission template.
            output_path: Where to save the file.
        """
        template = self.load_submission_template()
        if dates is None:
            dates = template[self.date_col]

        sub = pd.DataFrame({
            self.date_col: dates.values,
            self.target_col: predictions.values if hasattr(predictions, "values") else predictions,
        })

        assert len(sub) == len(template), \
            f"Submission length mismatch: got {len(sub)}, expected {len(template)}"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(output_path, index=False)
        logger.info(f"Submission saved → {output_path} ({len(sub)} rows)")
        logger.info(f"  USDIDR range: {sub[self.target_col].min():.0f} – {sub[self.target_col].max():.0f}")
        return sub
