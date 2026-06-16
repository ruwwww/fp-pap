"""
src/models/base_model.py
─────────────────────────
Abstract base class for ALL models (ML, DL, Ensemble).
Enforces a consistent interface across every model.
"""
import time
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import joblib

logger = logging.getLogger(__name__)


class BaseTimeSeriesModel(ABC):
    """
    Abstract base for all time series models.

    Every model must implement:
      - fit(X_train, y_train)
      - predict(X_test) → np.ndarray
      - get_params() → dict

    Provided for free:
      - save(path) / load(path)
      - fit_time, predict_time tracking
    """

    def __init__(self, name: str, category: str = "ML", params: Dict[str, Any] = None):
        self.name = name
        self.category = category
        self.params = params or {}
        self.model = None
        self.fit_time_: float = 0.0
        self.predict_time_: float = 0.0
        self._is_fitted = False

    # ── Abstract interface ──────────────────────────────────────────────────
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseTimeSeriesModel":
        """Train the model. Must return self."""
        ...

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions for X_test."""
        ...

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters."""
        ...

    # ── Timed wrappers ──────────────────────────────────────────────────────
    def fit_timed(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseTimeSeriesModel":
        """Fit with wall-clock timing."""
        t0 = time.time()
        result = self.fit(X_train, y_train)
        self.fit_time_ = time.time() - t0
        self._is_fitted = True
        logger.info(f"[{self.name}] Fitted in {self.fit_time_:.2f}s")
        return result

    def predict_timed(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict with wall-clock timing."""
        t0 = time.time()
        preds = self.predict(X_test)
        self.predict_time_ = time.time() - t0
        logger.debug(f"[{self.name}] Predicted in {self.predict_time_:.3f}s")
        return preds

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self, directory: str, suffix: str = "") -> str:
        """Save model to disk using joblib."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        fname = f"{self.name}{suffix}.joblib"
        joblib.dump(self, path / fname)
        logger.info(f"[{self.name}] Saved → {path / fname}")
        return str(path / fname)

    @classmethod
    def load(cls, filepath: str) -> "BaseTimeSeriesModel":
        """Load a saved model from disk."""
        return joblib.load(filepath)

    # ── Repr ────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self._is_fitted})"
