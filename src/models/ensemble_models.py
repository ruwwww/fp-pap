"""
src/models/ensemble_models.py
──────────────────────────────
Ensemble Models: Voting, Stacking, Blending
(sesuai instruksi: Model C dan D — salah satu HARUS deep learning)

All ensembles accept pre-fitted base model instances.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from .base_model import BaseTimeSeriesModel

logger = logging.getLogger(__name__)


# ── Weighted Voting Ensemble ────────────────────────────────────────────────
class VotingEnsemble(BaseTimeSeriesModel):
    """
    Simple weighted average of base model predictions.

    Args:
        base_models: List of fitted BaseTimeSeriesModel instances
        weights: List of floats (must sum to ~1). If None, equal weights.
    """

    def __init__(
        self,
        base_models: List[BaseTimeSeriesModel],
        weights: Optional[List[float]] = None,
        params: Dict[str, Any] = None,
    ):
        super().__init__(name="VotingEnsemble", category="Ensemble", params=params or {})
        self.base_models = base_models
        n = len(base_models)
        self.weights = weights if weights else [1.0 / n] * n
        assert len(self.weights) == n, "weights must match number of base models"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "VotingEnsemble":
        """Fit all base models."""
        for model in self.base_models:
            logger.info(f"  [VotingEnsemble] Fitting {model.name}...")
            model.fit_timed(X_train, y_train)
        self._is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Weighted average of all base model predictions."""
        preds = np.column_stack([
            m.predict(X_test) for m in self.base_models
        ])  # shape: (n_samples, n_models)
        weights = np.array(self.weights)
        return (preds * weights).sum(axis=1)

    def get_params(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "base_models": [m.name for m in self.base_models],
        }


# ── Stacking Ensemble ────────────────────────────────────────────────────────
class StackingEnsemble(BaseTimeSeriesModel):
    """
    Stacking with a meta-learner trained on out-of-fold predictions.

    Args:
        base_models: List of BaseTimeSeriesModel instances
        meta_learner: BaseTimeSeriesModel to use as meta-learner
        cv_folds: Number of time-series CV folds
    """

    def __init__(
        self,
        base_models: List[BaseTimeSeriesModel],
        meta_learner: BaseTimeSeriesModel,
        cv_folds: int = 5,
        params: Dict[str, Any] = None,
    ):
        super().__init__(name="StackingEnsemble", category="Ensemble", params=params or {})
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "StackingEnsemble":
        """
        1. Generate out-of-fold predictions from base models
        2. Train meta-learner on those predictions (no leakage)
        3. Refit base models on full training data
        """
        n = len(X_train)
        fold_size = n // self.cv_folds
        oof_preds = np.zeros((n, len(self.base_models)))

        # ── Out-of-fold predictions ─────────────────────────────────────
        for fold in range(self.cv_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.cv_folds - 1 else n
            tr_idx = list(range(0, val_start)) + list(range(val_end, n))
            val_idx = list(range(val_start, val_end))

            if len(tr_idx) == 0:
                continue

            X_tr_fold = X_train.iloc[tr_idx]
            y_tr_fold = y_train.iloc[tr_idx]
            X_val_fold = X_train.iloc[val_idx]

            for j, model in enumerate(self.base_models):
                model.fit(X_tr_fold, y_tr_fold)
                oof_preds[val_idx, j] = model.predict(X_val_fold)

        logger.info(f"[StackingEnsemble] OOF predictions generated across {self.cv_folds} folds")

        # ── Train meta-learner on OOF predictions ──────────────────────
        meta_X = pd.DataFrame(oof_preds, columns=[m.name for m in self.base_models])
        self.meta_learner.fit(meta_X, y_train)
        logger.info(f"[StackingEnsemble] Meta-learner '{self.meta_learner.name}' trained")

        # ── Refit base models on FULL training data ────────────────────
        for model in self.base_models:
            model.fit_timed(X_train, y_train)

        self._is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Base model predictions → meta-learner prediction."""
        base_preds = np.column_stack([m.predict(X_test) for m in self.base_models])
        meta_X = pd.DataFrame(base_preds, columns=[m.name for m in self.base_models])
        return self.meta_learner.predict(meta_X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "cv_folds": self.cv_folds,
            "base_models": [m.name for m in self.base_models],
            "meta_learner": self.meta_learner.name,
        }


# ── Blending Ensemble ──────────────────────────────────────────────────────
class BlendingEnsemble(BaseTimeSeriesModel):
    """
    Blending: train base models on train set, hold-out validation set
    used to train meta-learner (simpler than stacking, less risk of leakage).

    Args:
        base_models: List of BaseTimeSeriesModel instances
        meta_learner: BaseTimeSeriesModel to use as blender
        val_ratio: Fraction of training data to hold out for blending
    """

    def __init__(
        self,
        base_models: List[BaseTimeSeriesModel],
        meta_learner: BaseTimeSeriesModel,
        val_ratio: float = 0.2,
        params: Dict[str, Any] = None,
    ):
        super().__init__(name="BlendingEnsemble", category="Ensemble", params=params or {})
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.val_ratio = val_ratio

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BlendingEnsemble":
        n = len(X_train)
        split = int(n * (1 - self.val_ratio))

        X_tr = X_train.iloc[:split]
        y_tr = y_train.iloc[:split]
        X_val = X_train.iloc[split:]
        y_val = y_train.iloc[split:]

        # ── Fit base models on training portion ───────────────────────
        for model in self.base_models:
            logger.info(f"  [BlendingEnsemble] Fitting {model.name}...")
            model.fit_timed(X_tr, y_tr)

        # ── Generate blend predictions on validation portion ──────────
        blend_preds = np.column_stack([m.predict(X_val) for m in self.base_models])
        meta_X = pd.DataFrame(blend_preds, columns=[m.name for m in self.base_models])
        self.meta_learner.fit(meta_X, y_val)
        logger.info(f"[BlendingEnsemble] Meta-learner '{self.meta_learner.name}' blended")

        self._is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        base_preds = np.column_stack([m.predict(X_test) for m in self.base_models])
        meta_X = pd.DataFrame(base_preds, columns=[m.name for m in self.base_models])
        return self.meta_learner.predict(meta_X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "val_ratio": self.val_ratio,
            "base_models": [m.name for m in self.base_models],
            "meta_learner": self.meta_learner.name,
        }
