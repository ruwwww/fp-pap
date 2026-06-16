"""
src/models/ml_models.py
────────────────────────
Model A — Machine Learning Models
All inherit from BaseTimeSeriesModel.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

from .base_model import BaseTimeSeriesModel

logger = logging.getLogger(__name__)


# ── Random Forest ───────────────────────────────────────────────────────────
class RandomForestModel(BaseTimeSeriesModel):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="RandomForest", category="ML", params=params or {})

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestModel":
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test)

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.model.feature_names_in_,
            name="importance",
        ).sort_values(ascending=False)


# ── XGBoost ─────────────────────────────────────────────────────────────────
class XGBoostModel(BaseTimeSeriesModel):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="XGBoost", category="ML", params=params or {})

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGBoostModel":
        from xgboost import XGBRegressor
        self.model = XGBRegressor(**self.params, verbosity=0)
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test)

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.model.feature_names_in_,
            name="importance",
        ).sort_values(ascending=False)


# ── LightGBM ─────────────────────────────────────────────────────────────────
class LightGBMModel(BaseTimeSeriesModel):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="LightGBM", category="ML", params=params or {})

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LightGBMModel":
        import lightgbm as lgb
        p = {**self.params, "verbose": -1}
        self.model = lgb.LGBMRegressor(**p)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test)

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.model.feature_name_,
            name="importance",
        ).sort_values(ascending=False)


# ── Gradient Boosting ─────────────────────────────────────────────────────────
class GradientBoostingModel(BaseTimeSeriesModel):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="GradientBoosting", category="ML", params=params or {})

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "GradientBoostingModel":
        from sklearn.ensemble import GradientBoostingRegressor
        self.model = GradientBoostingRegressor(**self.params)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test)

    def get_params(self) -> Dict[str, Any]:
        return self.params


# ── Ridge Regression ──────────────────────────────────────────────────────────
class RidgeModel(BaseTimeSeriesModel):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="Ridge", category="ML", params=params or {})

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RidgeModel":
        from sklearn.linear_model import Ridge
        self.model = Ridge(**self.params)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test)

    def get_params(self) -> Dict[str, Any]:
        return self.params


# ── SVR ───────────────────────────────────────────────────────────────────────
class SVRModel(BaseTimeSeriesModel):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="SVR", category="ML", params=params or {})

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "SVRModel":
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(**self.params)),
        ])
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test)

    def get_params(self) -> Dict[str, Any]:
        return self.params


# ── CatBoost ──────────────────────────────────────────────────────────────────
class CatBoostModel(BaseTimeSeriesModel):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="CatBoost", category="ML", params=params or {})

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CatBoostModel":
        from catboost import CatBoostRegressor
        p = {**self.params, "verbose": 0}
        self.model = CatBoostRegressor(**p)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test)

    def get_params(self) -> Dict[str, Any]:
        return self.params


# ── Registry: quick lookup by name ───────────────────────────────────────────
ML_MODEL_REGISTRY = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "gradient_boosting": GradientBoostingModel,
    "ridge": RidgeModel,
    "svr": SVRModel,
    "catboost": CatBoostModel,
}
