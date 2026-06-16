import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


class RidgeRFEnsemble:
    def __init__(self, w_ridge=0.7, w_rf=0.3, ridge_alpha=506.16,
                 rf_n_estimators=200, rf_max_depth=10):
        self.w_ridge = w_ridge
        self.w_rf = w_rf
        self.ridge = Ridge(alpha=ridge_alpha)
        self.rf = RandomForestRegressor(
            n_estimators=rf_n_estimators, max_depth=rf_max_depth,
            random_state=42, n_jobs=-1
        )

    def fit(self, X_ridge, X_rf, y):
        self.ridge.fit(X_ridge, y)
        self.rf.fit(X_rf, y)
        return self

    def predict(self, X_ridge, X_rf):
        ridge_pred = self.ridge.predict(X_ridge)
        rf_pred = self.rf.predict(X_rf)
        return self.w_ridge * ridge_pred + self.w_rf * rf_pred

    def get_params(self):
        return {
            "w_ridge": self.w_ridge,
            "w_rf": self.w_rf,
            "ridge_alpha": self.ridge.alpha,
            "rf_n_estimators": self.rf.n_estimators,
            "rf_max_depth": self.rf.max_depth,
        }