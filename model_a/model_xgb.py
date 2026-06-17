import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler


class XGBoostPAP:
    def __init__(self, n_estimators=1000, max_depth=5, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        self.model = xgb.XGBRegressor(**self.params)
        self._scaler = StandardScaler()

    def fit(self, X, y, eval_set=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_s = self._scaler.fit_transform(X)

        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_s = self._scaler.transform(np.asarray(X_val, dtype=float))
            self.model.set_params(early_stopping_rounds=50)
            self.model.fit(X_s, y, eval_set=[(X_val_s, y_val)], verbose=False)
            self.model.set_params(n_estimators=self.model.best_iteration + 1)
        else:
            self.model.set_params(n_estimators=500, early_stopping_rounds=None)
            self.model.fit(X_s, y, verbose=False)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X_s = self._scaler.transform(X)
        return self.model.predict(X_s)
