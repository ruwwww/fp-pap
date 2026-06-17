import warnings
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import pacf


class ElasticNetPAC:
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000, tol=1e-4)
        self._scaler = StandardScaler()
        self.pacf_values = None
        self.sample_weights = None

    def _compute_pac_weights(self, y):
        nlags = min(30, len(y) // 3)
        self.pacf_values = pacf(y, nlags=nlags)
        abs_pacf = np.abs(self.pacf_values)
        n = len(y)
        weights = np.ones(n)
        for i in range(n):
            lag_idx = min(i + 1, nlags)
            weights[i] = abs_pacf[lag_idx]
        weights = weights / weights.sum() * n
        self.sample_weights = weights
        return weights

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_s = self._scaler.fit_transform(X)
        weights = self._compute_pac_weights(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            self.model.fit(X_s, y, sample_weight=weights)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X_s = self._scaler.transform(X)
        return self.model.predict(X_s)

    def get_params(self):
        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "coef": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
        }