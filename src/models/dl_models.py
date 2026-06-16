"""
src/models/dl_models.py
────────────────────────
Model B — Deep Learning Models (Keras/TensorFlow)
LSTM, GRU, CNN-LSTM, BiLSTM, Transformer

All models:
  - Accept tabular X_train/y_train for API consistency
  - Internally reshape to 3D sequences using 'lookback' window
  - Use EarlyStopping to prevent overfitting
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from .base_model import BaseTimeSeriesModel

logger = logging.getLogger(__name__)

# Suppress TF verbose logs
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ── Helpers ────────────────────────────────────────────────────────────────
def create_sequences(
    X: np.ndarray, y: np.ndarray, lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert (samples, features) → (samples, lookback, features) sequences.

    ⚠️  ANTI-LEAKAGE: Each sequence only includes PAST lookback steps.
    """
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback : i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def get_callbacks(patience: int = 15, monitor: str = "val_loss"):
    """Standard Keras callbacks for training stability."""
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=patience // 2, verbose=0
        ),
    ]


# ── Base DL Model ──────────────────────────────────────────────────────────
class BaseDLModel(BaseTimeSeriesModel):
    """
    Shared logic for all Keras-based DL models:
      - sequence creation, scaler, fit, predict
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name=name, category="DL", params=params)
        self.lookback: int = params.get("lookback", 30)
        self.epochs: int = params.get("epochs", 100)
        self.batch_size: int = params.get("batch_size", 32)
        self.learning_rate: float = params.get("learning_rate", 0.001)
        self.patience: int = params.get("patience", 15)
        self._scaler_X = None
        self._scaler_y = None
        self._n_features: int = 0
        self._X_train_raw: np.ndarray = None  # kept for autoregressive predict

    def _scale(self, X: np.ndarray, y: np.ndarray, fit: bool = True):
        from sklearn.preprocessing import MinMaxScaler
        if fit:
            self._scaler_X = MinMaxScaler()
            self._scaler_y = MinMaxScaler()
            X_s = self._scaler_X.fit_transform(X)
            y_s = self._scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        else:
            X_s = self._scaler_X.transform(X)
            y_s = None  # not needed for predict
        return X_s, y_s

    def _inverse_scale_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return self._scaler_y.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).flatten()

    def _build_model(self, input_shape: Tuple) -> None:
        """Override in subclasses to define the Keras model."""
        raise NotImplementedError

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseDLModel":
        import tensorflow as tf
        tf.random.set_seed(42)

        X = X_train.values.astype(float)
        y = y_train.values.astype(float)
        self._n_features = X.shape[1]
        self._X_train_raw = X  # save for iterative future prediction

        X_s, y_s = self._scale(X, y, fit=True)
        Xs, ys = create_sequences(X_s, y_s, self.lookback)

        # Validation split (last 10% of training sequences)
        val_split = max(1, int(len(Xs) * 0.1))
        X_tr, X_val = Xs[:-val_split], Xs[-val_split:]
        y_tr, y_val = ys[:-val_split], ys[-val_split:]

        self._build_model(input_shape=(self.lookback, self._n_features))
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        history = self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=get_callbacks(self.patience),
            verbose=0,
        )
        final_val_loss = min(history.history["val_loss"])
        logger.info(f"[{self.name}] Best val_loss={final_val_loss:.6f}")
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        X = X_test.values.astype(float)
        X_s, _ = self._scale(X, np.zeros(len(X)), fit=False)

        # Prepend last `lookback` steps from training data for context
        X_context = np.vstack([
            self._scaler_X.transform(self._X_train_raw[-self.lookback:]),
            X_s,
        ])

        preds = []
        for i in range(self.lookback, len(X_context)):
            seq = X_context[i - self.lookback : i][np.newaxis]  # (1, lookback, features)
            pred = self.model.predict(seq, verbose=0)[0][0]
            preds.append(pred)

        preds = np.array(preds[-len(X_test):])  # align to test length
        return self._inverse_scale_y(preds)

    def get_params(self) -> Dict[str, Any]:
        return self.params


# ── LSTM ───────────────────────────────────────────────────────────────────
class LSTMModel(BaseDLModel):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="LSTM", params=params or {})

    def _build_model(self, input_shape: Tuple) -> None:
        import tensorflow as tf
        units: List[int] = self.params.get("units", [128, 64])
        dropout: float = self.params.get("dropout", 0.2)

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for i, u in enumerate(units):
            return_seq = i < len(units) - 1
            x = tf.keras.layers.LSTM(u, return_sequences=return_seq, dropout=dropout)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dense(1)(x)
        self.model = tf.keras.Model(inputs, x)


# ── GRU ───────────────────────────────────────────────────────────────────
class GRUModel(BaseDLModel):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="GRU", params=params or {})

    def _build_model(self, input_shape: Tuple) -> None:
        import tensorflow as tf
        units: List[int] = self.params.get("units", [128, 64])
        dropout: float = self.params.get("dropout", 0.2)

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for i, u in enumerate(units):
            return_seq = i < len(units) - 1
            x = tf.keras.layers.GRU(u, return_sequences=return_seq, dropout=dropout)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dense(1)(x)
        self.model = tf.keras.Model(inputs, x)


# ── CNN-LSTM (hybrid) ──────────────────────────────────────────────────────
class CNNLSTMModel(BaseDLModel):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="CNN-LSTM", params=params or {})

    def _build_model(self, input_shape: Tuple) -> None:
        import tensorflow as tf
        filters: int = self.params.get("filters", 64)
        kernel_size: int = self.params.get("kernel_size", 3)
        lstm_units: List[int] = self.params.get("lstm_units", [64, 32])
        dropout: float = self.params.get("dropout", 0.2)

        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv1D(filters, kernel_size, activation="relu", padding="same")(inputs)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
        x = tf.keras.layers.Conv1D(filters // 2, kernel_size, activation="relu", padding="same")(x)
        for i, u in enumerate(lstm_units):
            return_seq = i < len(lstm_units) - 1
            x = tf.keras.layers.LSTM(u, return_sequences=return_seq, dropout=dropout)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dense(1)(x)
        self.model = tf.keras.Model(inputs, x)


# ── Bidirectional LSTM ─────────────────────────────────────────────────────
class BiLSTMModel(BaseDLModel):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="BiLSTM", params=params or {})

    def _build_model(self, input_shape: Tuple) -> None:
        import tensorflow as tf
        units: List[int] = self.params.get("units", [128, 64])
        dropout: float = self.params.get("dropout", 0.2)

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for i, u in enumerate(units):
            return_seq = i < len(units) - 1
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(u, return_sequences=return_seq, dropout=dropout)
            )(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dense(1)(x)
        self.model = tf.keras.Model(inputs, x)


# ── Simple RNN ─────────────────────────────────────────────────────────────
class RNNModel(BaseDLModel):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="RNN", params=params or {})

    def _build_model(self, input_shape: Tuple) -> None:
        import tensorflow as tf
        units: List[int] = self.params.get("units", [64, 32])
        dropout: float = self.params.get("dropout", 0.2)

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for i, u in enumerate(units):
            return_seq = i < len(units) - 1
            x = tf.keras.layers.SimpleRNN(u, return_sequences=return_seq, dropout=dropout)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dense(1)(x)
        self.model = tf.keras.Model(inputs, x)


# ── Registry ───────────────────────────────────────────────────────────────
DL_MODEL_REGISTRY = {
    "lstm": LSTMModel,
    "gru": GRUModel,
    "cnn_lstm": CNNLSTMModel,
    "bilstm": BiLSTMModel,
    "rnn": RNNModel,
}
