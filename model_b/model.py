import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback : i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


class GRUModel:
    def __init__(self, lookback=30, hidden_size=32, dropout=0.1,
                 epochs=100, batch_size=32, lr=0.001, patience=15):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.model = None
        self._scaler_X = None
        self._scaler_y = None
        self._X_train_raw = None
        self._n_features = None

    def _build_model(self, input_shape):
        import tensorflow as tf
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.GRU(self.hidden_size, dropout=self.dropout)(inputs)
        x = tf.keras.layers.Dense(1)(x)
        self.model = tf.keras.Model(inputs, x)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss="mse",
            metrics=["mae"],
        )

    def fit(self, X, y):
        import tensorflow as tf
        tf.random.set_seed(42)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._n_features = X.shape[1]
        self._X_train_raw = X

        self._scaler_X = MinMaxScaler()
        self._scaler_y = MinMaxScaler()
        X_s = self._scaler_X.fit_transform(X)
        y_s = self._scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        Xs, ys = create_sequences(X_s, y_s, self.lookback)

        val_split = max(1, int(len(Xs) * 0.1))
        X_tr, X_val = Xs[:-val_split], Xs[-val_split:]
        y_tr, y_val = ys[:-val_split], ys[-val_split:]

        self._build_model(input_shape=(self.lookback, self._n_features))

        history = self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=self.patience,
                    restore_best_weights=True, verbose=0
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5,
                    patience=self.patience // 2, verbose=0
                ),
            ],
            verbose=0,
        )
        final_val_loss = min(history.history["val_loss"])
        print(f"  GRU best val_loss={final_val_loss:.6f}")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X_s = self._scaler_X.transform(X)

        X_context = np.vstack([
            self._scaler_X.transform(self._X_train_raw[-self.lookback:]),
            X_s,
        ])

        preds = []
        for i in range(self.lookback, len(X_context)):
            seq = X_context[i - self.lookback : i][np.newaxis]
            pred = self.model.predict(seq, verbose=0)[0][0]
            preds.append(pred)

        preds = np.array(preds[-len(X):])
        return self._scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()

    def save(self, path):
        self.model.save(f"{path}_gru.keras")
        joblib.dump(self._scaler_X, f"{path}_scaler_X.pkl")
        joblib.dump(self._scaler_y, f"{path}_scaler_y.pkl")
        joblib.dump({
            "lookback": self.lookback,
            "hidden_size": self.hidden_size,
            "_n_features": self._n_features,
            "_X_train_raw": self._X_train_raw,
        }, f"{path}_params.pkl")

    def load(self, path):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(f"{path}_gru.keras")
        self._scaler_X = joblib.load(f"{path}_scaler_X.pkl")
        self._scaler_y = joblib.load(f"{path}_scaler_y.pkl")
        params = joblib.load(f"{path}_params.pkl")
        self.lookback = params["lookback"]
        self.hidden_size = params["hidden_size"]
        self._n_features = params["_n_features"]
        self._X_train_raw = params["_X_train_raw"]
        return self