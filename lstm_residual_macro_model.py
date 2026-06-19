#!/usr/bin/env python3
from __future__ import annotations

import math
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import assumption_driven_experiment as ade

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

ROOT = Path(".")
TRAIN_CSV = ROOT / "data_train.csv"
TEST_EXOG_CSV = ROOT / "data_test.csv"
TEST_ACTUAL_CSV = ROOT / "data_test_actual.csv"
DATE_COL = "Date"
TARGET_COL = "USDIDR"

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_exog = pd.read_csv(TEST_EXOG_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test_actual = pd.read_csv(TEST_ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return train, test_exog, test_actual

def prepare_combined(train: pd.DataFrame, test_exog: pd.DataFrame) -> pd.DataFrame:
    combined_raw = pd.concat([train, test_exog], ignore_index=True)
    combined_raw[DATE_COL] = pd.to_datetime(combined_raw[DATE_COL])
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        combined_raw[f"{col}_ret"] = np.log(combined_raw[col]).diff().fillna(0.0)
    combined_raw["bi_rate_change"] = combined_raw["BI_rate"].diff().fillna(0.0)
    for col in [c for c in combined_raw.columns if "ret" in c or "change" in c]:
        combined_raw[f"{col}_lag1"] = combined_raw[col].shift(1).fillna(0.0)
    return ade.make_causal_exog(combined_raw)

def build_trend_table(train_df: pd.DataFrame, combined: pd.DataFrame, selected_lags: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    levels = train_df[TARGET_COL].astype(float).tolist()
    diffs = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
    rows = []
    ys = []
    start = max(max(selected_lags, default=1), 252)
    for t in range(start, len(train_df)):
        feats = ade.build_row_features(combined.iloc[t], levels[:t], diffs[: t - 1], selected_lags, [], "trend")
        rows.append(feats)
        ys.append(float(math.log(levels[t] / levels[t - 1])))
    X = pd.DataFrame(rows).fillna(0.0)
    y = pd.Series(ys, dtype=float)
    return X, y

def build_residual_table(
    train_df: pd.DataFrame,
    combined: pd.DataFrame,
    trend_model,
    trend_X: pd.DataFrame,
    trend_y: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    trend_preds = trend_model.predict(trend_X)
    residuals = trend_y - trend_preds
    start = len(train_df) - len(trend_y)
    rows = []
    for t in range(start, len(train_df)):
        row_exog = combined.iloc[t]
        feats = {
            "SP500_ret_lag1": float(row_exog.get("SP500_ret_lag1", 0.0)),
            "VIX_ret_lag1": float(row_exog.get("VIX_ret_lag1", 0.0)),
            "bi_rate_change_lag1": float(row_exog.get("bi_rate_change_lag1", 0.0))
        }
        rows.append(feats)
    X = pd.DataFrame(rows)
    y = pd.Series(residuals)
    return X, y

def build_lstm_dataset(X_res: pd.DataFrame, y_res: pd.Series, time_steps: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Reshape tabular residual data to 3D temporal arrays [samples, time_steps, features] for LSTM."""
    # Convert DataFrame to array
    vals = X_res.values
    ys = y_res.values
    X_lstm = []
    y_lstm = []
    for i in range(time_steps, len(vals)):
        X_lstm.append(vals[i - time_steps:i])
        y_lstm.append(ys[i])
    return np.array(X_lstm), np.array(y_lstm)

def recursive_lstm_forecast(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    combined: pd.DataFrame,
    trend_model,
    lstm_model,
    scaler,
    selected_lags: list[int],
    time_steps: int = 5,
    vix_fac: float = 1.10,
    spread_fac: float = 1.06
) -> np.ndarray:
    history = train_df[TARGET_COL].astype(float).tolist()
    diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    trend_cols = trend_model.feature_names_in_
    
    # Pre-collect future exog features to feed temporal windows
    total_idx = len(combined)
    exog_list = []
    for idx in range(total_idx):
        row_exog = combined.iloc[idx]
        exog_list.append([
            float(row_exog.get("SP500_ret_lag1", 0.0)),
            float(row_exog.get("VIX_ret_lag1", 0.0)),
            float(row_exog.get("bi_rate_change_lag1", 0.0))
        ])
    exog_arr = scaler.transform(exog_list)  # Standardize all features beforehand
    
    preds = []
    for i in range(len(future_df)):
        idx = len(train_df) + i
        row_exog = combined.iloc[idx]
        
        # 1. Trend Prediction
        feats_trend = ade.build_row_features(row_exog, history, diffs, selected_lags, [], "trend")
        X_row_trend = pd.DataFrame([feats_trend]).reindex(columns=trend_cols, fill_value=0.0)
        ret_trend = float(trend_model.predict(X_row_trend)[0])
        
        # 2. LSTM Residual Prediction (Using 3D sliding window of scaled exog)
        # Select temporal window of size [time_steps, features]
        window_exog = exog_arr[idx - time_steps:idx]
        window_exog_3d = np.expand_dims(window_exog, axis=0) # [1, time_steps, features]
        ret_shock = float(lstm_model.predict(window_exog_3d, verbose=0)[0, 0])
        
        ret_total = ret_trend + ret_shock
        
        # Apply Gates
        vix_lag1 = float(row_exog.get("VIX_lag1", 18.0))
        bi_rate = float(row_exog.get("BI_rate", 5.75))
        us_rate = float(row_exog.get("US_rate", 5.08))
        spread = bi_rate - us_rate
        
        if ret_total > 0:
            if vix_lag1 > 14.0:
                ret_total *= vix_fac
            if spread < 0.8:
                ret_total *= spread_fac
                
        next_level = float(history[-1] * math.exp(ret_total))
        preds.append(next_level)
        history.append(next_level)
        diffs.append(next_level - history[-2])
        
    return np.asarray(preds, dtype=float)

def main() -> None:
    train, test_exog, test_actual = load_data()
    y_true = test_actual[TARGET_COL].astype(float).to_numpy(dtype=float)
    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]
    combined = prepare_combined(train, test_exog)
    
    # Fit Trend Model
    X_trend, y_trend = build_trend_table(train, combined, selected_lags)
    trend_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]).fit(X_trend, y_trend)
    
    # Fit Residual Model
    X_res, y_res = build_residual_table(train, combined, trend_model, X_trend, y_trend)
    
    # Scaling exog features
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res.values)
    X_res_scaled_df = pd.DataFrame(X_res_scaled, columns=X_res.columns)
    
    # Convert data into 3D LSTM Format (time_steps = 5 days)
    time_steps = 5
    X_lstm, y_lstm = build_lstm_dataset(X_res_scaled_df, y_res, time_steps=time_steps)
    
    # Build LSTM Model for residuals
    lstm_model = Sequential([
        LSTM(16, input_shape=(time_steps, X_lstm.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=0.005), loss="mse")
    lstm_model.fit(X_lstm, y_lstm, epochs=15, batch_size=32, verbose=1)
    
    # Run Forecast with LSTM residual Shock model
    lstm_preds = recursive_lstm_forecast(
        train, test_exog, combined, trend_model, lstm_model, scaler, selected_lags, time_steps=time_steps
    )
    lstm_rmse = rmse(y_true, lstm_preds)
    print(f"\nHybrid Ridge-LSTM Residual Gated Model OOS RMSE: {lstm_rmse:.4f}")

if __name__ == "__main__":
    main()
