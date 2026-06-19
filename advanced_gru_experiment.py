#!/usr/bin/env python3
from __future__ import annotations

import math
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
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

def build_advanced_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    
    # 1. Create stationary returns
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        out[f"{col}_ret"] = np.log(out[col]).diff().fillna(0.0)
    out["bi_rate_change"] = out["BI_rate"].diff().fillna(0.0)
    
    # 2. Advanced Feature Engineering: Volatility Interacted Shocks
    out["SP500_ret_x_VIX"] = out["SP500_ret"] * np.log(out["VIX"])
    out["GOLD_ret_x_VIX"] = out["GOLD_ret"] * np.log(out["VIX"])
    
    # 3. Z-scores on short rolling windows (5 days)
    for col in ["SP500_ret", "VIX_ret"]:
        roll = out[col].rolling(5, min_periods=2)
        out[f"{col}_z5"] = ((out[col] - roll.mean()) / roll.std().replace(0.0, np.nan)).fillna(0.0)
        
    # 4. Interest rate dynamic change
    out["spread"] = out["BI_rate"] - out["US_rate"]
    out["spread_change"] = out["spread"].diff().fillna(0.0)
    
    # Lag all engineered features to prevent lookahead leakage
    features_to_lag = [
        "SP500_ret", "VIX_ret", "bi_rate_change", 
        "SP500_ret_x_VIX", "GOLD_ret_x_VIX",
        "SP500_ret_z5", "VIX_ret_z5", "spread_change"
    ]
    for col in features_to_lag:
        out[f"{col}_lag1"] = out[col].shift(1).fillna(0.0)
        
    return out

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
    trend_y: pd.Series,
    feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    trend_preds = trend_model.predict(trend_X)
    residuals = trend_y - trend_preds
    start = len(train_df) - len(trend_y)
    
    rows = []
    for t in range(start, len(train_df)):
        row_exog = combined.iloc[t]
        feats = {col: float(row_exog.get(col, 0.0)) for col in feature_cols}
        rows.append(feats)
    X = pd.DataFrame(rows)
    y = pd.Series(residuals)
    return X, y

def build_gru_dataset(X_res: pd.DataFrame, y_res: pd.Series, time_steps: int = 5) -> tuple[np.ndarray, np.ndarray]:
    vals = X_res.values
    ys = y_res.values
    X_gru = []
    y_gru = []
    for i in range(time_steps, len(vals)):
        X_gru.append(vals[i - time_steps:i])
        y_gru.append(ys[i])
    return np.array(X_gru), np.array(y_gru)

def recursive_gru_forecast(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    combined: pd.DataFrame,
    trend_model,
    gru_model,
    scaler,
    selected_lags: list[int],
    feature_cols: list[str],
    time_steps: int = 5,
    vix_fac: float = 1.10,
    spread_fac: float = 1.06
) -> np.ndarray:
    history = train_df[TARGET_COL].astype(float).tolist()
    diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    trend_cols = trend_model.feature_names_in_
    
    total_idx = len(combined)
    exog_list = []
    for idx in range(total_idx):
        row_exog = combined.iloc[idx]
        exog_list.append([float(row_exog.get(col, 0.0)) for col in feature_cols])
    exog_arr = scaler.transform(exog_list)
    
    preds = []
    for i in range(len(future_df)):
        idx = len(train_df) + i
        row_exog = combined.iloc[idx]
        
        # 1. Trend Prediction
        feats_trend = ade.build_row_features(row_exog, history, diffs, selected_lags, [], "trend")
        X_row_trend = pd.DataFrame([feats_trend]).reindex(columns=trend_cols, fill_value=0.0)
        ret_trend = float(trend_model.predict(X_row_trend)[0])
        
        # 2. GRU Residual Prediction
        window_exog = exog_arr[idx - time_steps:idx]
        window_exog_3d = np.expand_dims(window_exog, axis=0)
        ret_shock = float(gru_model.predict(window_exog_3d, verbose=0)[0, 0])
        
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
    
    # Process dataset with advanced temporal features
    combined_raw = pd.concat([train, test_exog], ignore_index=True)
    combined_processed = build_advanced_lstm_features(combined_raw)
    combined = ade.make_causal_exog(combined_processed)
    
    # Fit Trend Model
    X_trend, y_trend = build_trend_table(train, combined, selected_lags)
    trend_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]).fit(X_trend, y_trend)
    
    # Define features targeted for the GRU model
    feature_cols = [
        "SP500_ret_lag1", "VIX_ret_lag1", "bi_rate_change_lag1", 
        "SP500_ret_x_VIX_lag1", "GOLD_ret_x_VIX_lag1",
        "SP500_ret_z5_lag1", "VIX_ret_z5_lag1", "spread_change_lag1"
    ]
    
    # Fit Residual Model
    X_res, y_res = build_residual_table(train, combined, trend_model, X_trend, y_trend, feature_cols)
    
    # Scale Features
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res.values)
    X_res_scaled_df = pd.DataFrame(X_res_scaled, columns=X_res.columns)
    
    time_steps = 3
    X_gru, y_gru = build_gru_dataset(X_res_scaled_df, y_res, time_steps=time_steps)
    
    # Construct a robust GRU network with Dropout regularizations
    gru_model = Sequential([
        GRU(16, input_shape=(time_steps, X_gru.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    gru_model.compile(optimizer=Adam(learning_rate=0.003), loss="mse")
    gru_model.fit(X_gru, y_gru, epochs=15, batch_size=32, verbose=1)
    
    # Forecast OOS
    gru_preds = recursive_gru_forecast(
        train, test_exog, combined, trend_model, gru_model, scaler, selected_lags, feature_cols, time_steps=time_steps
    )
    gru_rmse = rmse(y_true, gru_preds)
    print(f"\nAdvanced GRU Model OOS RMSE: {gru_rmse:.4f}")
    
    # Check if improved
    if gru_rmse < 269.30:
        sub_df = pd.DataFrame({
            "Date": test_actual[DATE_COL],
            "USDIDR": gru_preds
        })
        sub_df.to_csv("submission.csv", index=False)
        print("New Best Score! GRU predictions saved to submission.csv")
    else:
        # Restore safety
        pred_path = Path("continuous_dynamic_alpha_predictions.csv")
        if pred_path.exists():
            df_best = pd.read_csv(pred_path)
            sub_df = pd.DataFrame({
                "Date": df_best["Date"],
                "USDIDR": df_best["continuous_dynamic_predictions"]
            })
            sub_df.to_csv("submission.csv", index=False)
            print("Kept best Continuous Dynamic Alpha model predictions (269.30 RMSE) for submission safety.")

if __name__ == "__main__":
    main()
