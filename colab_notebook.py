#!/usr/bin/env python3
"""
USD/IDR Forecasting & Model Reconstruction Notebook for Google Colab
-------------------------------------------------------------------
This file is designed to be self-contained and run on Google Colab.
It includes:
1. Reconstruction of the best model (Two-Stage Decoupled Ridge Model with CV Gating and Bias Correction).
2. Skenario Dataset splits (80-20, 70-30, 60-40) evaluated on 3 models:
   - Model A: Machine Learning (Two-Stage Decoupled Ridge with 3-Layer Bias Correction)
   - Model B: Deep Learning (Ridge Trend + GRU Residual with Bias Correction)
   - Ensemble: Hybrid (Average of Model A and Model B)
3. Comprehensive evaluation using RMSE, MAE, MAPE, and R-squared.
4. Trajectory plotting showing actual training values transitioning into test predictions.
"""

import os
# Force CPU execution for local runs to avoid GPU PTX compilation delays
if not os.environ.get("COLAB_GPU", "") and not os.path.exists("/content"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# TensorFlow/Keras for Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Suppress warnings and set random seed for reproducibility
warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

# Global configuration
TARGET_COL = "USDIDR"
DATE_COL = "Date"
SELECTED_LAGS = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]
EXOG_COLS = ["SP500_ret_lag1", "VIX_ret_lag1", "bi_rate_change_lag10", "IHSG_ret_lag1", "OIL_ret_lag1"]

# Optimal Hyperparameters chosen strictly via Train CV
BEST_VIX = 1.05
BEST_SPREAD = 1.02
BEST_BETA = 0.25

# Metrics Calculations
def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    val_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    val_mae = float(mean_absolute_error(y_true, y_pred))
    val_mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    val_r2 = float(r2_score(y_true, y_pred))
    
    return {
        "RMSE": val_rmse,
        "MAE": val_mae,
        "MAPE": val_mape,
        "R2": val_r2
    }

# Setup files for Colab
def check_and_load_files():
    """Helper to check if CSV files are present, or download them from Github, or ask user to upload them."""
    import urllib.request
    
    urls = {
        "data_train.csv": "https://raw.githubusercontent.com/ruwwww/fp-pap/refs/heads/ubuntu/data_train.csv",
        "data_test.csv": "https://raw.githubusercontent.com/ruwwww/fp-pap/refs/heads/ubuntu/data_test.csv",
        "data_test_actual.csv": "https://raw.githubusercontent.com/ruwwww/fp-pap/refs/heads/ubuntu/data_test_actual.csv"
    }
    
    for filename, url in urls.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename} from Github raw...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"Successfully downloaded {filename}!")
            except Exception as e:
                print(f"Failed to download {filename} from Github: {e}")
                
    train_exists = os.path.exists("data_train.csv")
    test_exists = os.path.exists("data_test.csv")
    actual_exists = os.path.exists("data_test_actual.csv")
    
    if not train_exists or not test_exists:
        try:
            from google.colab import files
            print("CSV files not found. Please upload data_train.csv and data_test.csv:")
            uploaded = files.upload()
        except ImportError:
            print("Error: data_train.csv and data_test.csv are required to run this script.")
            # Create synthetic data to prevent script crash if run in a non-interactive/test environment
            print("Creating dummy synthetic data for demonstration purposes...")
            dates_train = pd.date_range(start="2010-01-01", periods=1000, freq="B")
            dummy_train = pd.DataFrame({
                "Date": dates_train,
                "OIL": np.random.normal(70, 5, 1000),
                "GOLD": np.random.normal(1500, 100, 1000),
                "USDIDR": np.linspace(9000, 15000, 1000) + np.random.normal(0, 100, 1000),
                "SP500": np.random.normal(3000, 200, 1000),
                "IHSG": np.random.normal(5000, 300, 1000),
                "VIX": np.random.normal(15, 2, 1000),
                "CPI": np.random.normal(3, 0.5, 1000),
                "BI_rate": np.random.normal(6, 0.5, 1000),
                "US_rate": np.random.normal(2, 0.5, 1000)
            })
            dummy_train.to_csv("data_train.csv", index=False)
            
            dates_test = pd.date_range(start="2023-06-01", periods=200, freq="B")
            dummy_test = pd.DataFrame({
                "Date": dates_test,
                "OIL": np.random.normal(75, 5, 200),
                "GOLD": np.random.normal(1900, 100, 200),
                "SP500": np.random.normal(4200, 200, 200),
                "IHSG": np.random.normal(6700, 300, 200),
                "VIX": np.random.normal(14, 2, 200),
                "CPI": np.random.normal(3.5, 0.5, 200),
                "BI_rate": np.random.normal(5.75, 0.5, 200),
                "US_rate": np.random.normal(5, 0.5, 200)
            })
            dummy_test.to_csv("data_test.csv", index=False)
            
    train = pd.read_csv("data_train.csv")
    test = pd.read_csv("data_test.csv")
    train[DATE_COL] = pd.to_datetime(train[DATE_COL])
    test[DATE_COL] = pd.to_datetime(test[DATE_COL])
    train = train.sort_values(DATE_COL).reset_index(drop=True)
    test = test.sort_values(DATE_COL).reset_index(drop=True)
    
    test_actual = None
    if actual_exists:
        test_actual = pd.read_csv("data_test_actual.csv")
        test_actual[DATE_COL] = pd.to_datetime(test_actual[DATE_COL])
        test_actual = test_actual.sort_values(DATE_COL).reset_index(drop=True)
        
    return train, test, test_actual

# Feature Engineering
def prepare_features(train, test_exog):
    """Creates lagged exogenous features and log returns safely across the dataset."""
    combined = pd.concat([train, test_exog], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    
    # Fill any NaNs
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX", "BI_rate", "US_rate"]:
        if col in combined.columns:
            combined[col] = combined[col].ffill().bfill()
            
    # 1. Stationary Log-Returns
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        combined[f"{col}_ret"] = np.log(combined[col]).diff().fillna(0.0)
    combined["bi_rate_change"] = combined["BI_rate"].diff().fillna(0.0)
    
    # 2. Causal Exogenous Lags
    combined["SP500_ret_lag1"] = combined["SP500_ret"].shift(1).fillna(0.0)
    combined["VIX_ret_lag1"] = combined["VIX_ret"].shift(1).fillna(0.0)
    combined["bi_rate_change_lag10"] = combined["bi_rate_change"].shift(10).fillna(0.0)
    combined["IHSG_ret_lag1"] = combined["IHSG_ret"].shift(1).fillna(0.0)
    combined["OIL_ret_lag1"] = combined["OIL_ret"].shift(1).fillna(0.0)
    combined["VIX_lag1"] = combined["VIX"].shift(1).fillna(15.0)
    
    return combined

def build_row_features(exog_row, level_hist, diff_hist, selected_lags, selected_exog, feature_mode):
    """Assembles all target and exogenous features for a single day prediction step."""
    # Custom helper imports from main project files if exists, else build row-wise
    level = np.asarray(level_hist, dtype=float)
    diff = np.asarray(diff_hist, dtype=float)
    feats = {}
    
    # Target lags
    for lag in selected_lags:
        feats[f"diff_lag{lag}"] = float(diff[-lag]) if len(diff) >= lag else 0.0
        
    if feature_mode in {"trend", "full"}:
        # Rolling stats
        if len(diff) >= 20:
            feats["rolling_mean_diff_20"] = float(np.mean(diff[-20:]))
            feats["rolling_std_diff_20"] = float(np.std(diff[-20:], ddof=0))
        else:
            feats["rolling_mean_diff_20"] = 0.0
            feats["rolling_std_diff_20"] = 0.0
            
        if len(diff) >= 252:
            feats["rolling_mean_diff_252"] = float(np.mean(diff[-252:]))
        else:
            feats["rolling_mean_diff_252"] = float(np.mean(diff)) if len(diff) else 0.0
            
        if len(level) >= 252:
            ma252 = float(np.mean(level[-252:]))
            feats["gap_from_trend"] = float(level[-1] - ma252)
        else:
            feats["gap_from_trend"] = 0.0
            
        if len(level) >= 90:
            ma90 = float(np.mean(level[-90:]))
            sd90 = float(np.std(level[-90:], ddof=0))
            z = (level[-1] - ma90) / sd90 if sd90 > 0 else 0.0
            threshold = 1.5
            feats["extreme_high"] = float(max(0.0, z - threshold))
            feats["extreme_low"] = float(min(0.0, z + threshold))
        else:
            feats["extreme_high"] = 0.0
            feats["extreme_low"] = 0.0
            
    for name in selected_exog:
        feats[name] = float(exog_row.get(name, 0.0))
        
    return feats

# Tabular Data Preparation for Models
def build_trend_dataset(train_df, combined, selected_lags):
    levels = train_df[TARGET_COL].astype(float).tolist()
    diffs = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
    X_rows = []
    y_vals = []
    start_idx = max(max(selected_lags), 252)
    
    # Fit row-wise features
    import assumption_driven_experiment as ade
    for t in range(start_idx, len(train_df)):
        row_exog = combined.iloc[t]
        feats = build_row_features(row_exog, levels[:t], diffs[:t-1], selected_lags, [], "trend")
        X_rows.append(feats)
        y_vals.append(float(math.log(levels[t] / levels[t - 1])))
        
    X = pd.DataFrame(X_rows).fillna(0.0)
    y = pd.Series(y_vals)
    return X, y, start_idx

def build_residual_dataset(train_df, combined, start_idx, trend_preds, y_trend):
    residuals = y_trend - trend_preds
    X_rows = []
    for t in range(start_idx, len(train_df)):
        row_exog = combined.iloc[t]
        feats = {
            "SP500_ret_lag1": float(row_exog.get("SP500_ret_lag1", 0.0)),
            "VIX_ret_lag1": float(row_exog.get("VIX_ret_lag1", 0.0)),
            "bi_rate_change_lag10": float(row_exog.get("bi_rate_change_lag10", 0.0)),
            "IHSG_ret_lag1": float(row_exog.get("IHSG_ret_lag1", 0.0)),
            "OIL_ret_lag1": float(row_exog.get("OIL_ret_lag1", 0.0))
        }
        X_rows.append(feats)
    X = pd.DataFrame(X_rows)
    y = pd.Series(residuals)
    return X, y

def build_gru_dataset(X_res, y_res, time_steps=5):
    vals = X_res.values
    ys = y_res.values
    X_3d, y_out = [], []
    for i in range(time_steps, len(vals)):
        X_3d.append(vals[i - time_steps:i])
        y_out.append(ys[i])
    return np.array(X_3d), np.array(y_out)

def fit_bias_models(train_df, combined, trend_model, res_model, vix_fac, spread_fac, selected_lags, trend_cols, res_cols):
    """Trains 3-layer bias models on rolling training simulation errors."""
    levels = train_df[TARGET_COL].astype(float).tolist()
    
    sim_len_segments = min(len(train_df) // 3, 754)
    step_size = 126
    start_indices = list(range(252, len(train_df) - sim_len_segments, step_size))
    if not start_indices:
        start_indices = [252]
        sim_len_segments = len(train_df) - 253
        
    bias_samples = []
    scaler_trend = trend_model.named_steps["scaler"]
    ridge_trend = trend_model.named_steps["model"]
    scaler_res = res_model.named_steps["scaler"]
    ridge_res = res_model.named_steps["model"]
    
    for start_idx in start_indices:
        history_sim = list(levels[:start_idx])
        diffs_sim = [history_sim[j] - history_sim[j - 1] for j in range(1, len(history_sim))]
        preds_segment = []
        actuals_segment = levels[start_idx : start_idx + sim_len_segments]
        
        for k in range(sim_len_segments):
            t = start_idx + k
            row_exog = combined.iloc[t]
            
            # Predict trend
            feats_trend = build_row_features(row_exog, history_sim, diffs_sim, selected_lags, [], "trend")
            trend_vec = np.array([[feats_trend.get(col, 0.0) for col in trend_cols]])
            trend_scaled = scaler_trend.transform(trend_vec)
            ret_trend = float(ridge_trend.predict(trend_scaled)[0])
            
            # Predict residual
            feats_res = {
                "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
                "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
                "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"],
                "IHSG_ret_lag1": row_exog["IHSG_ret_lag1"],
                "OIL_ret_lag1": row_exog["OIL_ret_lag1"]
            }
            res_vec = np.array([[feats_res.get(col, 0.0) for col in res_cols]])
            res_scaled = scaler_res.transform(res_vec)
            ret_shock = float(ridge_res.predict(res_scaled)[0])
            
            ret_total = ret_trend + ret_shock
            vix_lag1 = float(row_exog.get("VIX_lag1", 15.0))
            spread = float(row_exog.get("BI_rate", 5.75)) - float(row_exog.get("US_rate", 5.08))
            
            if ret_total > 0:
                if vix_lag1 > 14.0: ret_total *= vix_fac
                if spread < 0.8: ret_total *= spread_fac
                
            next_level = float(history_sim[-1] * math.exp(ret_total))
            preds_segment.append(next_level)
            history_sim.append(next_level)
            diffs_sim.append(next_level - history_sim[-2])
            
        errors_segment = np.array(actuals_segment) - np.array(preds_segment)
        for k in range(sim_len_segments - 4):
            t = start_idx + k
            row_exog = combined.iloc[t]
            target_bias = np.mean(errors_segment[k : k + 5])
            
            hist_window = preds_segment[max(0, k - 10) : k + 1]
            x_arr = np.arange(len(hist_window))
            current_level = preds_segment[k]
            slope = np.polyfit(x_arr, hist_window, 1)[0] / current_level if len(hist_window) >= 3 else 0.0
            curvature = np.polyfit(x_arr, hist_window, 2)[0] / current_level if len(hist_window) >= 3 else 0.0
            recent_direction = (preds_segment[k] - preds_segment[max(0, k-5)]) / preds_segment[max(0, k-5)] if len(preds_segment) >= 6 else 0.0
            
            feats_bias = {
                "forecast_age": float(k),
                "trend_slope": slope,
                "trend_curvature": curvature,
                "VIX": float(row_exog.get("VIX", 18.0)),
                "recent_forecast_direction": recent_direction,
                "GOLD_ret": float(row_exog.get("GOLD_ret", 0.0)),
                "SP500_ret": float(row_exog.get("SP500_ret", 0.0)),
                "target_bias": target_bias
            }
            bias_samples.append(feats_bias)
            
    df_bias = pd.DataFrame(bias_samples)
    stable_features = ["forecast_age", "trend_slope", "trend_curvature", "VIX", "recent_forecast_direction", "GOLD_ret", "SP500_ret"]
    bias_models = {}
    
    # Handle potentially small bias dataset in small splits
    regime_conditions = {
        "VIX_Low": df_bias["VIX"] < 14.0,
        "VIX_Med": (df_bias["VIX"] >= 14.0) & (df_bias["VIX"] <= 20.0),
        "VIX_High": df_bias["VIX"] > 20.0
    }
    
    for rname, condition in regime_conditions.items():
        sub = df_bias[condition]
        if len(sub) < 5:
            # Fallback to entire dataset if partition is empty to prevent crashes
            sub = df_bias
        model_b = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=100.0))
        ]).fit(sub[stable_features], sub["target_bias"])
        bias_models[rname] = model_b
        
    return bias_models, stable_features

# Main Pipeline Fitting Function
def fit_models(train_df, combined):
    """Fits shared Trend Model, Model A (Ridge Residual + Bias), and Model B (GRU Residual)."""
    # 1. Fit Ridge Trend Model
    X_trend, y_trend, start_idx = build_trend_dataset(train_df, combined, SELECTED_LAGS)
    trend_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    trend_pipeline.fit(X_trend, y_trend)
    trend_preds = trend_pipeline.predict(X_trend)
    
    # 2. Fit Model A: Ridge Residual Model
    X_res, y_res = build_residual_dataset(train_df, combined, start_idx, trend_preds, y_trend)
    res_ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0))
    ])
    res_ridge_pipeline.fit(X_res, y_res)
    
    # 3. Fit Bias Predictor Models
    trend_cols = list(trend_pipeline.feature_names_in_)
    res_cols = list(res_ridge_pipeline.feature_names_in_)
    bias_models, bias_features = fit_bias_models(
        train_df, combined, trend_pipeline, res_ridge_pipeline,
        BEST_VIX, BEST_SPREAD, SELECTED_LAGS, trend_cols, res_cols
    )
    
    # 4. Fit Model B: GRU Residual Model
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res.values)
    X_res_scaled_df = pd.DataFrame(X_res_scaled, columns=X_res.columns)
    
    time_steps = 5
    X_gru, y_gru = build_gru_dataset(X_res_scaled_df, y_res, time_steps=time_steps)
    
    # Train robust GRU model with regularization to prevent overfitting/instability
    gru_model = Sequential([
        GRU(16, input_shape=(time_steps, X_gru.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    gru_model.compile(optimizer=Adam(learning_rate=0.002), loss="mse")
    gru_model.fit(X_gru, y_gru, epochs=15, batch_size=32, verbose=0)
    
    return trend_pipeline, res_ridge_pipeline, bias_models, bias_features, gru_model, scaler, X_trend.columns

# Recursive Forecasting Pipeline (Pure Anti-Leakage Loop)
def recursive_forecast(train_df, test_df, combined, trend_model, res_model, bias_models, bias_features, model_type, trend_cols, scaler=None, time_steps=5):
    """Runs recursive time series prediction step-by-step applying gates and bias corrections."""
    history = train_df[TARGET_COL].astype(float).tolist()
    history_diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    
    if model_type == "GRU":
        exog_list = []
        for idx in range(len(combined)):
            row_exog = combined.iloc[idx]
            exog_list.append([float(row_exog.get(col, 0.0)) for col in EXOG_COLS])
        exog_arr = scaler.transform(exog_list)
        
    preds_base_list = []
    preds_final = []
    
    trend_cols = list(trend_cols)
    res_cols = ["SP500_ret_lag1", "VIX_ret_lag1", "bi_rate_change_lag10", "IHSG_ret_lag1", "OIL_ret_lag1"]
    
    scaler_bias_low = bias_models["VIX_Low"].named_steps["scaler"]
    ridge_bias_low = bias_models["VIX_Low"].named_steps["model"]
    scaler_bias_med = bias_models["VIX_Med"].named_steps["scaler"]
    ridge_bias_med = bias_models["VIX_Med"].named_steps["model"]
    scaler_bias_high = bias_models["VIX_High"].named_steps["scaler"]
    ridge_bias_high = bias_models["VIX_High"].named_steps["model"]
    
    for i in range(len(test_df)):
        idx = len(train_df) + i
        row_exog = combined.iloc[idx]
        
        # 1. Predict Trend Component
        feats_trend = build_row_features(row_exog, history, history_diffs, SELECTED_LAGS, [], "trend")
        X_row_trend = pd.DataFrame([feats_trend]).reindex(columns=trend_cols, fill_value=0.0)
        ret_trend = float(trend_model.predict(X_row_trend)[0])
        
        # 2. Predict Residual Component
        if model_type == "Ridge":
            feats_res = {col: float(row_exog.get(col, 0.0)) for col in EXOG_COLS}
            X_row_res = pd.DataFrame([feats_res]).reindex(columns=res_cols, fill_value=0.0)
            ret_shock = float(res_model.predict(X_row_res)[0])
        else: # GRU
            window_exog = exog_arr[idx - time_steps:idx]
            window_exog_3d = np.expand_dims(window_exog, axis=0)
            ret_shock = float(res_model.predict(window_exog_3d, verbose=0)[0, 0])
            
        ret_total = ret_trend + ret_shock
        
        # 3. Dynamic Asymmetrical Risk Gates
        vix_lag1 = float(row_exog.get("VIX_lag1", 15.0))
        spread = float(row_exog.get("BI_rate", 5.75)) - float(row_exog.get("US_rate", 5.08))
        
        if ret_total > 0:
            if vix_lag1 > 14.0: ret_total *= BEST_VIX
            if spread < 0.8: ret_total *= BEST_SPREAD
                
        # Clip return to avoid explosion
        ret_total = np.clip(ret_total, -0.05, 0.05)
                
        # Reconstruct base level
        pred_base = float(history[-1] * math.exp(ret_total))
        preds_base_list.append(pred_base)
        history.append(pred_base)
        history_diffs.append(pred_base - history[-2])
        
        # 4. Predict and Apply Bias Correction (applied to both to keep them optimized)
        hist_window = preds_base_list[max(0, i - 10) : i + 1]
        x_arr = np.arange(len(hist_window))
        slope = np.polyfit(x_arr, hist_window, 1)[0] / pred_base if len(hist_window) >= 3 else 0.0
        curvature = np.polyfit(x_arr, hist_window, 2)[0] / pred_base if len(hist_window) >= 3 else 0.0
        recent_direction = (preds_base_list[i] - preds_base_list[max(0, i-5)]) / preds_base_list[max(0, i-5)] if len(preds_base_list) >= 6 else 0.0
        
        bias_feats = {
            "forecast_age": float(i),
            "trend_slope": slope,
            "trend_curvature": curvature,
            "VIX": float(row_exog.get("VIX", 18.0)),
            "recent_forecast_direction": recent_direction,
            "GOLD_ret": float(row_exog.get("GOLD_ret", 0.0)),
            "SP500_ret": float(row_exog.get("SP500_ret", 0.0))
        }
        bias_vec = np.array([[bias_feats.get(col, 0.0) for col in bias_features]])
        
        if vix_lag1 < 14.0:
            bias_correction = float(ridge_bias_low.predict(scaler_bias_low.transform(bias_vec))[0])
        elif 14.0 <= vix_lag1 <= 20.0:
            bias_correction = float(ridge_bias_med.predict(scaler_bias_med.transform(bias_vec))[0])
        else:
            bias_correction = float(ridge_bias_high.predict(scaler_bias_high.transform(bias_vec))[0])
            
        preds_final.append(pred_base + BEST_BETA * bias_correction)
        
    return np.array(preds_final)

# Evaluation Routine for the Splits
def run_split_scenarios(train_full, combined):
    """Executes Model A, Model B, and Ensemble over 80/20, 70/30, and 60/40 splits of the train set."""
    split_scenarios = [
        ("Skenario 1 (80% Train - 20% Test)", 0.8),
        ("Skenario 2 (70% Train - 30% Test)", 0.7),
        ("Skenario 3 (60% Train - 40% Test)", 0.6)
    ]
    
    results = []
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    
    for i, (name, train_ratio) in enumerate(split_scenarios):
        print(f"\nRunning {name}...")
        split_idx = int(len(train_full) * train_ratio)
        
        train_part = train_full.iloc[:split_idx].reset_index(drop=True)
        test_part = train_full.iloc[split_idx:].reset_index(drop=True)
        
        # Prepare subset data
        combined_subset = prepare_features(train_part, test_part.drop(columns=[TARGET_COL]))
        
        # Train Models
        trend_model, res_ridge, bias_models, bias_features, res_gru, scaler, trend_cols = fit_models(train_part, combined_subset)
        
        # Forecast Model A: Ridge
        preds_a = recursive_forecast(train_part, test_part, combined_subset, trend_model, res_ridge, bias_models, bias_features, "Ridge", trend_cols)
        
        # Forecast Model B: GRU
        preds_b = recursive_forecast(train_part, test_part, combined_subset, trend_model, res_gru, bias_models, bias_features, "GRU", trend_cols, scaler)
        
        # Ensemble: Hybrid (Average of Model A and Model B)
        preds_ensemble = 0.5 * preds_a + 0.5 * preds_b
        
        y_true = test_part[TARGET_COL].astype(float).values
        
        # Calculate Metrics
        m_a = calculate_metrics(y_true, preds_a)
        m_b = calculate_metrics(y_true, preds_b)
        m_ens = calculate_metrics(y_true, preds_ensemble)
        
        # Store for rekapitulasi
        for model_name, metrics in [("Model A (Decoupled Ridge)", m_a), ("Model B (Deep GRU)", m_b), ("Ensemble (Hybrid)", m_ens)]:
            results.append({
                "Skenario": name,
                "Model": model_name,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE (%)": metrics["MAPE"],
                "R2": metrics["R2"]
            })
            
        # Plot predictions for this scenario showing history to prediction path
        hist_plot_len = min(150, len(train_part))
        hist_dates = train_part[DATE_COL].iloc[-hist_plot_len:]
        hist_vals = train_part[TARGET_COL].iloc[-hist_plot_len:]
        
        dates_test = test_part[DATE_COL]
        
        ax = axes[i]
        ax.plot(hist_dates, hist_vals, color="gray", label="Historical Train Data", alpha=0.7)
        ax.plot(dates_test, y_true, color="black", label="Actual Test Data", linewidth=2)
        ax.plot(dates_test, preds_a, color="blue", label=f"Model A (Ridge) [RMSE={m_a['RMSE']:.1f}]", linestyle="--")
        ax.plot(dates_test, preds_b, color="orange", label=f"Model B (GRU) [RMSE={m_b['RMSE']:.1f}]", linestyle=":")
        ax.plot(dates_test, preds_ensemble, color="red", label=f"Ensemble (Hybrid) [RMSE={m_ens['RMSE']:.1f}]", linewidth=2)
        
        ax.axvline(x=train_part[DATE_COL].iloc[-1], color="purple", linestyle="-.", label="Train-Test Split Line")
        ax.set_title(f"USDIDR Forecast Comparison: {name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("USDIDR Exchange Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("scenarios_comparison_plot.png", dpi=150)
    plt.show() # colab friendly display
    plt.close()
    
    # Save validation table summary
    df_results = pd.DataFrame(results)
    df_results.to_csv("validation_scenarios_summary.csv", index=False)
    print("\n=======================================================")
    print("REKAPITULASI HASIL FORECASTING DAN EVALUASI:")
    print("=======================================================")
    print(df_results.to_string(index=False))
    print("\nSplit scenarios evaluation plot saved and displayed as 'scenarios_comparison_plot.png'.")
    return df_results

# Main Reconstruct and Out-Of-Sample Kaggle Submission Logic
def main():
    print("=========================================================")
    print("USD/IDR FORECASTING & RECONSTRUCTION - GOOGLE COLAB SCRIPT")
    print("=========================================================\n")
    
    # Load dataset
    train_full, test_exog, test_actual = check_and_load_files()
    print(f"Loaded {len(train_full)} training rows and {len(test_exog)} test features.")
    if test_actual is not None:
        print(f"Loaded {len(test_actual)} actual test labels for final verification.")
    
    # 1. RUN BONUS SPLIT SCENARIOS
    print("\n--- STEP 1: Running Split Scenarios (80/20, 70/30, 60/40) ---")
    combined_full = prepare_features(train_full, test_exog)
    run_split_scenarios(train_full, combined_full)
    
    # 2. RUN RECONSTRUCTION OF BEST MODEL ON FULL TEST SET FOR SUBMISSION
    print("\n--- STEP 2: Reconstructing Best Model (Two-Stage Decoupled Ridge with Gating & Bias Correction) ---")
    # Fit models on the complete training set
    trend_model, res_ridge, bias_models, bias_features, _, _, trend_cols = fit_models(train_full, combined_full)
    
    # Forecast recursively into the Kaggle OOS period (Model A - Best Decoupled Ridge)
    test_preds = recursive_forecast(train_full, test_exog, combined_full, trend_model, res_ridge, bias_models, bias_features, "Ridge", trend_cols)
    
    # Save the submission csv file for Kaggle
    sub_df = pd.DataFrame({
        "Date": test_exog[DATE_COL].dt.strftime("%Y-%m-%d"),
        "USDIDR": test_preds
    })
    sub_df.to_csv("submission.csv", index=False)
    print("Kaggle Submission saved successfully to 'submission.csv'!")
    
    # Plot final predicted path alongside history
    plt.figure(figsize=(15, 6))
    hist_len = min(300, len(train_full))
    plt.plot(train_full[DATE_COL].iloc[-hist_len:], train_full[TARGET_COL].iloc[-hist_len:], color="gray", label="Historical USDIDR", alpha=0.7)
    
    if test_actual is not None:
        y_true_final = test_actual[TARGET_COL].astype(float).values
        final_metrics = calculate_metrics(y_true_final, test_preds)
        plt.plot(test_actual[DATE_COL], y_true_final, color="black", label="Actual Test USDIDR", linewidth=2)
        plt.plot(test_exog[DATE_COL], test_preds, color="red", label=f"Predicted USDIDR (RMSE={final_metrics['RMSE']:.2f})", linewidth=2.5)
        print("\n=======================================================")
        print("FINAL OUT-OF-SAMPLE TEST SET EVALUATION (MODEL A):")
        print("=======================================================")
        for k, v in final_metrics.items():
            print(f"  Final {k}: {v:.4f}")
    else:
        plt.plot(test_exog[DATE_COL], test_preds, color="red", label="Predicted USDIDR", linewidth=2.5)
        
    plt.axvline(x=train_full[DATE_COL].iloc[-1], color="purple", linestyle="-.", label="Forecast Origin Boundary")
    plt.title("USDIDR Out-Of-Sample Forecast: Reconstructed Best Model A (No Leakage)")
    plt.xlabel("Date")
    plt.ylabel("USDIDR Exchange Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("submission_predictions_plot.png", dpi=150)
    plt.show() # colab friendly display
    plt.close()
    print("Out-of-sample prediction path plot saved and displayed as 'submission_predictions_plot.png'.")

if __name__ == "__main__":
    main()
