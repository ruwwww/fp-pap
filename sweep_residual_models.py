#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import math
import warnings
warnings.filterwarnings("ignore")

# Define RMSE
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))

# Load data
train = pd.read_csv("data_train.csv")
test_exog = pd.read_csv("data_test.csv")
test_actual = pd.read_csv("data_test_actual.csv")

y_true = test_actual["USDIDR"].astype(float).to_numpy()

# 1. Prepare Features & Log-Returns
combined = pd.concat([train, test_exog], ignore_index=True)
combined["Date"] = pd.to_datetime(combined["Date"])

# Create stationary macro returns
for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
    combined[f"{col}_ret"] = np.log(combined[col]).diff().fillna(0.0)
combined["bi_rate_change"] = combined["BI_rate"].diff().fillna(0.0)

# Generate lags for exogenous features
exog_base = [f"{col}_ret" for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]] + ["bi_rate_change"]
for col in exog_base:
    combined[f"{col}_lag1"] = combined[col].shift(1).fillna(0.0)
    combined[f"{col}_lag2"] = combined[col].shift(2).fillna(0.0)

# Trend base model setup
selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]

# Prepare training tables
levels = train["USDIDR"].astype(float).tolist()
diffs = [levels[i] - levels[i - 1] for i in range(1, len(levels))]

# Build trend model features (Ridge alpha=1.0)
import assumption_driven_experiment as ade
X_trend_rows = []
y_trend = []
start_idx = max(max(selected_lags), 252)

for t in range(start_idx, len(train)):
    row_exog = combined.iloc[t]
    feats = ade.build_row_features(row_exog, levels[:t], diffs[:t-1], selected_lags, [], "trend")
    X_trend_rows.append(feats)
    y_trend.append(float(math.log(levels[t] / levels[t - 1])))

X_trend = pd.DataFrame(X_trend_rows).fillna(0.0)
y_trend = pd.Series(y_trend)

trend_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])
trend_pipeline.fit(X_trend, y_trend)

# Generate training residuals (de-trended returns)
trend_preds = trend_pipeline.predict(X_trend)
train_residuals = y_trend - trend_preds

# Build residual dataset (X_res, y_res)
res_rows = []
for idx, t in enumerate(range(start_idx, len(train))):
    row_exog = combined.iloc[t]
    # Build a rich feature set for residual learning: Lags 1 & 2 of macro, plus trend predictions
    feats = {
        "trend_pred": trend_preds[idx],
        "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
        "SP500_ret_lag2": row_exog["SP500_ret_lag2"],
        "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
        "VIX_ret_lag2": row_exog["VIX_ret_lag2"],
        "bi_rate_change_lag1": row_exog["bi_rate_change_lag1"],
        "bi_rate_change_lag2": row_exog["bi_rate_change_lag2"],
        # Interaction terms
        "vix_x_sp500": row_exog["VIX_ret_lag1"] * row_exog["SP500_ret_lag1"],
    }
    res_rows.append(feats)

X_res = pd.DataFrame(res_rows)
y_res = pd.Series(train_residuals)

# --- Define Different Residual Shock Predictors ---

# 1. Ridge Baseline (alpha=10.0)
ridge_res = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
ridge_res.fit(X_res, y_res)

# 2. LightGBM (Non-linear interactions)
lgb_res = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.03, random_state=42, verbose=-1)
lgb_res.fit(X_res, y_res)

# 3. LSTM / RNN approach for sequential residual modeling
# Let's create sequential inputs for LSTM
def build_lstm_data(series_res, X_res_df, time_steps=10):
    Xs, ys = [], []
    # Include features in sequential format
    scaled_feats = StandardScaler().fit_transform(X_res_df)
    for i in range(len(series_res) - time_steps):
        Xs.append(scaled_feats[i:(i + time_steps)])
        ys.append(series_res.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 5
X_lstm_train, y_lstm_train = build_lstm_data(y_res, X_res, time_steps=time_steps)

lstm_model = Sequential([
    LSTM(16, activation='tanh', input_shape=(time_steps, X_res.shape[1])),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_lstm_train, y_lstm_train, epochs=15, batch_size=64, verbose=0)

# --- Evaluation Loop ---
def run_forecast(res_type="ridge"):
    history = train["USDIDR"].astype(float).tolist()
    diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    
    preds = []
    
    # Keep rolling buffer of recent test features for LSTM sequential forecasting
    recent_features = X_res.tail(time_steps).values.tolist()
    
    for i in range(len(test_exog)):
        idx = len(train) + i
        row_exog = combined.iloc[idx]
        
        # 1. Trend pred
        feats_trend = ade.build_row_features(row_exog, history, diffs, selected_lags, [], "trend")
        X_row_trend = pd.DataFrame([feats_trend]).reindex(columns=X_trend.columns, fill_value=0.0)
        ret_trend = float(trend_pipeline.predict(X_row_trend)[0])
        
        # 2. Residual shock pred
        feat_res_dict = {
            "trend_pred": ret_trend,
            "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
            "SP500_ret_lag2": row_exog["SP500_ret_lag2"],
            "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
            "VIX_ret_lag2": row_exog["VIX_ret_lag2"],
            "bi_rate_change_lag1": row_exog["bi_rate_change_lag1"],
            "bi_rate_change_lag2": row_exog["bi_rate_change_lag2"],
            "vix_x_sp500": row_exog["VIX_ret_lag1"] * row_exog["SP500_ret_lag1"],
        }
        
        X_row_res = pd.DataFrame([feat_res_dict])
        
        if res_type == "ridge":
            ret_shock = float(ridge_res.predict(X_row_res)[0])
        elif res_type == "lgb":
            ret_shock = float(lgb_res.predict(X_row_res)[0])
        elif res_type == "lstm":
            # Scale feature
            scaler = StandardScaler().fit(X_res)
            scaled_row = scaler.transform(X_row_res)[0]
            # Update recent features sequence
            seq_input = np.array(recent_features[-time_steps:])
            # Predict
            ret_shock = float(lstm_model.predict(seq_input.reshape(1, time_steps, -1), verbose=0)[0][0])
            # Append new feature row to sequence history
            recent_features.append(scaled_row.tolist())
            
        ret_total = ret_trend + ret_shock
        
        # Apply Risk Gates
        vix_lag1 = float(row_exog.get("VIX_lag1", 18.0))
        bi_rate = float(row_exog.get("BI_rate", 5.75))
        us_rate = float(row_exog.get("US_rate", 5.08))
        spread = bi_rate - us_rate
        
        if ret_total > 0:
            if vix_lag1 > 14.0:
                ret_total *= 1.10
            if spread < 0.8:
                ret_total *= 1.06
                
        next_level = float(history[-1] * math.exp(ret_total))
        preds.append(next_level)
        history.append(next_level)
        diffs.append(next_level - history[-2])
        
    return preds

print("Evaluating setups...")
ridge_preds = run_forecast("ridge")
lgb_preds = run_forecast("lgb")
lstm_preds = run_forecast("lstm")

print(f"Ridge Residual Model RMSE: {rmse(y_true, ridge_preds):.4f}")
print(f"LightGBM Residual Model RMSE: {rmse(y_true, lgb_preds):.4f}")
print(f"LSTM Sequential Residual Model RMSE: {rmse(y_true, lstm_preds):.4f}")

# Save results
pd.DataFrame({
    "model": ["Ridge Residual", "LightGBM Residual", "LSTM Residual"],
    "rmse": [rmse(y_true, ridge_preds), rmse(y_true, lgb_preds), rmse(y_true, lstm_preds)]
}).to_csv("sweep_residual_results.csv", index=False)
