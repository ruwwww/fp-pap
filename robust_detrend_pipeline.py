#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import math
import warnings
warnings.filterwarnings("ignore")

# Define RMSE
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))

def load_data():
    train = pd.read_csv("data_train.csv")
    test_exog = pd.read_csv("data_test.csv")
    test_actual = pd.read_csv("data_test_actual.csv")
    return train, test_exog, test_actual

def prepare_features(train, test_exog):
    combined = pd.concat([train, test_exog], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    
    # Exogenous log-returns
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        combined[f"{col}_ret"] = np.log(combined[col]).diff().fillna(0.0)
    combined["bi_rate_change"] = combined["BI_rate"].diff().fillna(0.0)
    
    # Lag Exogenous
    combined["SP500_ret_lag1"] = combined["SP500_ret"].shift(1).fillna(0.0)
    combined["VIX_ret_lag1"] = combined["VIX_ret"].shift(1).fillna(0.0)
    combined["bi_rate_change_lag10"] = combined["bi_rate_change"].shift(10).fillna(0.0)
    
    return combined

def main():
    train, test_exog, test_actual = load_data()
    y_true = test_actual["USDIDR"].astype(float).to_numpy()
    
    combined = prepare_features(train, test_exog)
    
    # 1. Fit Smooth Linear Trend on Train
    x_train = np.arange(len(train))
    slope, intercept = np.polyfit(x_train, train["USDIDR"], 1)
    
    # Generate Trend Level
    train_trend = slope * x_train + intercept
    
    # Get Detrended Log-Returns on Training Data
    train_log_ret = np.log(train["USDIDR"]).diff().fillna(0.0)
    train_mean_ret = train_log_ret.mean()
    train_detrended_ret = train_log_ret - train_mean_ret
    
    # 2. Build feature matrix for Detrended Return Prediction
    # Lag Endogenous (Lag 1, 2, 3, 6 of detrended target return)
    # We will build rows dynamically
    X_rows = []
    y_target = []
    
    start_idx = 30 # Safe buffer for lags
    for t in range(start_idx, len(train)):
        row_exog = combined.iloc[t]
        hist_ret = train_detrended_ret.iloc[:t].tolist()
        
        feats = {
            "lag_1": hist_ret[-1],
            "lag_2": hist_ret[-2],
            "lag_3": hist_ret[-3],
            "lag_6": hist_ret[-6],
            "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
            "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
            "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"]
        }
        X_rows.append(feats)
        y_target.append(train_detrended_ret.iloc[t])
        
    X_train = pd.DataFrame(X_rows)
    y_train = pd.Series(y_target)
    
    # 3. Fit Models with Robust Loss Function (Huber / MAE)
    # LightGBM with Huber Loss
    lgb_model = lgb.LGBMRegressor(
        objective="huber",
        alpha=0.9, # Huber threshold
        n_estimators=100,
        max_depth=4,
        learning_rate=0.03,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # LSTM with Huber Loss
    def build_lstm_data(y_series, X_df, time_steps=5):
        Xs, ys = [], []
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X_df)
        for i in range(len(y_series) - time_steps):
            Xs.append(scaled_X[i:(i + time_steps)])
            ys.append(y_series.iloc[i + time_steps])
        return np.array(Xs), np.array(ys), scaler
    
    time_steps = 5
    X_lstm, y_lstm, scaler = build_lstm_data(y_train, X_train, time_steps=time_steps)
    
    lstm_model = Sequential([
        LSTM(16, activation="tanh", input_shape=(time_steps, X_train.shape[1])),
        Dense(1)
    ])
    # Use tensorflow huber loss
    lstm_model.compile(optimizer="adam", loss="huber")
    lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=64, verbose=0)
    
    # 4. Out-of-Sample Recursive Forecasting Loop
    def forecast_recursive(model_type="lgb"):
        # We start recursive loop from end of train
        # We need historical actual detrended return buffer
        hist_detrend_ret = train_detrended_ret.tolist()
        history_level = train["USDIDR"].astype(float).tolist()
        
        preds = []
        
        # LSTM feature buffer
        scaled_train_X = scaler.transform(X_train)
        recent_scaled_features = scaled_train_X[-time_steps:].tolist()
        
        for i in range(len(test_exog)):
            idx = len(train) + i
            row_exog = combined.iloc[idx]
            
            # Construct feature row using recursive history
            feats = {
                "lag_1": hist_detrend_ret[-1],
                "lag_2": hist_detrend_ret[-2],
                "lag_3": hist_detrend_ret[-3],
                "lag_6": hist_detrend_ret[-6],
                "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
                "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
                "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"]
            }
            
            X_row = pd.DataFrame([feats])
            
            if model_type == "lgb":
                pred_detrend_ret = float(lgb_model.predict(X_row)[0])
            elif model_type == "lstm":
                scaled_row = scaler.transform(X_row)[0]
                seq_input = np.array(recent_scaled_features[-time_steps:])
                pred_detrend_ret = float(lstm_model.predict(seq_input.reshape(1, time_steps, -1), verbose=0)[0][0])
                recent_scaled_features.append(scaled_row.tolist())
                
            # Reconstruct total log-return (predicted detrend return + historical mean return)
            total_log_ret = pred_detrend_ret + train_mean_ret
            
            # Level prediction
            next_level = history_level[-1] * math.exp(total_log_ret)
            
            # Save predictions and update loop
            preds.append(next_level)
            history_level.append(next_level)
            
            # Calculate detrended log-return of our predicted level to feed into the next lag step
            pred_log_ret = math.log(next_level / history_level[-2])
            pred_detrend_ret_actual = pred_log_ret - train_mean_ret
            hist_detrend_ret.append(pred_detrend_ret_actual)
            
        return np.array(preds)

    print("Running forecasting configurations...")
    lgb_preds = forecast_recursive("lgb")
    lstm_preds = forecast_recursive("lstm")
    
    lgb_rmse = rmse(y_true, lgb_preds)
    lstm_rmse = rmse(y_true, lstm_preds)
    
    print(f"1. Gated/Robust LightGBM (Huber Loss) OOS RMSE: {lgb_rmse:.4f}")
    print(f"2. Gated/Robust LSTM (Huber Loss) OOS RMSE: {lstm_rmse:.4f}")

if __name__ == "__main__":
    main()
