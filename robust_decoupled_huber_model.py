#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

def prepare_combined_data(train, test_exog):
    combined = pd.concat([train, test_exog], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    
    # 1. Stationary Log-Returns for Macro variables
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        combined[f"{col}_ret"] = np.log(combined[col]).diff().fillna(0.0)
    combined["bi_rate_change"] = combined["BI_rate"].diff().fillna(0.0)
    
    # 2. Apply Causal Lags matching Step 2 (Feature Engineering)
    combined["SP500_ret_lag1"] = combined["SP500_ret"].shift(1).fillna(0.0)
    combined["VIX_ret_lag1"] = combined["VIX_ret"].shift(1).fillna(0.0)
    combined["bi_rate_change_lag10"] = combined["bi_rate_change"].shift(10).fillna(0.0)
    
    return combined

def main():
    train, test_exog, test_actual = load_data()
    y_true = test_actual["USDIDR"].astype(float).to_numpy()
    
    combined = prepare_combined_data(train, test_exog)
    
    # Trend Model setup using selected PACF Lags
    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]
    
    levels = train["USDIDR"].astype(float).tolist()
    diffs = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
    
    # 1. Fit Trend Model (Ridge alpha=1.0)
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
    
    # Calculate Training Residuals (De-trended log-returns)
    trend_preds = trend_pipeline.predict(X_trend)
    train_residuals = y_trend - trend_preds
    
    # 2. Build de-trend features (including both macro features and endogenous memory lags)
    # Fitur Endogen: Lag_1, Lag_2, Lag_3, Lag_6 dari train_residuals
    X_de_trend_rows = []
    y_de_trend = []
    
    res_list = train_residuals.tolist()
    
    # Offset by 6 to build lags
    for idx, t in enumerate(range(start_idx + 6, len(train))):
        res_idx = idx + 6
        row_exog = combined.iloc[t]
        
        feats = {
            "lag_1": res_list[res_idx - 1],
            "lag_2": res_list[res_idx - 2],
            "lag_3": res_list[res_idx - 3],
            "lag_6": res_list[res_idx - 6],
            "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
            "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
            "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"]
        }
        X_de_trend_rows.append(feats)
        y_de_trend.append(res_list[res_idx])
        
    X_de_trend = pd.DataFrame(X_de_trend_rows)
    y_de_trend = pd.Series(y_de_trend)
    
    # 3. Fit De-trended Model using Huber Regressor (Robust to Fat Tails)
    de_trend_huber = Pipeline([
        ("scaler", StandardScaler()),
        ("model", HuberRegressor(epsilon=1.35, alpha=10.0))
    ])
    de_trend_huber.fit(X_de_trend, y_de_trend)
    
    # 4. Out-of-Sample Recursive Forecast Loop
    def run_forecast():
        history_level = train["USDIDR"].astype(float).tolist()
        history_diffs = [history_level[i] - history_level[i - 1] for i in range(1, len(history_level))]
        
        # Buffer of de-trended residuals for endogenous memory lag prediction
        hist_de_trend_res = train_residuals.tolist()
        
        preds = []
        
        for i in range(len(test_exog)):
            idx = len(train) + i
            row_exog = combined.iloc[idx]
            
            # Step A: Predict long-term Trend Log-Return
            feats_trend = ade.build_row_features(row_exog, history_level, history_diffs, selected_lags, [], "trend")
            X_row_trend = pd.DataFrame([feats_trend]).reindex(columns=X_trend.columns, fill_value=0.0)
            ret_trend = float(trend_pipeline.predict(X_row_trend)[0])
            
            # Step B: Predict short-term de-trended shock using Huber (Robust)
            feats_de_trend = {
                "lag_1": hist_de_trend_res[-1],
                "lag_2": hist_de_trend_res[-2],
                "lag_3": hist_de_trend_res[-3],
                "lag_6": hist_de_trend_res[-6],
                "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
                "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
                "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"]
            }
            X_row_de_trend = pd.DataFrame([feats_de_trend])
            ret_shock = float(de_trend_huber.predict(X_row_de_trend)[0])
            
            # Step C: Reconstruct total log-return
            ret_total = ret_trend + ret_shock
            
            # Step D: Apply Risk Gates (VIX & Spread)
            vix_lag1 = float(row_exog.get("VIX_lag1", 18.0))
            bi_rate = float(row_exog.get("BI_rate", 5.75))
            us_rate = float(row_exog.get("US_rate", 5.08))
            spread = bi_rate - us_rate
            
            if ret_total > 0:
                if vix_lag1 > 14.0:
                    ret_total *= 1.10
                if spread < 0.8:
                    ret_total *= 1.06
                    
            next_level = float(history_level[-1] * math.exp(ret_total))
            preds.append(next_level)
            
            # Update history level for recursive loops
            history_level.append(next_level)
            history_diffs.append(next_level - history_level[-2])
            
            # Calculate detrended log-return prediction error to update the residual memory buffer causal lag
            actual_ret_pred = math.log(next_level / history_level[-2])
            hist_de_trend_res.append(actual_ret_pred - ret_trend)
            
        return preds

    huber_preds = run_forecast()
    huber_rmse = rmse(y_true, huber_preds)
    print(f"Two-Stage Decoupled Ridge-Huber Model OOS RMSE: {huber_rmse:.4f}")

if __name__ == "__main__":
    main()
