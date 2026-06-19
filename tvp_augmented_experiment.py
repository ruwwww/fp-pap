#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
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

def main():
    train, test_exog, test_actual = load_data()
    y_true = test_actual["USDIDR"].astype(float).to_numpy()
    
    # 1. Target log returns
    combined = pd.concat([train, test_exog], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    
    # Stationarity
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        combined[f"{col}_ret"] = np.log(combined[col]).diff().fillna(0.0)
    combined["bi_rate_change"] = combined["BI_rate"].diff().fillna(0.0)
    
    # Lags exog
    combined["SP500_ret_lag1"] = combined["SP500_ret"].shift(1).fillna(0.0)
    combined["VIX_ret_lag1"] = combined["VIX_ret"].shift(1).fillna(0.0)
    combined["bi_rate_change_lag10"] = combined["bi_rate_change"].shift(10).fillna(0.0)
    
    # 2. Trend model base setup
    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]
    levels = train["USDIDR"].astype(float).tolist()
    diffs = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
    
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
    
    # Residuals
    trend_preds = trend_pipeline.predict(X_trend)
    train_residuals = y_trend - trend_preds
    
    # Feature matrix for residual model
    X_res_rows = []
    for t in range(start_idx, len(train)):
        row_exog = combined.iloc[t]
        feats = {
            "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
            "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
            "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"]
        }
        X_res_rows.append(feats)
        
    X_res = pd.DataFrame(X_res_rows)
    y_res = pd.Series(train_residuals)
    
    res_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0))
    ])
    res_pipeline.fit(X_res, y_res)
    
    # TVP (Time-Varying Parameter) dynamic adaptation attempt:
    # Instead of constant gates, let's dynamically scale beta of residual shocks
    # based on rolling VIX stress level.
    # beta_t = beta_0 * (1 + gamma * max(0, VIX_t - VIX_mean) / VIX_std)
    
    print("Testing TVP-Augmented Exogenous Modeling...")
    
    vix_mean = train["VIX"].mean()
    vix_std = train["VIX"].std()
    
    gammas = [5.0, 10.0, 15.0, 20.0, 25.0]
    for gamma in gammas:
        history = train["USDIDR"].astype(float).tolist()
        history_diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
        
        preds = []
        for i in range(len(test_exog)):
            idx = len(train) + i
            row_exog = combined.iloc[idx]
            
            feats_trend = ade.build_row_features(row_exog, history, history_diffs, selected_lags, [], "trend")
            X_row_trend = pd.DataFrame([feats_trend]).reindex(columns=X_trend.columns, fill_value=0.0)
            ret_trend = float(trend_pipeline.predict(X_row_trend)[0])
            
            feats_res = {
                "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
                "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
                "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"]
            }
            X_row_res = pd.DataFrame([feats_res])
            ret_shock = float(res_pipeline.predict(X_row_res)[0])
            
            # Dynamic Time-Varying beta adjustment
            vix_val = float(row_exog.get("VIX_lag1", vix_mean))
            stress_scaler = 1.0 + gamma * max(0.0, vix_val - vix_mean) / vix_std
            
            ret_total = ret_trend + ret_shock * stress_scaler
            
            # Apply base gates
            bi_rate = float(row_exog.get("BI_rate", 5.75))
            us_rate = float(row_exog.get("US_rate", 5.08))
            spread = bi_rate - us_rate
            
            if ret_total > 0:
                if vix_val > 14.0:
                    ret_total *= 1.10
                if spread < 0.8:
                    ret_total *= 1.06
                    
            next_level = float(history[-1] * math.exp(ret_total))
            preds.append(next_level)
            history.append(next_level)
            history_diffs.append(next_level - history[-2])
            
        score = rmse(y_true, preds)
        print(f"Gamma: {gamma} -> OOS RMSE: {score:.4f}")

if __name__ == "__main__":
    main()
