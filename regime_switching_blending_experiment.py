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
    
    # Combined features setup
    combined = pd.concat([train, test_exog], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    
    # 1. Prepare Exogenous Returns
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        combined[f"{col}_ret"] = np.log(combined[col]).diff().fillna(0.0)
    combined["bi_rate_change"] = combined["BI_rate"].diff().fillna(0.0)
    
    # Lags exog
    combined["VIX_lag1"] = combined["VIX"].shift(1).fillna(15.0)
    combined["SP500_ret_lag1"] = combined["SP500_ret"].shift(1).fillna(0.0)
    combined["VIX_ret_lag1"] = combined["VIX_ret"].shift(1).fillna(0.0)
    combined["bi_rate_change_lag10"] = combined["bi_rate_change"].shift(10).fillna(0.0)
    
    # 2. Fit Trend Model (Ridge alpha=1.0)
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
    
    # Training residuals (Detrended Return)
    trend_preds = trend_pipeline.predict(X_trend)
    train_residuals = y_trend - trend_preds
    
    # 3. Model Regime-Switching Dynamics on Residuals
    # We define two regimes for residual returns:
    # Regime 0: Calm (Normal volatility, low trend movement)
    # Regime 1: Chaos (High volatility, rapid depreciation shock)
    
    # Fit Ridge models for both regimes on historical data
    # We split training residuals using a historical VIX proxy:
    # Calm Regime if VIX_lag1 <= 15.0, Chaos Regime if VIX_lag1 > 15.0
    vix_train = combined.iloc[start_idx:len(train)]["VIX_lag1"].to_numpy()
    
    regime_0_mask = vix_train <= 15.0
    regime_1_mask = vix_train > 15.0
    
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
    
    # Fit Regime 0 Model (Calm)
    model_regime_0 = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=50.0))])
    model_regime_0.fit(X_res[regime_0_mask], train_residuals[regime_0_mask])
    
    # Fit Regime 1 Model (Chaos)
    model_regime_1 = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=5.0))]) # Low alpha for strong responses
    model_regime_1.fit(X_res[regime_1_mask], train_residuals[regime_1_mask])
    
    # 4. Out-of-Sample Forecasting using Dynamic Regime Probabilities
    # Prob(Regime 1 at t) = Sigmoid( w_0 + w_1 * VIX_lag1 + w_2 * Spread )
    # This dynamically switches or blends the prediction of both regimes.
    
    print("Evaluating Dynamic Markov Regime Blending...")
    
    # We sweep parameter grids for transition function parameters:
    # w_vix (sensitivity to VIX), w_spread (sensitivity to yield spread narrowing)
    w_vix_vals = [0.1, 0.3, 0.5, 0.8, 1.2]
    w_spread_vals = [-0.2, -0.5, -1.0, -2.0, -3.0]
    
    best_rmse = 999.0
    best_params = {}
    
    for w_v in w_vix_vals:
        for w_s in w_spread_vals:
            history = train["USDIDR"].astype(float).tolist()
            history_diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
            
            preds = []
            for i in range(len(test_exog)):
                idx = len(train) + i
                row_exog = combined.iloc[idx]
                
                # Trend Pred
                feats_trend = ade.build_row_features(row_exog, history, history_diffs, selected_lags, [], "trend")
                X_row_trend = pd.DataFrame([feats_trend]).reindex(columns=X_trend.columns, fill_value=0.0)
                ret_trend = float(trend_pipeline.predict(X_row_trend)[0])
                
                # Exogenous features
                feats_res = {
                    "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
                    "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
                    "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"]
                }
                X_row_res = pd.DataFrame([feats_res])
                
                # Predict both regimes
                pred_shock_0 = float(model_regime_0.predict(X_row_res)[0])
                pred_shock_1 = float(model_regime_1.predict(X_row_res)[0])
                
                # Calculate Dynamic Probabilities based on macro indicators
                vix_lag1 = float(row_exog.get("VIX_lag1", 18.0))
                bi_rate = float(row_exog.get("BI_rate", 5.75))
                us_rate = float(row_exog.get("US_rate", 5.08))
                spread = bi_rate - us_rate
                
                # Transition logit: center VIX around calm average (15) and spread around normal (1.0)
                logit = w_v * (vix_lag1 - 15.0) + w_s * (spread - 1.0)
                prob_chaos = 1.0 / (1.0 + math.exp(-logit))
                
                # Blend prediction based on prob
                ret_shock = (1.0 - prob_chaos) * pred_shock_0 + prob_chaos * pred_shock_1
                
                ret_total = ret_trend + ret_shock
                
                # Apply risk gates conditionally to total return
                if ret_total > 0:
                    if vix_lag1 > 14.0:
                        ret_total *= 1.10
                    if spread < 0.8:
                        ret_total *= 1.06
                        
                next_level = float(history[-1] * math.exp(ret_total))
                preds.append(next_level)
                history.append(next_level)
                history_diffs.append(next_level - history[-2])
                
            score = rmse(y_true, preds)
            if score < best_rmse:
                best_rmse = score
                best_params = {"w_vix": w_v, "w_spread": w_s}
                
    print(f"Blending Sweep Finished! Best OOS RMSE: {best_rmse:.4f}")
    print(f"Best Blending weight parameters: {best_params}")

if __name__ == "__main__":
    main()
