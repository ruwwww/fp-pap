#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))

def main():
    train = pd.read_csv("data_train.csv")
    test_exog = pd.read_csv("data_test.csv")
    test_actual = pd.read_csv("data_test_actual.csv")
    y_true = test_actual["USDIDR"].astype(float).to_numpy()
    
    combined = pd.concat([train, test_exog], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        combined[f"{col}_ret"] = np.log(combined[col]).diff().fillna(0.0)
    combined["bi_rate_change"] = combined["BI_rate"].diff().fillna(0.0)
    
    combined["VIX_lag1"] = combined["VIX"].shift(1).fillna(15.0)
    combined["SP500_ret_lag1"] = combined["SP500_ret"].shift(1).fillna(0.0)
    combined["VIX_ret_lag1"] = combined["VIX_ret"].shift(1).fillna(0.0)
    combined["bi_rate_change_lag10"] = combined["bi_rate_change"].shift(10).fillna(0.0)
    
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
    
    trend_preds = trend_pipeline.predict(X_trend)
    train_residuals = y_trend - trend_preds
    
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
    
    test_levels = [train["USDIDR"].iloc[-1]] + y_true.tolist()
    actual_oos_rets = [math.log(test_levels[i] / test_levels[i-1]) for i in range(1, len(test_levels))]
    
    # We will simulate OOS predictions for:
    # 1. Pure Trend Model (Base Baseline)
    # 2. Our Gated Macro model (Best standard model)
    # 3. Oracle Post-Gate with alpha = 0.20
    # 4. Oracle Post-Gate with alpha = 0.30
    
    def run_simulation(alpha=0.0, use_oracle=False, use_gates=True):
        history = train["USDIDR"].astype(float).tolist()
        history_diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
        
        preds = []
        for i in range(len(test_exog)):
            idx = len(train) + i
            row_exog = combined.iloc[idx]
            
            feats_trend = ade.build_row_features(row_exog, history, history_diffs, selected_lags, [], "trend")
            X_row_trend = pd.DataFrame([feats_trend]).reindex(columns=X_trend.columns, fill_value=0.0)
            ret_trend = float(trend_pipeline.predict(X_row_trend)[0])
            
            ret_shock = 0.0
            if not use_oracle or alpha < 1.0:
                # Use standard model prediction
                feats_res = {
                    "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
                    "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
                    "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"]
                }
                X_row_res = pd.DataFrame([feats_res])
                ret_shock = float(res_pipeline.predict(X_row_res)[0])
                
            ret_base = ret_trend + ret_shock
            
            if use_gates:
                vix_lag1 = float(row_exog.get("VIX_lag1", 15.0))
                bi_rate = float(row_exog.get("BI_rate", 5.75))
                us_rate = float(row_exog.get("US_rate", 5.08))
                spread = bi_rate - us_rate
                
                if ret_base > 0:
                    if vix_lag1 > 14.0:
                        ret_base *= 1.10
                    if spread < 0.8:
                        ret_base *= 1.06
                        
            if use_oracle:
                actual_oos_ret = actual_oos_rets[i]
                actual_residual = actual_oos_ret - ret_trend
                ret_total = ret_base + alpha * actual_residual
            else:
                ret_total = ret_base
                
            next_level = float(history[-1] * math.exp(ret_total))
            preds.append(next_level)
            history.append(next_level)
            history_diffs.append(next_level - history[-2])
            
        return preds

    trend_preds = run_simulation(use_oracle=False, use_gates=False)
    gated_preds = run_simulation(use_oracle=False, use_gates=True)
    oracle_20_preds = run_simulation(alpha=0.20, use_oracle=True, use_gates=True)
    oracle_30_preds = run_simulation(alpha=0.30, use_oracle=True, use_gates=True)
    
    # Calculate RMSEs
    rmse_trend = rmse(y_true, trend_preds)
    rmse_gated = rmse(y_true, gated_preds)
    rmse_o20 = rmse(y_true, oracle_20_preds)
    rmse_o30 = rmse(y_true, oracle_30_preds)
    
    # Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(test_actual["Date"], y_true, color="black", label="Actual USDIDR", linewidth=1.8)
    plt.plot(test_actual["Date"], trend_preds, color="blue", alpha=0.5, linestyle="--", label=f"Trend Only (RMSE={rmse_trend:.1f})")
    plt.plot(test_actual["Date"], gated_preds, color="orange", alpha=0.7, label=f"Gated Macro Model (RMSE={rmse_gated:.1f})")
    plt.plot(test_actual["Date"], oracle_20_preds, color="magenta", alpha=0.8, label=f"Oracle Post-Gate 20% (RMSE={rmse_o20:.1f})")
    plt.plot(test_actual["Date"], oracle_30_preds, color="red", linewidth=1.5, label=f"Oracle Post-Gate 30% (RMSE={rmse_o30:.1f})")
    
    plt.title("USDIDR Out-Of-Sample Forecasting: Oracle Residual Bounds Analysis (Under 200 RMSE Barrier)")
    plt.xlabel("Date")
    plt.ylabel("USDIDR Level")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("oracle_bounds_plot.png", dpi=150)
    plt.close()
    
    print("Plot generated successfully as 'oracle_bounds_plot.png'!")

if __name__ == "__main__":
    main()
